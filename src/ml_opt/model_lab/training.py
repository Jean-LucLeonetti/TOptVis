"""Train all model families using identical case-separated snapshot provenance."""

from dataclasses import asdict
from datetime import datetime,timezone
import json
from pathlib import Path
import time

import numpy as np
from scipy.linalg import eigh
import torch

from ..config import CantileverConfig
from ..fem.element import q4_stiffness
from ..rom.operators import MAX_OPERATOR_BYTES,prepare_element_operators
from ..rom.training import pod_basis,save_metadata
from .config import POD_FAMILIES,SIGNATURE,TrainOptions
from .data import features,file_hash,fixed_features,full_targets,load_dataset,split_cases
from .networks import make_network,predict_network


def device_for(choice):
    if choice=='cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA is unavailable in this PyTorch environment; choose CPU or Auto')
    return 'cuda' if choice in ('auto','cuda') and torch.cuda.is_available() else 'cpu'


def checkpoint(directory,metadata,payload):
    path=directory/'model.pt'; temporary=directory/'model.tmp'
    torch.save({'schema_version':1,'id':metadata['id'],'options':metadata['options'],
        'config':metadata['config'],'source_sha256':metadata['source_sha256'],
        'displacement_scale':metadata.get('displacement_scale',1.),'rank':metadata.get('rank'),
        **payload},temporary)
    temporary.replace(path)
    metadata['model_sha256']=file_hash(path)
    metadata['checkpoint_ready']=True
    save_metadata(directory,metadata)


def energy_basis(snapshots,mesh,config,options):
    lifted=np.zeros((mesh.dof_count,snapshots.shape[1])); lifted[mesh.free_dofs]=snapshots
    element=lifted[mesh.element_dofs]
    stiffness=q4_stiffness(config.length/config.nx,config.height/config.ny,config.poisson_ratio,config.thickness)
    forces=stiffness @ element
    gram=np.einsum('eis,eit->st',element,forces,optimize=True)
    norms=np.sqrt(np.maximum(np.diag(gram),np.finfo(float).tiny))
    gram=gram/norms[:,None]/norms[None,:]
    values,vectors=eigh(.5*(gram+gram.T))
    order=np.argsort(values)[::-1]; values=values[order]; vectors=vectors[:,order]
    positive=values>np.finfo(float).eps*max(snapshots.shape)*values[0]
    values=values[positive]; vectors=vectors[:,positive]
    cumulative=np.cumsum(values)/np.sum(values)
    rank=min(options.max_modes,len(values),int(np.searchsorted(cumulative,options.energy_target))+1)
    vectors=(snapshots/norms) @ (vectors[:,:rank]/np.sqrt(values[:rank]))
    basis,_=np.linalg.qr(vectors)
    return basis,{'rank':rank,'captured_energy':float(cumulative[rank-1]),
        'metric':'Full-solid stiffness energy inner product; unit energy normalisation'}


def clusters(fields,count,seed):
    unique=np.unique(fields,axis=0)
    count=min(count,len(unique))
    centers=unique[np.random.default_rng(seed).choice(len(unique),count,replace=False)].copy()
    for _ in range(30):
        distance=np.mean((fields[:,None]-centers[None])**2,axis=2)
        labels=distance.argmin(axis=1)
        updated=np.array([fields[labels==i].mean(axis=0) if np.any(labels==i)
            else fields[np.argmax(distance.min(axis=1))] for i in range(count)])
        if np.allclose(updated,centers,atol=1e-10): break
        centers=updated
    distance=np.mean((fields[:,None]-centers[None])**2,axis=2)
    labels=distance.argmin(axis=1)
    present=np.unique(labels)
    return centers[present],np.searchsorted(present,labels)


def fit_pod(dataset,columns,options,emit,stopped):
    config=dataset['cases'][0]; mesh=dataset['mesh']
    fields=np.log(dataset['moduli'][:,columns].T/config.young_modulus)
    if options.family=='local_pod': centers,labels=clusters(fields,options.clusters,options.seed)
    else: centers,labels=np.array([fields.mean(axis=0)]),np.zeros(len(columns),dtype=int)
    bases,operators,diagnostics=[],[],[]
    total_bytes=0
    for cluster in range(len(centers)):
        if stopped(): return None
        emit({'event':'ml_phase','message':f'Building basis and reduced operators · cluster {cluster+1}/{len(centers)}'})
        selected=columns[labels==cluster]
        snapshots=dataset['displacements'][:,selected]
        if options.family=='energy_pod': basis,summary=energy_basis(snapshots,mesh,config,options)
        else:
            basis,_,_,summary=pod_basis(snapshots,options.energy_target,options.max_modes,True)
        size=mesh.element_count*basis.shape[1]*(basis.shape[1]+1)//2*8
        total_bytes+=size
        if total_bytes>MAX_OPERATOR_BYTES:
            raise ValueError('Combined model operators exceed 512 MiB; reduce modes or clusters')
        packed=prepare_element_operators(config,basis,stop_requested=stopped)
        if packed is None: return None
        bases.append(torch.from_numpy(basis.copy()))
        operators.append(torch.from_numpy(packed.copy()))
        diagnostics.append(summary)
    return dict(bases=bases,operators=operators,centers=torch.from_numpy(centers)),diagnostics,total_bytes


def physics_loss(predicted,reference,moduli,stiffness,element_dofs,forces):
    flattened=predicted.permute(0,2,3,1).reshape(len(predicted),-1)
    target=reference.permute(0,2,3,1).reshape(len(reference),-1)
    element_error=(flattened-target)[:,element_dofs]
    target_element=target[:,element_dofs]
    energy_error=torch.einsum('bei,ij,bej->be',element_error,stiffness,element_error)
    energy_target=torch.einsum('bei,ij,bej->be',target_element,stiffness,target_element)
    energy=((moduli*energy_error).sum(1)/(moduli*energy_target).sum(1).clamp_min(1e-16)).mean()
    internal=moduli[:,:,None]*(flattened[:,element_dofs] @ stiffness)
    global_force=torch.zeros_like(flattened).scatter_add(1,
        element_dofs.flatten()[None].expand(len(flattened),-1),internal.flatten(1))
    # Reactions at the clamped left edge are not part of free equilibrium.
    free_mask=forces[:,2].flatten(1).repeat_interleave(2,dim=1)==0
    residual=(((global_force-forces[:,:2].permute(0,2,3,1).reshape_as(flattened))*free_mask)**2).sum(1).mean()
    return energy,residual


def train_model(source,directory,options:TrainOptions,*,emit=lambda event:None,stop_requested=lambda:False):
    started=time.perf_counter(); directory=Path(directory); directory.mkdir(parents=True,exist_ok=False)
    metadata=dict(schema_version=1,id=directory.name,status='training',checkpoint_ready=False,
        created_at=datetime.now(timezone.utc).isoformat(),options=asdict(options),family=options.family,
        history=[],scope='Fixed mesh/supports/material constants; compact experimental reference models')
    save_metadata(directory,metadata)
    threads=torch.get_num_threads()
    try:
        torch.set_num_threads(min(4,threads))
        dataset=load_dataset(source); config=dataset['cases'][0]; mesh=dataset['mesh']
        train,validation,split=split_cases(dataset['cases'],dataset['case_indices'],options.validation_fraction,options.seed)
        metadata.update(source_training_id=dataset['metadata']['id'],source_sha256=dataset['source_sha256'],
            source_case_sha256=dataset['case_provenance_sha256'],config=asdict(config),
            signature={key:getattr(config,key) for key in SIGNATURE},split=split,
            train_states=len(train),validation_states=len(validation))
        save_metadata(directory,metadata)
        if stop_requested():
            metadata['status']='cancelled'; save_metadata(directory,metadata); return metadata
        if options.family in POD_FAMILIES:
            fitted=fit_pod(dataset,train,options,emit,stop_requested)
            if fitted is not None:
                payload,diagnostics,size=fitted
                metadata.update(rank=max(item['rank'] for item in diagnostics),
                    ranks=[item['rank'] for item in diagnostics],basis_diagnostics=diagnostics,operator_bytes=size,device='cpu')
                checkpoint(directory,metadata,payload)
                # Evaluate on the validation split without fitting to its states.
                from .inference import Predictor
                predictor=Predictor(directory,config,expected_sha256=metadata['model_sha256'])
                try:
                    errors={'train':[],'validation':[]}
                    for column in np.concatenate((train,validation)):
                        if stop_requested(): break
                        case=dataset['cases'][int(dataset['case_indices'][column])]
                        if predictor.config!=case: predictor.set_config(case)
                        result=predictor.raw_state(dataset['moduli'][:,column])
                        reference=dataset['displacements'][:,column]
                        split_name='train' if column in train else 'validation'
                        errors[split_name].append(float(np.linalg.norm(result.displacement[mesh.free_dofs]-reference)/np.linalg.norm(reference)))
                    if errors['train'] and errors['validation']:
                        row=dict(epoch=1,train_loss=float(np.mean(np.square(errors['train']))),
                            validation_loss=float(np.mean(np.square(errors['validation']))),
                            validation_displacement_error=float(np.mean(errors['validation'])),seconds=time.perf_counter()-started)
                        metadata['history'].append(row); emit({'event':'ml_epoch',**row})
                finally: predictor.close()
        else:
            if mesh.dof_count//2>6000:
                raise ValueError('Neural training is limited to 6000 mesh nodes in this local lab; use a smaller dataset')
            metadata['device']=device_for(options.device)
            all_columns=np.arange(len(dataset['case_indices']))
            target=full_targets(dataset,all_columns)
            scale=float(np.sqrt(np.mean(target[train]**2)))
            if not np.isfinite(scale) or scale<=0: raise ValueError('Training displacements have invalid scale')
            metadata['displacement_scale']=scale
            inputs=[]
            for column in all_columns:
                case=dataset['cases'][int(dataset['case_indices'][column])]
                inputs.append(features(torch.tensor(dataset['moduli'][:,column],dtype=torch.float32)[None],case)[0])
            inputs=torch.stack(inputs)
            targets=torch.tensor(target.reshape(len(target),config.ny+1,config.nx+1,2).transpose(0,3,1,2),dtype=torch.float32)
            moduli=torch.tensor(dataset['moduli'].T,dtype=torch.float32)
            basis=None
            if options.family=='pod_mlp':
                free_basis,_,_,diagnostic=pod_basis(dataset['displacements'][:,train],options.energy_target,options.max_modes,True)
                basis=np.zeros((mesh.dof_count,free_basis.shape[1])); basis[mesh.free_dofs]=free_basis
                metadata['rank']=basis.shape[1]; metadata['basis_diagnostics']=[diagnostic]
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(options.seed)
                network=make_network(options,config,rank=metadata.get('rank'),basis=basis).to(metadata['device'])
                optimizer=torch.optim.Adam(network.parameters(),lr=options.learning_rate)
                metadata['parameter_count']=sum(p.numel() for p in network.parameters())
                stiffness=torch.tensor(q4_stiffness(config.length/config.nx,config.height/config.ny,
                    config.poisson_ratio,config.thickness),dtype=torch.float32,device=metadata['device'])
                element_dofs=torch.tensor(mesh.element_dofs,device=metadata['device'])
                generator=torch.Generator().manual_seed(options.seed)
                best=float('inf'); stale=0
                for epoch in range(1,options.epochs+1):
                    if stop_requested(): break
                    network.train(); losses=[]
                    order=torch.tensor(train)[torch.randperm(len(train),generator=generator)]
                    for indices in order.split(options.batch_size):
                        if stop_requested(): break
                        x=inputs[indices].to(metadata['device']); reference=targets[indices].to(metadata['device'])
                        prediction=predict_network(network,x,scale)
                        relative=((prediction-reference)**2).flatten(1).sum(1)/reference.square().flatten(1).sum(1).clamp_min(1e-16)
                        loss=relative.mean()
                        if options.energy_loss_weight or options.residual_loss_weight:
                            energy,residual=physics_loss(prediction,reference,moduli[indices].to(metadata['device']),
                                stiffness,element_dofs,x[:,2:5])
                            loss=loss+options.energy_loss_weight*energy+options.residual_loss_weight*residual
                        if not torch.isfinite(loss): raise ValueError('Training loss became nonfinite; reduce the learning rate or physics weights')
                        optimizer.zero_grad(); loss.backward()
                        torch.nn.utils.clip_grad_norm_(network.parameters(),1.)
                        optimizer.step(); losses.append(float(loss.detach()))
                    if not losses: break
                    network.eval(); validation_errors=[]
                    with torch.no_grad():
                        for indices in torch.tensor(validation).split(options.batch_size):
                            x=inputs[indices].to(metadata['device']); reference=targets[indices].to(metadata['device'])
                            prediction=predict_network(network,x,scale)
                            relative=(prediction-reference).square().flatten(1).sum(1)/reference.square().flatten(1).sum(1).clamp_min(1e-16)
                            validation_errors.extend(relative.cpu().tolist())
                    value=float(np.mean(validation_errors))
                    row=dict(epoch=epoch,train_loss=float(np.mean(losses)),validation_loss=value,
                        validation_displacement_error=float(np.mean(np.sqrt(validation_errors))),seconds=time.perf_counter()-started)
                    metadata['history'].append(row)
                    if value<best:
                        best=value; stale=0; metadata['best_epoch']=epoch
                        checkpoint(directory,metadata,dict(weights={key:value.detach().cpu() for key,value in network.state_dict().items()}))
                    else: stale+=1
                    save_metadata(directory,metadata); emit({'event':'ml_epoch',**row})
                    if stale>=options.patience:
                        metadata['early_stopped']=True; break
        metadata.update(status='cancelled' if stop_requested() else 'trained',elapsed_seconds=time.perf_counter()-started)
        save_metadata(directory,metadata)
        return metadata
    except Exception as error:
        metadata.update(status='failed',error=str(error),elapsed_seconds=time.perf_counter()-started)
        save_metadata(directory,metadata)
        raise
    finally:
        torch.set_num_threads(threads)
