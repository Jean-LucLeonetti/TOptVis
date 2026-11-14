"""Case-separated snapshots and differentiable material/load field features."""

from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from ..config import CantileverConfig
from ..fem.mesh import StructuredMesh
from ..material import elastic_modulus
from ..rom.training import read_metadata
from .config import SIGNATURE


def file_hash(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):
            digest.update(block)
    return digest.hexdigest()


def case_key(config):
    angle=np.deg2rad(config.load_angle)
    return tuple(getattr(config,key) for key in SIGNATURE)+(config.load_node_x or config.nx,
        config.load_node_y if config.load_node_y is not None else config.ny//2,
        round(float(np.cos(angle)),10),round(float(np.sin(angle)),10),
        config.volume_fraction,config.material_model,config.penalisation,
        config.filter_radius,config.projection_beta,config.projection_eta)


def split_cases(cases, case_indices, fraction, seed):
    groups={}
    for index in np.unique(case_indices):
        groups.setdefault(case_key(cases[int(index)]),[]).append(int(index))
    if len(groups)<2:
        raise ValueError("Training needs at least two independent cases so a complete case can be held out")
    keys=list(groups)
    permutation=np.random.default_rng(seed).permutation(len(keys))
    count=max(1,min(len(keys)-1,int(np.ceil(fraction*len(keys)))))
    held_out={index for position in permutation[:count] for index in groups[keys[position]]}
    validation=np.array([int(i) in held_out for i in case_indices])
    return np.flatnonzero(~validation),np.flatnonzero(validation),{
        'train_case_indices':sorted(set(case_indices[~validation].tolist())),
        'validation_case_indices':sorted(held_out),'independent_case_count':len(groups),
        'method':'Whole physical cases; duplicate parameter cases remain in the same split',
        'scope':'Validation cases select checkpoints; independent unseen-case FOM validation remains separate'}


def load_dataset(directory):
    directory=Path(directory)
    metadata=read_metadata(directory)
    if metadata.get('status') not in ('snapshots_ready','basis_ready','cancelled'):
        raise ValueError("Choose a saved dataset with completed snapshots")
    cases=[CantileverConfig(**item) for item in metadata['cases']]
    if not cases or any(c.robust or c.state_solver!='fom' for c in cases):
        raise ValueError("Model Lab requires single-field full-order snapshot cases")
    if any(any(getattr(c,key)!=getattr(cases[0],key) for key in SIGNATURE) for c in cases):
        raise ValueError("All dataset cases must have the same mesh and physical material constants")
    mesh=StructuredMesh(cases[0].nx,cases[0].ny,cases[0].length,cases[0].height)
    path=directory/'snapshots.npz'
    with np.load(path,allow_pickle=False) as archive:
        u=archive['displacements']; densities=archive['densities']; indices=archive['case_indices']
        free=archive['free_dofs']; residuals=archive['residuals']
        provenance=json.loads(str(archive['metadata_json']))
    if provenance.get('cases')!=metadata['cases']:
        raise ValueError("Snapshot case provenance does not match the dataset metadata")
    count=len(indices)
    if (count<2 or u.shape!=(len(mesh.free_dofs),count) or densities.shape!=(mesh.element_count,count)
        or indices.ndim!=1 or not np.issubdtype(indices.dtype,np.integer)
        or np.any(indices<0) or np.any(indices>=len(cases)) or not np.array_equal(free,mesh.free_dofs)
        or not np.all(np.isfinite(u)) or not np.all(np.isfinite(densities))
        or np.any(densities<0) or np.any(densities>1) or residuals.shape!=(count,)
        or not np.all(np.isfinite(residuals)) or np.any(residuals>1e-7)):
        raise ValueError("Invalid or non-equilibrium full-order snapshot arrays")
    moduli=np.empty_like(densities)
    for index in np.unique(indices):
        selected=indices==index
        moduli[:,selected]=elastic_modulus(torch.from_numpy(densities[:,selected]),cases[int(index)]).numpy()
    return dict(metadata=metadata,cases=cases,mesh=mesh,displacements=u,densities=densities,
        moduli=moduli,case_indices=indices,source_sha256=file_hash(path),
        case_provenance_sha256=hashlib.sha256(json.dumps(metadata['cases'],sort_keys=True).encode()).hexdigest())


def nodal_average(elements, config):
    """B x elements -> B x nodal height x width, including boundary nodes."""
    fields=elements.reshape(-1,config.ny,config.nx)
    total=sum(F.pad(fields,padding) for padding in ((0,1,0,1),(1,0,0,1),(0,1,1,0),(1,0,1,0)))
    counts=sum(F.pad(torch.ones_like(fields),padding) for padding in ((0,1,0,1),(1,0,0,1),(0,1,1,0),(1,0,1,0)))
    return total/counts


def fixed_features(config, *, device='cpu',dtype=torch.float32):
    mesh=StructuredMesh(config.nx,config.ny,config.length,config.height)
    node=None if config.load_node_x is None else (config.load_node_x,config.load_node_y)
    force=mesh.cantilever_force(config.load,config.load_angle,node).reshape(config.ny+1,config.nx+1,2)/config.load
    y,x=torch.meshgrid(torch.linspace(0,1,config.ny+1,device=device,dtype=dtype),
        torch.linspace(0,1,config.nx+1,device=device,dtype=dtype),indexing='ij')
    clamp=torch.zeros_like(x); clamp[:,0]=1
    return torch.stack((torch.as_tensor(force[:,:,0],device=device,dtype=dtype),
        torch.as_tensor(force[:,:,1],device=device,dtype=dtype),clamp,x,y))


def features(moduli, config, fixed=None):
    fixed=fixed_features(config,device=moduli.device,dtype=moduli.dtype) if fixed is None else fixed
    linear=nodal_average(moduli/config.young_modulus,config)
    logarithmic=nodal_average(torch.log(moduli/config.minimum_modulus)/np.log(config.young_modulus/config.minimum_modulus),config)
    return torch.cat((linear[:,None],logarithmic[:,None],fixed[None].expand(len(linear),-1,-1,-1)),dim=1)


def full_targets(dataset, columns):
    mesh=dataset['mesh']; cases=dataset['cases']; indices=dataset['case_indices']
    result=np.zeros((len(columns),mesh.dof_count))
    result[:,mesh.free_dofs]=dataset['displacements'][:,columns].T
    result/=np.array([cases[int(indices[i])].load for i in columns])[:,None]
    return result
