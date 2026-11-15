"""Predictions, consistent neural derivatives, and optional checked FEM correction."""

import json
from pathlib import Path

import numpy as np
import torch
from petsc4py import PETSc

from ..config import CantileverConfig
from ..fem.elasticity import ElasticitySolver,ElasticityState
from ..fem.element import q4_stiffness
from ..fem.mesh import StructuredMesh
from ..material import elastic_modulus
from ..problem import CantileverProblem,Evaluation
from ..projection import heaviside_projection
from ..rom.operators import MAX_OPERATOR_BYTES,prepare_element_operators
from ..rom.solver import GalerkinSolver
from ..rom.training import read_metadata
from .config import POD_FAMILIES,SIGNATURE,TrainOptions
from .data import features,file_hash,fixed_features
from .networks import make_network,predict_network


def state_from_displacement(displacement,moduli,mesh,stiffness,force):
    displacement=np.asarray(displacement,dtype=np.float64)
    if displacement.shape!=(mesh.dof_count,) or not np.all(np.isfinite(displacement)):
        raise ValueError('The model predicted invalid displacement')
    element=displacement[mesh.element_dofs]
    unit_force=element @ stiffness
    energy=np.einsum('ei,ei->e',element,unit_force)
    internal=np.bincount(mesh.element_dofs.ravel(),weights=(moduli[:,None]*unit_force).ravel(),minlength=mesh.dof_count)
    residual=np.linalg.norm((internal-force)[mesh.free_dofs])/np.linalg.norm(force[mesh.free_dofs])
    return ElasticityState(displacement,float(force@displacement),energy,float(residual),0)


class Predictor:
    def __init__(self,directory,config,*,mode='surrogate',expected_sha256=None,stop_requested=lambda:False):
        if mode not in ('surrogate','corrected'): raise ValueError('Unknown evaluation mode')
        self.directory=Path(directory); self.metadata=read_metadata(directory)
        path=self.directory/'model.pt'; digest=file_hash(path)
        expected=expected_sha256 or self.metadata.get('model_sha256')
        if not expected or expected!=digest:
            raise ValueError('The saved model changed or is incomplete; refresh the model library')
        payload=torch.load(path,map_location='cpu',weights_only=True)
        if (payload.get('schema_version')!=1 or payload.get('id')!=self.metadata['id']
            or payload.get('options')!=self.metadata['options'] or payload.get('config')!=self.metadata['config']
            or payload.get('source_sha256')!=self.metadata['source_sha256']):
            raise ValueError('Model checkpoint provenance does not match its metadata')
        self.options=TrainOptions(**payload['options']); self.family=self.options.family
        self.mode=mode; self.stopped=stop_requested
        self.solve_count=self.predictions=self.fom_solves=self.correction_iterations=self.lu_fallbacks=self.basis_updates=0
        self.corrector=self.full=None; self.solvers=[]; self.last_state=None; self.last_used_fom=False
        self.closed=False; self.rom_info=None
        self.bases=[]; self.operators=[]; self.centers=None
        self.scale=payload.get('displacement_scale',1.)
        if self.family in POD_FAMILIES:
            self.bases=[value.numpy().copy() for value in payload['bases']]
            self.operators=[value.numpy() for value in payload['operators']]
            self.centers=payload['centers'].numpy()
            self.network=None
            self.model_info={'rank':max(b.shape[1] for b in self.bases),'model_id':self.metadata['id'],
                'model_sha256':digest,'family':self.family}
        else:
            self.network=make_network(self.options,CantileverConfig(**payload['config']),rank=payload.get('rank'))
            self.network.load_state_dict(payload['weights'],strict=True)
            self.network.eval()
        try: self.set_config(config)
        except BaseException:
            self.close(); raise

    def set_config(self,config):
        trained=CantileverConfig(**self.metadata['config'])
        for key in SIGNATURE:
            if getattr(config,key)!=getattr(trained,key):
                raise ValueError(f'The model {key} must match its training dataset ({getattr(trained,key)})')
        if config.robust:
            raise ValueError('Model Lab experiments currently use single-field elasticity')
        for solver in self.solvers: solver.close()
        if self.corrector is not None: self.corrector.close(); self.corrector=None
        if self.full is not None: self.full.close(); self.full=None
        self.config=config; self.mesh=StructuredMesh(config.nx,config.ny,config.length,config.height)
        self.stiffness=q4_stiffness(config.length/config.nx,config.height/config.ny,config.poisson_ratio,config.thickness)
        node=None if config.load_node_x is None else (config.load_node_x,config.load_node_y)
        self.force=self.mesh.cantilever_force(config.load,config.load_angle,node)
        self.fixed=fixed_features(config)
        self.solvers=[GalerkinSolver(config,basis,reduced_element_stiffness=operators)
                      for basis,operators in zip(self.bases,self.operators)]

    def tensor(self,moduli):
        inputs=features(moduli.float(),self.config,self.fixed)
        field=predict_network(self.network,inputs,self.scale*self.config.load)
        return field.permute(0,2,3,1).reshape(len(field),-1).double()

    def raw_state(self,moduli):
        moduli=np.asarray(moduli,dtype=np.float64)
        if (moduli.shape!=(self.mesh.element_count,) or not np.all(np.isfinite(moduli)) or np.any(moduli<=0)):
            raise ValueError('Expected one positive finite element modulus')
        self.predictions+=1
        if self.solvers:
            field=np.log(moduli/self.config.young_modulus)
            cluster=int(np.mean((self.centers-field)**2,axis=1).argmin())
            return self.solvers[cluster].solve(moduli)
        with torch.no_grad():
            displacement=self.tensor(torch.from_numpy(moduli)[None])[0].numpy()
        return state_from_displacement(displacement,moduli,self.mesh,self.stiffness,self.force)

    def elasticity(self,linear_solver=None):
        config=self.config
        return ElasticitySolver(self.mesh,config.poisson_ratio,config.thickness,config.load,config.linear_tolerance,
            load_angle=config.load_angle,load_node=None if config.load_node_x is None else (config.load_node_x,config.load_node_y),
            linear_solver=linear_solver or config.linear_solver)

    def solve(self,moduli):
        if self.closed: raise RuntimeError('The model solver is closed')
        raw=self.raw_state(moduli)
        self.last_used_fom=False
        if self.mode=='corrected':
            if self.corrector is None:
                self.corrector=self.elasticity('cg-gamg')
                self.corrector._solver.setInitialGuessNonzero(True)
            if not self.lu_fallbacks:
                self.corrector._solution.array[:]=raw.displacement[self.mesh.free_dofs]
            try: state=self.corrector.solve(moduli)
            except (RuntimeError,PETSc.Error):
                # A difficult contrast/mesh can defeat the iterative configuration.
                # Record the fallback and retain the same checked equilibrium target.
                self.correction_iterations+=self.corrector._solver.getIterationNumber()
                self.corrector.close()
                self.corrector=self.elasticity('lu')
                self.lu_fallbacks+=1
                state=self.corrector.solve(moduli)
            self.correction_iterations+=state.iterations; self.fom_solves+=1; self.last_used_fom=True
        elif self.family=='adaptive_pod' and raw.relative_residual>self.options.residual_threshold:
            if self.full is None: self.full=self.elasticity()
            state=self.full.solve(moduli); self.fom_solves+=1; self.last_used_fom=True
        else: state=raw
        self.last_state=state; self.solve_count+=1
        return state

    def accept(self):
        """Only accepted outer states can enrich; GCMMA trials never mutate V."""
        if self.family!='adaptive_pod' or not self.last_used_fom or self.stopped(): return
        basis=self.bases[0]; rank=basis.shape[1]+1
        size=self.mesh.element_count*rank*(rank+1)//2*8
        if rank>self.options.adaptive_max_modes or size>MAX_OPERATOR_BYTES: return
        column=self.last_state.displacement[self.mesh.free_dofs]
        residual=column-basis@(basis.T@column)
        residual-=basis@(basis.T@residual)
        norm=np.linalg.norm(residual)
        if norm<=1e-10*np.linalg.norm(column): return
        expanded=np.column_stack((basis,residual/norm))
        operators=prepare_element_operators(self.config,expanded,stop_requested=self.stopped)
        if operators is None: return
        replacement=GalerkinSolver(self.config,expanded,reduced_element_stiffness=operators)
        self.solvers[0].close(); self.solvers=[replacement]
        self.bases=[expanded]; self.operators=[operators]
        self.model_info['rank']=rank; self.basis_updates+=1

    def stats(self):
        return dict(predictions=self.predictions,fom_solves=self.fom_solves,
            correction_iterations=self.correction_iterations,lu_fallbacks=self.lu_fallbacks,
            basis_updates=self.basis_updates,basis_ranks=[b.shape[1] for b in self.bases])

    def close(self):
        if self.closed: return
        self.closed=True
        for solver in self.solvers: solver.close()
        if self.corrector is not None: self.corrector.close()
        if self.full is not None: self.full.close()


class ModelProblem(CantileverProblem):
    def __init__(self,config,predictor):
        super().__init__(config,state_solver=predictor)

    def evaluate(self,design,sensitivities=True):
        # Galerkin and converged FEM solves use the physical self-adjoint chain.
        if self.solver.network is None or self.solver.mode=='corrected':
            return super().evaluate(design,sensitivities)
        array=np.asarray(design,dtype=np.float64)
        if array.shape!=(self.mesh.element_count,) or not np.all(np.isfinite(array)) or np.any(array<0) or np.any(array>1):
            raise ValueError('Expected one design density in [0,1] per element')
        x=torch.tensor(array,dtype=torch.float64,requires_grad=sensitivities)
        with torch.set_grad_enabled(sensitivities):
            density=heaviside_projection(self.density_filter.apply(x),self.projection_beta,self.config.projection_eta)
            moduli=elastic_modulus(density,self.config)
            displacement=self.solver.tensor(moduli[None])[0]
            objective=torch.as_tensor(self.solver.force,dtype=torch.float64)@displacement
            volume=density.mean(); volume_error=volume/self.config.volume_fraction-1
            constraint=(-volume_error if self.config.volume_constraint=='gt' else
                        volume_error.abs() if self.config.volume_constraint=='eq' else volume_error)
            if sensitivities:
                gradient=torch.autograd.grad(objective,x,retain_graph=True)[0].numpy().copy()
                derivative=volume_error if self.config.volume_constraint=='eq' else constraint
                volume_gradient=torch.autograd.grad(derivative,x)[0].numpy().copy()
                constraint_gradient=(volume_gradient*np.sign(float(volume_error.detach()))
                                     if self.config.volume_constraint=='eq' else volume_gradient)
            else: gradient=volume_gradient=constraint_gradient=None
        state=state_from_displacement(displacement.detach().numpy(),moduli.detach().numpy(),
            self.mesh,self.solver.stiffness,self.solver.force)
        self.solver.last_state=state; self.solver.solve_count+=1; self.solver.predictions+=1
        if not np.isfinite(state.compliance) or (gradient is not None and not np.all(np.isfinite(gradient))):
            raise ValueError('The neural prediction or its design gradient is nonfinite')
        return Evaluation(state.compliance,float(volume.detach()),float(constraint.detach()),
            density.detach().numpy().copy(),state.displacement,state.relative_residual,0,
            gradient,constraint_gradient,projection_beta=self.projection_beta,
            volume_error=float(volume_error.detach()),
            volume_gradient=volume_gradient if self.config.volume_constraint=='eq' else None)
