"""Choose the state solver consistently in the main and robust field workers."""

from .elasticity import ElasticitySolver
from .mesh import StructuredMesh


def create_state_solver(config):
    if config.state_solver == "rom":
        from ..rom.solver import GalerkinSolver, load_basis
        basis, information, operators = load_basis(config.rom_basis_path, config,
            expected_sha256=config.rom_basis_sha256, include_operators=True)
        solver = GalerkinSolver(config, basis, reduced_element_stiffness=operators)
        solver.rom_info = information
        return solver
    return ElasticitySolver(StructuredMesh(config.nx, config.ny, config.length, config.height,
        config.boundary_conditions, config.load_cases[0].id),
        config.poisson_ratio, config.thickness, config.load, config.linear_tolerance,
        linear_solver=config.linear_solver,
        load_cases=config.load_cases, load_aggregation=config.load_aggregation,
        load_angle=config.load_angle, load_node=None if config.load_node_x is None else
        (config.load_node_x, config.load_node_y))
