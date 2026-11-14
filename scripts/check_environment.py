"""Check PyTorch autograd and a PETSc linear solve in the active environment."""

import numpy as np
import torch
from petsc4py import PETSc


def main() -> None:
    values = torch.tensor([1.0, 2.0], dtype=torch.float64, requires_grad=True)
    values.square().sum().backward()
    torch.testing.assert_close(values.grad, 2.0 * values.detach())

    matrix = PETSc.Mat().createAIJ([2, 2], nnz=2, comm=PETSc.COMM_SELF)
    matrix.setValues([0, 1], [0, 1], [[2.0, -1.0], [-1.0, 2.0]])
    matrix.assemble()
    solution, rhs = matrix.createVecs()
    rhs.setValues([0, 1], [1.0, 0.0])
    rhs.assemble()
    solver = PETSc.KSP().create(comm=PETSc.COMM_SELF)
    solver.setOperators(matrix)
    solver.setType("cg")
    solver.getPC().setType("jacobi")
    solver.setTolerances(rtol=1e-12)
    solver.solve(rhs, solution)
    if solver.getConvergedReason() <= 0:
        raise RuntimeError(f"PETSc solve failed: {solver.getConvergedReason()}")
    np.testing.assert_allclose(solution.getArray(), [2.0 / 3.0, 1.0 / 3.0])

    print(f"PyTorch {torch.__version__}: autograd passed")
    print(f"NumPy {np.__version__}")
    print(f"PETSc {'.'.join(map(str, PETSc.Sys.getVersion()))}: CG solve passed")
    print(f"PETSc scalar dtype: {np.dtype(PETSc.ScalarType)}")
    print(f"PETSc index dtype: {np.dtype(PETSc.IntType)}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    solver.destroy()
    solution.destroy()
    rhs.destroy()
    matrix.destroy()


if __name__ == "__main__":
    main()
