# Third-party licensing

The MIT License in [LICENSE](LICENSE) applies to original TOptVis code and
documentation unless a file specifies another license. It does not relicense
third-party code or remove the obligations of its license.

## MMA and GCMMA

TOptVis depends on [`mmapy` 0.3.1](https://github.com/arjendeetman/GCMMA-MMA-Python),
licensed under **GPL-3.0-or-later**. The upstream project credits Krister
Svanberg's original MMA/GCMMA implementation and Arjen Deetman's Python port.

The following bundled files carry GPL-3.0-or-later identifiers:

- [`src/ml_opt/optimizers/_subproblem.py`](src/ml_opt/optimizers/_subproblem.py):
  compiled adaptation of the upstream primal-dual Newton solver, using dense
  broadcasting and LAPACK solves.
- [`src/ml_opt/optimizers/_residual.py`](src/ml_opt/optimizers/_residual.py):
  fused evaluation of the solver's residual equations.

Their existing attribution and license identifiers are retained. The complete
[GPL text](src/ml_opt/optimizers/GPL-3.0-or-later.txt) is included in the repository.

The application integrates these components into its optimisation workflow.
Distributing the combined application requires GPL-compliant distribution,
including the corresponding source, license text, and notices. The MIT grant
for original components does not make the combined application MIT-only. See
the [GNU license compatibility FAQ](https://www.gnu.org/licenses/gpl-faq.html#AllCompatibility).

## Other dependencies

Dependencies retain their respective licenses. This notice does not grant
rights to third-party packages beyond the terms supplied by their authors.
