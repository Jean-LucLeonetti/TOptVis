# TOptVis guide

A Python project for topology optimisation with PyTorch and PETSc through
`petsc4py`.

## Performance

Exact state caching and persistent three-field workers are described in
[`benchmarks/SOLVER_RESULTS.md`](../benchmarks/SOLVER_RESULTS.md), including
comparisons of LU, Cholesky, CG/ICC, and CG/GAMG. In **Solver → Full-order
method**, choose the PETSc solver; LU remains the default. Cholesky was fastest
for the larger measured optimisation case. The iterative choices were accurate
but slower on these meshes. The CLI accepts `--linear-solver cholesky` (or
`lu`, `cg-icc`, `cg-gamg`). These methods apply to full-order elasticity.

An exact match of the latest element moduli reuses displacement and element
energies while rebuilding the PyTorch graph. This avoids the repeated accepted
trial solve in GCMMA and keeps material/projection gradients current. State
solve counters count actual numerical solves. Parallel robust app runs reuse
three compatible workers between Run clicks, with fresh per-run caches and
counters. Changed state-solver inputs recreate the group; workers retire after
60 idle seconds, when a nonparallel/training job starts, or at shutdown.
Numerical factors and iterative preconditioners are rebuilt for changed moduli.

Measured timings, bottleneck locations, methodology, and raw data are in
[`benchmarks/IMPROVEMENTS.md`](../benchmarks/IMPROVEMENTS.md), with the original
baseline retained in [`benchmarks/RESULTS.md`](../benchmarks/RESULTS.md). Source comments `PERF B1`–`B4`
mark the measured hotspots. The app now prewarms and reuses one solver worker;
each problem retains its PETSc matrix graph, vectors, KSP, and symbolic LU
factorisation. Numerical LU factors are updated for every density evaluation.
MMA/GCMMA diagonal scaling uses NumPy broadcasting through an isolated adapter
to the pinned mmapy 0.3.1 routines. Reproduce the serial Python/PETSc/HTTP benchmarks
with:

```powershell
.\.venv\Scripts\python.exe benchmarks\benchmark.py
```

For the browser's actual canvas drawing functions, run
`python benchmarks/render_benchmark.py` and open its printed loopback URL.

## Visual interface

The interface uses a compact toolbar, readable section navigation, and a
consistent settings panel beside the design and results. Solver choices live
under **Solver**; parameter values stay in their controls. Repeated technical
badges, parameter summaries, instructional captions, and technology footers
are omitted. Run status, errors, result metrics, and relevant legends remain
visible. See the product and interface guidance in [`AGENTS.md`](../AGENTS.md)
when extending the UI.

Constrained MMA/GCMMA subproblems use a cached Numba-compiled version of the
existing primal-dual Newton loop. The equations, constraints and tolerances are
unchanged. A single-constraint run retains the existing NumPy kernel.
The first solver preparation after installation, interpreter/CPU changes, or a
kernel source update can take several minutes to compile; later processes reuse
the cache. The interface shows **Preparing solver…** and enables Run when ready.
The existing Python kernel remains available if the acceleration runtime is
missing; update the environment with `python -m pip install -e ".[dev]"` to
install the declared Numba dependency.

The compiled kernel is derived from GPL-licensed mmapy 0.3.1; attribution and
the complete licence are included in `src/ml_opt/optimizers/`.
Measured solid/void iteration speedups and remaining bottlenecks are documented
in [the size-constraint performance report](../benchmarks/SIZE_CONSTRAINTS.md).

Double-click **Start TOptVis.bat** on Windows or **Start TOptVis.command** on
macOS in the project folder. For first-time Mac setup, see
[Local macOS environment](#local-macos-environment). Both launchers use the existing
`.venv`, starts the app, and opens the interface in your default browser once
the server is ready. Keep the launcher window open while using the app.
The launcher reuses a running server only when it matches this checkout's
current source. If an older server is still holding port 8000, it starts the
updated app on an available nearby port and opens that URL. Existing runs
stay undisturbed; the launcher prints the address it uses.

Start the local interface from the project folder:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[ui]" --no-build-isolation
.\.venv\Scripts\python.exe -m ml_opt.web
```

Open [the cantilever workspace](http://127.0.0.1:8000/) in your browser.
The Windows setup script includes the interface dependencies. You can also
launch it with `toptvis-ui` after activating the environment; use `--port 8001`
if port 8000 is already occupied.

The top tabs separate **Optimisation** from the **Machine learning** workspace.
Inside Machine learning, **Regression** is the first and default method, followed
by **Reduced models**, which contains snapshot generation, POD basis building,
and held-out validation. Switching tabs preserves settings, results, and any
active job. Open these workspaces directly with `/#cantilever-2d` and
`/#ml/regression` (or `/#ml`) and `/#ml/rom`. Existing `/#rom-training` links
also open Reduced models.

Within Optimisation, the far-left tabs separate **Geometry**, **Boundaries**, **Material**,
**Filters**, **Objective**, **Constraints**, **Optimiser**, **Solver**, and **Convergence**.
**Objective** describes compliance minimisation, including the worst of three
material fields when robustness is enabled. **Constraints** contains the volume
relation and target, maximum solid size, and maximum void size. **Optimiser**
contains MMA/GCMMA, the outer iteration limit, and GCMMA's inner retry limit.
**Solver** contains the full/reduced model selection, linear solver, saved
reduced model, and sequential/parallel state execution for three-field robustness.
The configuration panel shows the selected
tab; the central workspace displays the mesh/topology or convergence graphs. Tab changes retain
all values. Use arrow keys, Home, and End to navigate the tab rail.
On desktop, the workspace uses the full window width and height. The active
configuration form fills its column, and the mesh and topology viewers share
the available height. Columns scroll only when their content exceeds the
available space; training results retain independent scrolling. Starting a run
scrolls its result into view within that column when necessary.
Run messages use the existing result footer, and metric captions retain their
space when empty, so starting and finishing a run do not resize the viewers.
On desktop, error alerts overlay the workspace and can be dismissed without
changing its dimensions.

1. In **Geometry**, set the domain length, height, and number of elements. The geometry and mesh
   preview updates immediately.
2. Choose **Point**, **Line**, or **Surface** above the mesh. Select nodes, then
   assign a fixed support or force in **Boundaries**. The initial left-edge
   support and point force are listed there and can be edited or removed.
   Angles are counterclockwise from the positive x direction, so -90 degrees
   points downward.
3. In **Material**, choose SIMP or RAMP and its penalty value.
4. In **Filters**, set the Helmholtz length and the Heaviside projection's
   sharpness β and threshold η. Larger β sharpens the solid/void transition;
   fixed β = 0 disables projection. Use **β schedule** for optional capped
   increases during the run; length and threshold remain fixed.
5. In **Objective**, review the compliance objective for the selected material fields.
6. In **Constraints**, choose the material volume relation and target, and optional
   maximum solid/void diameters.
7. In **Optimiser**, choose MMA or GCMMA and the iteration limits. In **Solver**,
   choose the state equation solver. Click
   **Run optimisation** to see the topology, compliance, volume, and iteration
   values update as the actual solver runs. **Stop** retains the last evaluated
   design.

### Mesh selection and boundary conditions

Use **Point** to select an individual mesh node. For **Line**, drag along a mesh
row or column, or click its two endpoints; diagonal gestures snap to the nearest
horizontal or vertical direction. For **Surface**, drag a rectangle or click
opposite corners to select its nodes. Selections include their endpoints and
region boundaries. The column/row controls provide an exact keyboard alternative.
On the focused canvas, arrow keys move the cursor, Enter selects an endpoint,
and Escape clears the selection.

In **Boundaries**, choose **Fixed support**, **Restrain x**, **Restrain y**,
**Prescribed displacement**, or **Force**, then click **Assign**. Fixed supports
restrain both translations; directional restraints fix only the selected
component. Prescribed motion can target either or both components with signed
values in model units. Overlapping restraints must prescribe consistent values.
The restrained translations must remove all three planar rigid-body modes;
a pin plus a suitably located roller is supported.
Line forces are uniformly distributed over the selected segment and surface
forces over the selected area. **Total force** is the resultant for that
assignment, shared using consistent nodal weights, rather than repeated at
every node. Overlapping forces add. Loads applied to fixed nodes remain in the
force vector and contribute to support reactions; at least one nonzero free
load component or nonzero prescribed motion must remain in every load case.

Click an assigned condition to edit its selection, type, or force, then click
**Update**. **Remove** deletes one condition; **Reset cantilever** restores the
default supports and force on the current mesh. **Clear selection** clears only
the current selection.
Selections are stored as fractions of the domain and snap to the current mesh
after refinement. A selection that collapses on a coarser mesh is rejected.
Selections and condition edits lock while optimisation is running.

Custom conditions are sent to the same numerical solve, included in saved
configuration, and propagated to all robust field workers. A saved reduced
basis can only be used with the support DOFs it was trained on; incompatible
supports are rejected before optimisation. New loads remain subject to the
basis coverage and residual checks.

### Independent load cases and prescribed motion

Use **Add case** in **Boundaries** to duplicate the active case's forces into a
new independent loading scenario. Rename the case, change its weight, then
edit or replace its forces. Supports and prescribed displacements are shared
by all cases. The active case controls the force overlay and displayed case
objective; it does not select which cases are optimised. Removing a case also
removes its forces. Up to 16 cases and 128 total assignments are supported.

Choose **Weighted mean** or **Worst case**. For case objectives \(C_k\) and
positive weights \(w_k\), these are respectively

\[
C = \frac{\sum_k w_k C_k}{\sum_k w_k},
\qquad C = \max_k C_k.
\]

Worst-case aggregation ignores the weights and uses an active-case subgradient
at a tie. Cases are solved independently; their force vectors are never
combined into one loading scenario. The PETSc matrix is assembled once per
material field and reused across that field's right-hand sides. PyTorch carries
the consistently aggregated sensitivities through material interpolation,
filtering, and projection. Robust optimisation aggregates the cases separately
within each of its three material fields.

For prescribed translations \(g\), the free equilibrium equation is

\[
K_{ff}u_f^{(k)} = f_f^{(k)} - K_{fc}g.
\]

With zero prescribed motion, the objective remains force compliance. With
nonzero prescribed motion, each case uses total elastic work
\(C_k = u_k^{\mathsf T}K u_k\), including support reactions. Its sensitivity is

\[
\frac{\partial C_k}{\partial E_e}
= u_{k,e}^{\mathsf T}K_e^0u_{k,e}
- 2\lambda_{k,e}^{\mathsf T}K_e^0u_{k,e},
\qquad K_{ff}\lambda_{k,f}=f_{k,f},\quad\lambda_{k,c}=0.
\]

Under imposed motion, minimising elastic work can favour a softer structure;
it must not be interpreted as force-only stiffness maximisation. The interface
labels this objective **Elastic work**. For force-driven cases the expression
reduces to the existing compliance sensitivity. Saved reduced models currently
support a single case with their original fully fixed supports; use the full
model for directional restraints, prescribed motion, or independent cases.

### Save, load, and method references

**Save** downloads a versioned `TOptVis-project.json` containing geometry,
boundary assignments, load cases and weights, optimisation settings, and the
latest density fields and convergence history when available. **Load** validates
the file and server-side model configuration before replacing the workspace.
Invalid files leave the current model unchanged. Imported results are labelled
as saved snapshots; starting a new run recomputes the model rather than resuming
an optimiser's internal state. Model inputs and loading lock during a run;
saving a consistent current snapshot remains available.

Project imports are limited to 64 MiB, ten million mesh cells, and 1001 history
samples. A selected reduced-model ID is saved as a reference; its training
dataset and basis are not embedded. If that dataset is unavailable, select an
available model or switch to the full model before recomputing.

**References** opens an optional methods dialog from the topology or reduced-model
workspace. It includes short implementation notes and original-paper DOI links
for SIMP, RAMP, Helmholtz filtering, projection and robustness, maximum feature
sizes, MMA, GCMMA, and POD. The numerical backend remains PETSc and PyTorch.

Run the solver-independent editor checks with
`node --test tests/test_mesh_selection.mjs tests/test_boundary_editor.mjs tests/test_project_io.mjs`.
Python boundary and integration checks are in `tests/test_boundaries.py` and
`tests/test_boundary_interface.py`; directional-support, imposed-motion, and
multiple-case gradient checks are in `tests/test_extended_supports.py`.

### Convergence visualisation

Select **Convergence** on the far-left rail to see five live graphs:
compliance, physical material volume, maximum design change, relative
objective change, and individual iteration times versus outer iteration.
The graphs use the actual evaluated
history rows received by the interface, including the projected physical volume.
Dashed reference lines show the selected volume target and the design/objective
stopping tolerances reported by the server when the run starts.

Each graph has its own **Log scale** toggle for the vertical axis. Scales can be
mixed and remain selected as new iterations arrive or runs change. The iteration
axis stays linear. Zero and negative values remain in the history and readouts,
but are omitted from log plots with gaps in the curve; inspecting one marks it
as not shown on the log scale. Logarithmic time bars start at the lower positive
axis limit rather than zero.

Compliance and volume include iteration 0; change curves and time bars start at iteration 1
because there is no previous design for the initial evaluation. Hover over a
graph to inspect the nearest iteration across all plots, or focus a graph
and use Left/Right, Home, and End. The sidebar always shows the latest iteration.

History continues collecting while the mesh is visible. Switching tabs preserves
it, and **Back to design** returns to the model view. A completed or stopped run
keeps its history; a new run or valid model edit clears the old history. A flat
curve by itself is not a convergence certificate; the displayed run status reports
the solver's actual stopping reason.

The plots are rendered locally with canvas and no additional dependencies.
Only scalar history is retained; density images are not saved for every iteration.

**Iteration time** is shown as one bar per completed outer iteration, using the
difference between its cumulative timestamp and the previous iteration's.
Each bar includes all GCMMA trials and other work between those samples.
Setup and the initial evaluation at iteration 0 are excluded from the bars.
Hover or use the arrow keys to read an individual iteration's seconds.

The sidebar's **Elapsed solver time** uses a monotonic wall clock started before problem
creation. Each evaluated history row records cumulative `elapsed_seconds`,
including setup, the initial solve, previous callbacks, and all optimisation
work through that iteration. Process/import warm-up, browser/network delays,
and cleanup after the last sample are excluded. Timing resets for each run,
and stopping retains the last evaluated sample. The same values are saved in
`history.csv` and the final sample is included in `summary.json`. An older
running server without timing data is labelled as unavailable until restarted.

Model inputs are locked during a run; tabs remain available for inspection.
Editing the model afterward clears
the previous result so it cannot be mistaken for the updated load case.
The interface uses the same solver as the command-line example. The
**Filter length** control in the Filters tab sets the density filter length
in model units (initially 0.1). Drag the slider or enter a value: the purple
circle shows the Helmholtz length ℓ as its radius
at the actual geometry scale. The same reference is drawn over the
live topology so it can be compared with the members. Geometry and mesh
changes update that reference
immediately. Set ℓ to 0 to disable filtering. Controls lock during a run;
changing the size afterward clears the previous result.
Mesh and load edits update the preview before the run starts. A local worker
process starts in the background with the server and streams each evaluated
outer iteration to the browser. It stays warm between runs; geometry, loads,
and optimiser state are recreated for each run. Closing the live connection
cancels that run; the server allows one active optimisation at a time.
`/api/health` reports `worker_ready` separately from HTTP server readiness and
indicates projection support. If an older server is still running after an
update, the page asks for a restart before enabling Run. Close the original
launcher window, reopen **Start TOptVis.bat**, and refresh the page.

## Non-design regions

In **Optimisation > Boundaries**, choose **Assignment > Non-design: solid** or
**Non-design: void**, then select a surface on the mesh and click **Assign**.
Drag opposite corners, use two clicks, or enter start/end mesh coordinates.
The selection covers elements between the selected mesh nodes. Assigned regions
can be selected again for editing or removed in **Non-design regions**.

Solid regions remain at density 1 and void regions at density 0 in every iterate,
including all robust fields after filtering and projection. Only the remaining
elements are optimisation variables. The volume constraint measures the whole
domain, including fixed solid material; overlaps of opposite kinds, an entirely
fixed domain and impossible volume targets are rejected. Selections are stored
as domain fractions and snapped to the current mesh; refine the mesh if a region
would shrink below one element. Saved projects and imported regression setups
retain these regions. The PETSc/PyTorch state solves and sensitivities are reused.

Checks: `python -m pytest -q tests/test_non_design_regions.py tests/test_non_design_interface.py`
and `node --test tests/*.mjs`.

## Regression experiments

Click **Regression** in Machine learning to open its dropdown, then choose
**Compliance** or **Topology**. Both views share saved datasets and generation
settings, but save their fitted models separately. Direct links are
`/#ml/regression/compliance` and `/#ml/regression/topology`; existing
`/#ml/regression` links open Compliance.

Open **Machine learning > Regression** to learn a scalar input-output relationship
from actual optimisation runs. The input is permitted volume fraction; the target
is final compliance achieved by the optimiser, not a certified global optimum.

Choose **Load setup** and select a JSON project saved with **Optimisation > Save**.
Set the volume range and number of cases; all other settings come from that setup.
The regression page validates and previews the setup directly. Generation preserves
its geometry, supports, forces and load cases, material law, filters, projection
schedule, constraints, stopping rules and solver settings; only the volume target
varies. The iteration limit comes from the saved setup. Saved result snapshots are
ignored and each case starts a fresh optimisation. A reduced-model setup requires
its referenced basis in this app's training library; it is validated and fingerprinted
before generation. Invalid or unavailable models are rejected without starting a job.
For prescribed motion, the target is elastic work rather than force compliance;
minimising elastic work can favour a softer structure.

**Generate dataset** streams live topology and case progress through the same worker
as Optimisation and Reduced models. Each completed case is saved atomically with
its density, actual physical volume, compliance, stopping reason, feasibility,
iteration count and elapsed time. **Stop** keeps evaluated cases, including the
cancelled case for inspection. Failed cases keep their error message. A dataset
can therefore finish with cases unsuitable for training; inspect **Reference cases**.

Set **Maximum polynomial order**, choose **Fitting method**, and click **Train
models** to fit polynomials from order 1 through that maximum. There is no fixed
order cap. The defaults are ordinary least squares and order 2,
retaining the linear and quadratic comparison:

\[
\widehat C(v)=a+bv,\qquad \widehat C(v)=a+bv+cv^2.
\]

All orders use the same deterministic whole-case split, controlled by **Test share** and
**Split seed**. Range endpoints stay in training; random held-out interior cases
measure interpolation. At least six eligible cases are required, and at least
three remain in training. Ordinary least squares (including ridge with α = 0)
requires at least p + 1 training cases for order p and a numerically independent
polynomial basis. A positive ridge penalty permits more coefficients than
training cases. Invalid or numerically unidentifiable fits produce an error
without replacing saved models.
Inputs are scaled to [-1, 1] using the training volume range for stable fitting
and prediction. Displayed coefficients refer to the original volume variable when
representable. If expansion into original-volume coefficients overflows, the
interface labels and displays the scaled-volume coefficients instead; predictions
continue using the scaled basis. Downloads retain scaled coefficients and their
input transformation, with `coefficients: null` when the original expansion
cannot be represented. Nonfinite extrapolations are shown as unavailable.
Higher orders can overfit: compare their training and test errors rather than
assuming a higher order predicts better. Test targets never enter fitting or the training-mean
baseline. By default, only converged cases satisfying every active constraint
(volume, robust epigraph, and solid/void size where enabled) are eligible.
**Include feasible unfinished runs** explicitly includes iteration-limited cases;
cancelled, failed and infeasible cases remain excluded. The saved report records
the included unfinished cases and exact train/test/excluded case indices.

The plot distinguishes training, test and excluded cases. Select a point, use its
keyboard controls (Enter, Space, Left/Right, Home/End), or choose a case to inspect
the reference topology. The model selector switches between fitted curves and
coefficient readouts. The volume slider extends beyond the training interval;
dashed curves and shaded regions indicate extrapolation, with a separate warning
for nonpositive predictions. These models predict a scalar compliance and do not
generate or certify a topology.

Model comparison reports mean absolute percentage error, defined as

\[
\operatorname{MAPE}=\frac{100}{n}\sum_{i=1}^{n}
\frac{|\widehat C_i-C_i|}{C_i}.
\]

The portable model report additionally includes MAE and RMSE. Generation time is
reported separately from the selected model's least-squares training time and
one vectorised prediction of 201 values. Timings exclude browser/network delay;
tiny fitting/prediction measurements vary with timer resolution and system load.
Data-generation cost must be included when assessing any overall speed benefit.

Experiments persist in `outputs/regression/<id>/dataset.json` and `models.json`.
Refresh or restart, then select a saved experiment to restore its settings, cases,
split and fitted models. Downloads contain the complete dataset (including densities
and fixed solver configuration), the imported setup where used, and all fitted
models. Older saved linear/quadratic models remain supported. A SHA-256 dataset fingerprint
prevents silently loading a model against changed data. Atomic writes preserve the
last usable file if interrupted; a cancelled fit retains the previous model.

API: `GET /api/regression/datasets`, `POST /api/regression/generate`,
`GET /api/regression/<id>`, `POST /api/regression/<id>/fit`, and
`GET /api/regression/<id>/artifacts/{dataset,models}`.
The fit request accepts `degrees` (distinct positive integers), defaulting to
`[1, 2]`; there is no maximum list length or fixed order limit. The interface
requests every order up to the chosen maximum for polynomial fits. `method` is
`"ols"` (default), `"ridge"` or `"neural"`; `ridge_alpha` is finite and nonnegative,
defaulting to 0.01. Neural fitting has separate controls and artifacts below.
Generation and fitting stream NDJSON; `POST /api/run/<job-id>/stop` cancels either task. Existing job exclusion and
disconnect cancellation apply. Python entry points live in `ml_opt.regression`.

Method: [ordinary least squares and polynomial regression](https://scikit-learn.org/stable/modules/linear_model.html#ordinary-least-squares).
Checks: `python -m pytest -q tests/test_regression.py tests/test_regression_interface.py`
and `node --test tests/*.mjs`. Numerical and worker tests use native PETSc/PyTorch;
the UI tests import the production modules without rewriting their source.

### Ridge regression

In either regression view, select **Fitting method → Ridge regression** and set
**Ridge penalty (α)**. In the scaled-volume monomial basis, the fit minimises

\[
\sum_{i\in\mathrm{train}} \lVert y_i-\widehat y_i\rVert^2
+\alpha\sum_{j=1}^{p}\lVert b_j\rVert^2.
\]

The intercept is not penalised. For topology targets, each element's coefficients
are penalised independently. Training uses only the training cases to set the
input scaling, feature means and target mean. A centred economy SVD computes the
ridge solution without forming normal equations or a square matrix whose size
is the polynomial order. No additional numerical framework is required.

A positive α shrinks nonconstant coefficients and supports underdetermined
polynomial fits; α = 0 follows the ordinary least-squares path and its rank/data
checks. The objective uses the **sum**, not the mean, of squared training errors.
A larger α therefore has a stronger effect at a fixed case count. Ridge does not
guarantee lower test error, bounded scalar predictions or feasible topologies.
The penalty applies to the scaled coefficients, not the displayed original-volume
coefficients. Very high orders are still subject to floating-point precision,
available memory and computation time.

The fitting method and effective penalty are saved with both scalar and topology
model reports and restored independently in the two views. Reports written
before this option are treated as ordinary least squares. The test split and
mean-training baseline remain independent of the chosen method. Cancellation
preserves the previous saved artifact.

Method reference: [ridge regression](https://scikit-learn.org/stable/modules/linear_model.html#ridge-regression-and-classification).

### Neural-network regression

In **Regression → Compliance** or **Regression → Topology**, select **Fitting
method → Neural network**. Both use a fully connected PyTorch network with one
tanh hidden layer. Controls are hidden-neuron count (default 8), epochs (2000),
learning rate (0.01), weight penalty (0), network seed (42), and validation share
(20% of the cases remaining after test selection). Training is deterministic
full-batch Adam on CPU float64 tensors. An epoch is one optimiser update through
all fitting cases; epoch zero records the untrained network.

The test split uses the same whole-case selection as polynomial and ridge
regression. Validation cases are then selected only from the non-test cases,
keeping both volume endpoints in fitting. All three sets are disjoint. The input
range and compliance target mean/standard deviation are computed from fitting
cases only. Validation targets choose the saved checkpoint; test targets never
select weights, scaling or epochs. The interface marks validation cases
separately. The validation share refers to non-test cases, rather than the full
dataset. Six eligible cases remain the minimum for this introductory experiment.

Compliance predicts one scalar objective, using a linear output and training
normalised objective MSE. The label remains elastic work for prescribed-motion
datasets. Topology predicts all designable physical densities using shared hidden
features and sigmoid outputs. Fixed solid and void elements are restored exactly
and are omitted from the fitting loss. Topology training curves show mean squared
physical-density error over designable elements; final density accuracy metrics
include the full mesh. The optional weight penalty adds its coefficient times
the sum of squared hidden/output weight matrices to training MSE; biases are not
penalised. These MSE units differ from ridge's sum-of-squares objective.

Live training and validation curves retain up to roughly 200 sampled epochs,
plus the best validation epoch. Training runs for the requested number of epochs
and saves the weights with the lowest validation loss, including epoch zero when
appropriate. The dashed line marks that selected epoch. The displayed final
training/validation/test errors use the selected weights, whereas the curves
show the full training trajectory. Reducing validation loss, rather than using
test error to choose settings, avoids adapting the model to its final test set.

Polynomial/ridge and neural artifacts remain separate. The model selector keeps
both available, and training a network does not overwrite polynomial results.
Selecting a model updates its split labels, baseline, metrics and download link.
Comparison tables report each model's saved training, validation and test errors.
Matching test-share/split-seed settings give identical final test cases; a warning
appears when saved test splits differ. Neural fitting reserves additional
validation cases, so its fitting set is smaller than polynomial training under
the same outer split. Choosing many settings by repeatedly inspecting the test
column still risks test-set overfitting.

Compliance curves and topology previews evaluate the exported network weights
locally in the browser. Python inference uses the same tanh/linear/sigmoid map;
round-trip tests compare both implementations. Topology predictions can be checked
with **Check structure**, preserving the existing nominal-field PETSc analysis
and constraint checks. Sigmoid bounds do not guarantee volume feasibility,
connectivity, structural performance or optimality. No optimiser initialisation
or solver replacement is involved.

Each dataset saves its latest networks in `neural-compliance.json` and
`neural-topology.json`, alongside `models.json` and `topology-models.json`. Network
artifacts contain portable JSON weights, architecture, exact dataset fingerprint,
all splits, preprocessing, seed, training history and selected epoch. They use no
pickle/checkpoint deserialisation. Refresh/restart restores both model families.
Stop cancels training and preserves the previous saved artifact; switching views
keeps the current job running. Divergent nonfinite training fails without
replacing saved weights. Training and structural checks share the existing worker
lock and stream cancellation rules with other experiments.

The existing fit endpoints accept `method: "neural"` with `neurons`, `epochs`,
`learning_rate`, `weight_penalty`, `network_seed`, `validation_fraction`, and the
existing test split/quality settings. Polynomial degree and ridge settings are
ignored for neural training. `neural_training` NDJSON events carry epoch, training
loss, validation loss and elapsed time. Dataset reads include `neural_models`
with compliance/topology reports. Download at
`/api/regression/<id>/artifacts/neural-compliance` or `neural-topology`. The topology
evaluation endpoint accepts `model_key: "neural_network"` instead of `degree`.

References: [PyTorch neural-network models](https://docs.pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html)
and [training model parameters](https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html).

### Topology regression

Choose **Regression → Topology**, load an existing experiment (or generate one
from a saved optimisation setup), set the polynomial order and split, then click
**Train models**. Each element's physical density is a separate regression
output, with a shared polynomial feature matrix and training cases. Inputs are
scaled using the training volume range. There is no fixed order cap. Both
ordinary least squares and ridge use the existing case-quality filters. Whole optimisation cases are held out; test densities
never enter fitting or the mean-training-topology baseline. Matching split
settings give the same cases as compliance regression.

The volume slider previews predicted physical densities. The model selector
compares polynomial orders. Selecting a reference case moves the prediction to
that exact volume; the case list labels training, test and excluded cases. A
held-out case is selected first after training. Prediction and reference views
share the mesh and density colours. Absolute density difference uses a fixed
white-to-red scale from 0 to 1 and appears only at matching reference volumes.
Model comparison reports mean absolute density error after clipping, rather
than a percentage relative to void densities. Downloads also retain RMSE,
maximum case MAE, per-case errors and raw out-of-bounds fractions.

Predictions are clipped to [0, 1]. Fixed solid and void regions remain exactly
fixed. The interface reports the number of clipped elements and the resulting
physical volume. Clipping does not enforce the volume target, connectivity or
manufacturing constraints. Topology changes can make smooth polynomial
interpolation inaccurate even when training error is small.

**Check structure** evaluates the current predicted physical field through the
existing PETSc/PyTorch problem, preserving material, supports, independent load
cases and prescribed motion. Filtering and projection are disabled for this
check because the learned targets are already physical densities. The check
uses full-order analysis even if the dataset used a reduced model. At an exact
saved volume, the reference physical field is re-evaluated in the same way.
The report compares compliance (elastic work with prescribed motion), physical
volume, equilibrium residual and nominal constraint feasibility. For robust
experiments only nominal material is checked; this does not validate the
original eroded/dilated constraints. Structural checks do not certify optimality.
Changing the model or volume clears the prior check. These predictions do not
initialise or run optimisation.

Topology models persist in `outputs/regression/<id>/topology-models.json`, with
the exact dataset fingerprint, split, polynomial coefficients in scaled volume,
training range and metrics. Cancelled fits preserve the previous artifact.
Refresh/restart restores both kinds of saved models. Checks are interactive
results rather than persisted certificates.

API: `POST /api/regression/<id>/topology/fit` takes the same options as scalar
fitting. `POST /api/regression/<id>/topology/evaluate` takes `degree` and `volume`.
Both stream NDJSON through the shared cancellable worker; evaluation emits a
`topology_evaluation` event. Download the fitted artifact at
`GET /api/regression/<id>/artifacts/topology-models`. The dataset response includes
`topology_models` alongside the existing scalar `models`.

This direct per-element polynomial model is an introductory baseline. For a
research example that regresses compressed topology representations, see
[Garayalde et al., Real-time topology optimization via learnable mappings](https://arxiv.org/abs/2311.08473).

### Predicting compliance from a design

Choose **Regression → Design → compliance**. This experiment maps the full
physical density field to its structural objective, with the mesh, geometry,
material, supports and loads fixed. It is separate from volume-to-compliance and
volume-to-topology regression.

1. Select an existing regression dataset as **Base designs**. Generate at least
   three distinct designs in the existing views first if none are saved.
2. Set the number and size of density variations, then **Generate dataset**.
   The original designs and their smooth, bounded perturbations are all solved
   afresh. Fixed solid and void regions stay unchanged.
3. Train polynomial regression, ridge, or the existing one-hidden-layer tanh NN.
   Artifacts for all three methods coexist. Compare training and validation
   errors; keep test results for final assessment.
4. Select points in the actual-versus-predicted plot to inspect their designs.
   Click the density canvas or enter an element number, change its density and
   **Apply density**. Prediction updates immediately. **Check structure** obtains
   the edited field's actual objective; **Reset design** restores the saved case.
   Edits and checks are temporary and never change training labels.

Generation evaluates physical densities directly, with no second filter or
projection. Robust sources are evaluated as nominal material, rather than using
the original worst-field label. ROM sources are re-evaluated with the existing
full structural solver. Multiple load cases retain their saved combination.
Prescribed motion predicts elastic work. These evaluations measure the objective;
they do not certify the optimisation constraints or optimality.

All variations of one base design form a family. The test share is drawn from
whole families, and the validation share from the remaining families. At least
one family remains in each partition. Exact duplicate densities unite families,
so they cannot cross a partition. Polynomial, ridge and NN fits use identical
partitions for identical settings; the NN selects its checkpoint using validation
loss only. The comparison explicitly flags saved models with different splits.
The training-mean and volume-only linear baselines use fitting cases only.

To broaden the base layouts, set **Generation method → Optimise other cases**.
Select a saved regression experiment as **Original case**, or first prepare its
connections in Optimisation and click **Use current optimisation setup**. This
copies settings; the current result is not used as a training label.

The batch includes a fresh original-case optimisation and auxiliary cases with
clamped-edge or pinned/roller support patterns, random perimeter load nodes and
random force directions. Mesh, material, volume, filtering, projection and
connection regions stay fixed. Auxiliary mechanical constraints need not coincide
with the connection anchors. Each auxiliary case uses one force and a validated,
rigid-motion-stable support pattern; custom load-case combinations and prescribed
motion are retained only for the original optimisation and original-case evaluation.

Every final nominal density and its perturbations are analysed anew under the
**original** loads and supports. Auxiliary objective values never become training
labels. Families store the auxiliary setup, convergence and constraint feasibility
separately, including iteration-limited results; those can still supply valid
original-case forward-evaluation labels. The case inspector can switch the drawn
loads/supports between original and optimisation cases. Predictions and **Check
structure** always use the original setup, whichever overlay is selected.

Whole auxiliary-case families remain together in the split. This is a broader
layout test than adding noise to one volume sweep, but it does not establish
accuracy on arbitrary topologies. Keep a separate final test set when using
validation results to choose sampling settings. Cross-case generation runs through
the same cancellable worker; completed labels survive cancellation.

`POST /api/design-regression/cross-cases` accepts one of `source_id`, `problem`,
or `original_dataset_id` (reuse a cross-case experiment’s saved original setup),
plus `case_count`, `max_iterations`, `variants`, `amplitude` and `seed`. It emits
`design_optimization` progress and `design_case` labels, and saves a standard design
regression dataset with `source_kind: "cross_cases"` and per-family configurations.

**Fit log objective**, enabled by default, fits `log(C)` and exponentiates the
prediction. This keeps predictions positive and reduces domination by very soft
structures. NN target standardisation also uses fitting cases only. All reported
errors, including percentage error, refer to the physical objective. Without log
fitting, unconstrained predictions can be nonpositive; log plots omit those
points, while the linear plot and score table retain them.

For `m` designable elements, an explicit polynomial expansion can contain
`binomial(m+p,p)` features. The implementation evaluates the equivalent weighted
polynomial basis through its sample Gram matrix in PyTorch:

\[
K_p(r,s)=\left(\frac{1+\frac{1}{m}\sum_{j=1}^{m}(2r_j-1)(2s_j-1)}{2}\right)^p.
\]

This includes all element cross terms through order `p`. The binomial expansion
sets the monomial weights, so its ridge penalty has a different scale from the
one-dimensional volume experiment. Both features and targets are centred to
leave an intercept unpenalised. Polynomial least squares uses an eigendecomposition
pseudoinverse and the minimum feature-norm solution when underdetermined; ridge
uses `(K_centred + α I)⁻¹`. `α = 0` gives the same least-squares solution. The
number of polynomial features may exceed the number of designs, with no fixed
order cap. High orders can become numerically indistinguishable or overfit; a
larger order does not establish better generalisation. Memory grows quadratically
with the number of training samples instead of the expanded feature count.

The NN consumes every designable density, scaled to `[-1,1]`, using the existing
float64 PyTorch tanh/linear network and Adam trainer. The initial learning rate
is smaller here because every element contributes an input. No image-convolution model
or additional numerical framework is introduced. Portable JSON inference in the
browser matches the Python model. All methods are tied to the original mesh and
structural setup; a changed load, support or mesh requires a new experiment.

Design experiments are saved under `outputs/design-regression/<id>/`, with
`dataset.json` and independent `models-ols.json`, `models-ridge.json` and
`models-neural.json` artifacts. Refresh/restart restores datasets and models.
Cancelled generation keeps completed cases; cancelled fitting preserves the
previous artifact. Dataset fingerprints reject stale model files.

API: `POST /api/design-regression/generate` accepts `source_id`, `variants`,
`amplitude` and `seed`. `POST /api/design-regression/<id>/fit` accepts the existing
regression fitting options plus `log_target`; `include_unconverged` is irrelevant
because all labels here are direct structural evaluations. `POST
/api/design-regression/<id>/evaluate` accepts the full `density` list. All use the
shared cancellable worker and NDJSON streams. Evaluation emits a
`design_evaluation` event. Dataset/model downloads are exposed under
`/api/design-regression/<id>/artifacts/`.

Method references: [polynomial kernel](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.polynomial_kernel.html)
and [kernel ridge regression](https://scikit-learn.org/stable/modules/kernel_ridge.html).

## Offline ROM training

Open **Machine learning** and select **Reduced models** and use the left-side **Geometry**, **Loads**, **Material**,
**Filters**, and **Sampling** tabs to configure its independent mesh, right-edge load
heights (fractions of domain height), load angles, volume fractions, filter,
projection, iteration limit, and snapshot spacing. Comma-separated lists form
the Cartesian product of training cases. Click **Generate snapshots** to run
the existing full-order MMA/PETSc optimiser for each case. Initial states,
every selected outer iterate, and final states are captured; trial evaluations
are excluded. Loads are unit magnitude, rounded to the nearest right-edge
node. Material defaults and support conditions match the cantilever solver.

Progress shows actual full-order compliance, equilibrium residuals, and the
latest physical density. **Stop training** preserves collected snapshots as
a partial dataset. Training and cantilever optimisation share one worker and
cannot run simultaneously. Switching workspaces keeps a job running;
disconnecting its live stream cancels it. Snapshot generation has no fixed
mesh, case-count, snapshot-count, array-size or iteration ceiling. The form
estimates displacement/density storage before starting; the estimate is also
saved as `estimated_snapshot_bytes` in dataset metadata. Snapshots are still
held in memory, and saving/POD require additional working memory. Actual RAM,
disk space and runtime determine feasible job sizes. Numerical checks remain:
positive integer mesh sizes/iteration counts, finite parameters, valid load
heights and volume fractions, and distinct load positions after node snapping.
Optimisation mesh/domain inputs also accept larger saved ROM geometries.

Choose a saved dataset under **Build the reduced basis**, set the target POD
energy and maximum mode count, then click **Build reduced basis**. This is
uncentred displacement POD on free DOFs, using an economy SVD. Equal snapshot
weighting (the default) normalises each column before SVD, preventing large
displacements in soft designs from dominating. Turning it off uses the raw
snapshot magnitudes. Rank selection respects numerical rank and the mode
limit, and explicitly reports an unmet energy target when the cap is too low.
The spectrum and maximum per-snapshot relative reconstruction error describe
training displacements, not elastic strain energy or unseen-case accuracy.
Use **Validate on unseen cases** for held-out accuracy checks. To use a saved
basis in optimisation, choose **Reduced model** in Optimisation's **Solver** settings (see below). Offline snapshot generation and held-out reference
trajectories always use full-order state solves.

Datasets persist in `outputs/rom-training/<id>/`, relative to the source
checkout. Refreshing or restarting reloads the dataset library. Downloads:

- `snapshots.npz`: free-DOF displacement columns, corresponding physical
  densities, case indices, outer iteration numbers, compliances, residuals,
  free-DOF indices, mesh coordinates/connectivity, and JSON provenance.
- `basis.npz`: orthonormal free-DOF basis columns, free-DOF mapping, singular
  values, cumulative POD energy, original snapshot norms, reconstruction
  errors, packed reduced element stiffness operators, and JSON provenance.
  Clamped displacement components are zero.
- `metadata.json`: complete training cases, settings, completion state, and
  basis diagnostics. Load NPZ files with `numpy.load(..., allow_pickle=False)`.

A dataset can be used to rebuild a basis with different settings. Cancelled
builds leave the previous saved basis intact. The numerical workflow is also
available through `ml_opt.rom` (`SnapshotConfig`, `generate_snapshots`, and
`build_basis`). HTTP endpoints are `POST /api/rom/snapshots`,
`POST /api/rom/<id>/basis`, `GET /api/rom/trainings`, and
`GET /api/rom/<id>/artifacts/{snapshots,basis,metadata}`. Generation and building
stream NDJSON events; `POST /api/run/<job-id>/stop` cancels either stage.

### Held-out ROM validation

Select a dataset with a built basis, then configure independent load heights,
directions, and volume fractions under **Validate on unseen cases**. The mesh,
material, support conditions, filter, and projection are inherited from that
single-field training dataset. Every combination is checked against all
declared training cases after snapping to mesh nodes and normalising load
directions, including the equivalence of -180 and 180 degrees. Training
overlap and duplicate physical cases are rejected before a worker job starts.

Full-order MMA generates each new design trajectory. At the initial design,
every comparison interval, and the final design, validation solves the frozen
Galerkin system `(V.T K V) a = V.T f` and lifts `u_ROM = V a`. Validation never
adds snapshots to training, updates the basis, or uses ROM gradients to drive
these trajectories. The compared quantities are:

- Displacement: `||u_ROM - u_FOM||_2 / ||u_FOM||_2` on nodal DOFs (the fixed
  components are zero). This is a Euclidean norm, not an energy norm.
- Compliance: `abs(C_ROM - C_FOM) / abs(C_FOM)`, with `C = f.T u`.
- Design gradient: `||g_ROM - g_FOM||_2 / ||g_FOM||_2` for the raw design
  variables, through the same SIMP, Helmholtz-filter and projection autograd
  pipeline. The fixed-basis self-adjoint ROM derivative is checked against
  finite differences in the tests. A zero reference with nonzero difference
  is reported as undefined and cannot pass its tolerance.

The tab reports per-design comparisons, mean and maximum errors, selected
tolerances, ROM equilibrium residuals, and the number of designs passing all
three thresholds. A completed report is within tolerances only when every
sampled design passes; a stopped report is always partial. These are empirical
checks on the sampled held-out trajectories, not guarantees for arbitrary
material distributions or unsampled intermediate designs. Validation is
limited to 16 cases, 128 comparison states and 128 MiB of comparison arrays.

Reports persist under `outputs/rom-training/<id>/validations/<validation-id>/`.
`report.json` records the complete validation cases, settings, thresholds,
errors, and SHA-256 of the exact frozen basis artifact. `comparisons.npz`
contains raw designs, physical densities, full and reduced displacements and
design gradients, DOF mapping, case/iteration indices, and JSON provenance.
Cancellation preserves collected comparisons as partial evidence; cancelling
before a comparison leaves any previous report intact. Rebuilding the basis
invalidates the current validation link; prior files remain on disk.

The API adds `POST /api/rom/<id>/validate` (streamed),
`GET /api/rom/<id>/validation` and downloads at
`GET /api/rom/<id>/artifacts/{validation,comparisons}`. Python callers can use
`ValidationConfig` and `validate_basis` from `ml_opt.rom`.

### Optimise with a saved ROM

In **Optimisation**, open **Solver** and choose **Model: Reduced model** and select
a saved model. **Refresh models** reloads the training library. A model requires
a completed basis, but an absent or failed validation does not block explicit
selection; its validation status is displayed alongside the model. **Use model
mesh** applies its element counts and domain dimensions, preserving the load's
relative node position. The API independently rejects incompatible mesh,
geometry, Poisson ratio, thickness, DOF mapping, or malformed basis artifacts
before a worker job starts. Loads whose projection into the basis is effectively
zero are rejected rather than optimising an unrepresented force.

Both MMA and GCMMA use the frozen Galerkin state and its consistent adjoint
sensitivities at every evaluation, including GCMMA trials. The same material,
filter and projection differentiation path is used as with FOM. Robust
three-field optimisation uses the selected basis for all three material fields;
both sequential and parallel field execution are supported. Single-field
held-out validation does not certify robust-field or other untested accuracy.
Basis building also precomputes the exact reduced element stiffness matrices
`V_e.T K_e(E=1) V_e`. Each online solve assembles only their modulus-weighted
sum, then solves the small dense system. It does not build or project a full
stiffness matrix. Full displacement, element energies for the consistent
gradient, and a matrix-free full equilibrium residual are still evaluated.
Choosing a reduced model does not by itself guarantee a speedup.

The operators are stored as float64 upper triangles in `basis.npz`, together
with a format version and byte count in the basis metadata. Preparation uses
bounded blocks and rejects operator arrays above 512 MiB before allocation;
reduce the mode count if that limit is reached. Cancellation preserves the
previous saved model. Older basis artifacts remain supported: their operators
are prepared once at solver construction. Rebuild an older basis in Training
to move that cost offline. Material interpolation, density, and loads may vary;
the mesh, geometry, Poisson ratio, thickness, and frozen basis remain fixed.

The selected artifact's SHA-256 is pinned before the job and rechecked when
each solver loads it. Rebuilding a basis cannot change an active run's basis.
Model type, mode count, basis identity, and full equilibrium residual are
reported in the live interface and histories. ROM compliance, gradients, and
stopping decisions are approximate; reaching the stopping criteria does not
certify full-order accuracy. There is no implicit fallback to a full state solve.
The default remains **FOM**.

The larger-mesh ROM benchmark is reproducible with
`python benchmarks/rom_online.py`. It trains separate bases at 120×40 and
240×80, alternates FOM/ROM timing trials for MMA and GCMMA, checks the new
assembly against the previous Galerkin implementation, and evaluates each
final ROM design with FOM for displacement, compliance, and gradient errors.
It records offline training cost, online setup/loading, and timing-only
break-even run counts. The command uses benchmark-only datasets outside the
interactive training library; it does not change the training tab's mesh
limits. See [the measurements](../benchmarks/ROM_ONLINE.md) for scope and results.

Saved `summary.json` and `result.npz` identify the state solver and ROM basis.
CLI/Python configurations use `state_solver="rom"` and `rom_basis_path`:

```powershell
.\.venv\Scripts\python.exe -m ml_opt --nx 30 --ny 10 --state-solver rom --rom-basis-path outputs/rom-training/<id>/basis.npz
```

The web API takes `state_solver="rom"` and `rom_training_id=<id>` in
`POST /api/run`. Browser requests select only datasets within the app's training
root; arbitrary artifact paths are not accepted. Omit the selection or use
`state_solver="fom"` for the original full-order solve.

## 2D cantilever example

The example minimises compliance `C = f^T u` subject to `K(rho) u = f`,
`mean(rho) / volume_fraction - 1 <= 0`, and element density bounds `0 <= rho <= 1`.
The rectangular cantilever is clamped along its left edge and loaded downward
at the midpoint of its right edge by default. An odd number of elements through the height
splits the load equally between the two adjacent nodes.

Defaults: 60 x 20 bilinear quadrilateral elements, a 3 x 1 domain, plane stress,
unit thickness and load, Young's modulus 1, Poisson ratio 0.3, volume fraction
0.4, and SIMP penalisation 3. The material law is
`E(rho) = E_min + (E_solid - E_min) * rho**p`, with `E_min = 1e-6`.
The **Material** tab selects SIMP (`p >= 1`) or RAMP (`q >= 0`) and the
penalty value. RAMP uses
`E(rho) = E_min + (E_solid - E_min) * rho / (1 + q * (1 - rho))`;
`q = 0` and SIMP `p = 1` give linear interpolation. Both use the physical
density after filtering/projection, including all three robust fields.
See the [SIMP/RAMP material interpolation reference](https://doc.comsol.com/6.3/doc/com.comsol.help.opt/opt_ug_opt_interface.6.21.html).
Python uses `material_model="simp"` or `"ramp"` with `penalisation` for p or q;
CLI flags are `--material-model ramp --penalisation 3`. The material defaults
remain SIMP with p = 3; volume is never stiffness-penalised.
These are nondimensional benchmark parameters. The Python/CLI default
`filter_radius=0` and `projection_beta=0` retain the original unfiltered
benchmark; the visual interface initially uses `filter_radius=0.1`,
`projection_beta=2`, and `projection_eta=0.5`.

### Volume constraint modes

**Constraints → Volume constraint** selects `<`, `>`, or `=` for the physical
material volume target. `<` remains the default upper bound; `>` is a lower
bound and allows more material, including a fully solid solution in compliance
minimisation. Equality constrains the normalised error
`abs(V / V_target - 1)` to the configured feasibility tolerance (default `1e-4`).
The UI symbols use numerical upper/lower bounds with this tolerance, rather
than mathematically strict inequalities.

MMA/GCMMA receive signed upper/lower rows. Equality uses two smooth rows with
opposite gradients; the absolute error is used only for reporting and stopping.
The live volume target, convergence reference, history, and exported feasibility
follow the selected mode. In robust mode all modes target the existing **dilated
field's** volume; equality does not require all three fields to have equal volume.

Python: `volume_constraint="lt"`, `"gt"`, or `"eq"`.
CLI: `--volume-constraint lt|gt|eq`.

### Connections through pseudo-temperature

In **Optimisation → Constraints → Connections**, select points, lines or surfaces
and assign a **Connection anchor** and one or more **Connection targets**. The
**Point / Line / Surface** selector, node-coordinate fields, **Select nodes** and
assignment buttons are all inside Connections. Select a role, choose the nodes,
then **Assign anchor** or **Assign target**. These are
independent of the mechanical supports and forces. Each target must conduct unit
total heat to at least one selected anchor. Additional anchors act as alternative
sinks; they are not required to connect to one another. Only selected targets are
constrained, so this does not eliminate every possible unselected solid island.

The scalar auxiliary problem is

\[
-\nabla\cdot(k(\rho)\nabla T_j)=q_j,
\qquad T_j=0\text{ at anchors},
\qquad k(\rho)=\epsilon+(1-\epsilon)\rho^{p_k}.
\]

Exterior faces are insulated. Each target has its own density-independent,
unit-total source, so removing its material cannot turn its heat source off.
The existing PETSc backend solves a cell-centred finite-volume discretisation
with harmonic face conductivities. Only face neighbours exchange heat, avoiding
artificial diagonal connections. Surfaces use their enclosed cells; point/line
selections use adjacent cells. Anchor and target cells must not overlap, and
neither may include fixed void cells. Selections are stored in model coordinates
and revalidated on mesh changes.

For each target the constraint is

\[
g_j=\frac{\lVert T_j\rVert_{8,\Omega_j}}
{\lambda\lVert T_{j,\mathrm{solid}}\rVert_{8,\Omega_j}}-1\le0.
\]

The reference is solved on the fully solid mesh with the same sources and sinks.
**Maximum temperature ratio** is the allowed multiplier \(\lambda\), initially
5. A smaller multiplier demands a stronger conducting path. This is a normalised
8-norm response, rather than a literal maximum-temperature ratio. The norm bounds
each source-cell temperature but does not remove geometric or parameter sensitivity.
Conductivity exponent and void conductivity are in **Connection settings**.
The nonzero void conductivity keeps the solve defined for disconnected designs;
it also means the constraint is a surrogate, rather than a binary proof of connectivity.

The adjoint uses the same retained PETSc matrix factors. PyTorch propagates its
face-conductance derivatives through density interpolation, projection and the
Helmholtz filter. Fixed solid/void variables retain zero gradients. MMA and GCMMA
receive one smooth constraint per target, including in their feasibility and
convergence checks. Robust optimisation evaluates connectivity on eroded material.
Disabled connections add no diffusion solve.

The result **Display** selector switches between material density and each target's
normalised pseudo-temperature field. Its colour scale runs from zero to the
chosen temperature limit. The status separately reports the surrogate limit and
a geometric flood-fill check at physical density ≥ 0.5. This diagnostic requires
all target cells to be reachable from qualifying anchor cells through face-connected
material. A passing temperature constraint does not guarantee that the diagnostic
passes. A failure of either check should be inspected before treating a topology
as a suitable connected design. Projects preserve regions, settings, temperature
snapshots and geometric checks; older projects remain supported.

The implementation is a targeted solid-conduction adaptation of the virtual
scalar-field/temperature family, not a reproduction of the global void-removal
or nonlinear heat-source methods. Key references:

- [Li, Chen, Liu & Tong (2016), Structural topology optimization considering connectivity constraint](https://doi.org/10.1007/s00158-016-1459-5): the general virtual scalar-field approach and temperature constraints.
- [Hård, Wallin & Ristinmaa (2024), Connectivity Constraints Ensuring Continuous Electrodes in Topology Optimization of Electroactive Polymer](https://doi.org/10.1115/1.4064980): solid connectivity with a temperature norm and fully solid reference. Their density-dependent heating differs from the fixed target sources used here.
- [Luo, Sigmund, Li & Liu (2020), Additive manufacturing oriented topology optimization of structures with self-supported enclosed voids](https://doi.org/10.1016/j.cma.2020.113385): nonlinear virtual heating for enclosed-void identification; this extension is not implemented here.
- [Cool, Aage & Sigmund (2025), A practical review on promoting connectivity in topology optimization](https://arxiv.org/abs/2501.09402): comparisons, parameter sensitivity and the distinction between solid/void and periodic connectivity requirements.

### Morphological maximum solid size

In **Constraints → Maximum solid size**, enter the maximum disk **diameter** in
model units; `0` disables this optional constraint. The blue reference circle
updates on the mesh and topology. The purple Helmholtz-length reference keeps
its existing meaning. The diameter must exceed two element spacings in both
directions; refine the mesh to resolve a smaller target.

The solid-phase constraint follows section 5 of
[Lazarov and Wang (2017)](https://doi.org/10.1016/j.cma.2017.02.018).
A compact circular hat kernel of radius `R = max_solid_size / 2` averages the
physical density, followed by an erosion projection with threshold one:

```text
s = W_R rho_physical
e = 1 - tanh(b * (1 - s)) / tanh(b)
g_size = mean(e) / epsilon - 1 <= 0
```

Outside the rectangular design domain is void (zero extension). The defaults
are erosion sharpness `b = max_size_beta = 16` and allowed eroded fraction
`epsilon = max_size_tolerance = 0.001`. These remain separate from projection
β and its schedule. In robust mode the constraint acts on the dilated field.
The physical density used for stiffness and volume is unchanged by this test.

FFT convolution reuses the fixed kernel transform; its symmetric transpose
propagates the size gradient through projection and the Helmholtz adjoint.
MMA/GCMMA receive one additional constraint with no epigraph relaxation.
Live history, CSV and summary exports report the constraint and eroded fraction;
convergence and final feasibility include the size constraint.

This is a relaxed, mesh-discretised restriction. Gray densities and the finite
erosion sharpness/tolerance can permit local thickness deviations; the circle
is the target structuring element, not a measured thickness certificate.

```powershell
python -m ml_opt --filter-radius 0.04 --projection-beta 8 --max-solid-size 0.35
```

### Morphological maximum void size

**Constraints → Maximum void size** adds the matching void-phase option from
[Lazarov and Wang (2017), section 5](https://doi.org/10.1016/j.cma.2017.02.018).
Choose a disk diameter in model units; `0` disables it. Its green circle is
independent of the blue solid-size circle, and either or both constraints may
be enabled. The mesh-resolution requirements are the same as for solid size.

The constraint limits the void left after a material dilation at threshold zero:

```text
s = W_R rho_physical, R = max_void_size / 2
v_remaining = 1 - tanh(b_void * s) / tanh(b_void)
g_void = mean(v_remaining) / epsilon_void - 1 <= 0
```

Material is zero outside the design domain, preserving the existing exterior
void convention. Simply eroding `1 - rho` with zero padding would instead
introduce fictitious exterior solid. In robust mode the void constraint uses
the **eroded material field**, which has the largest void regions; the solid
constraint continues to use dilated material.

Void sharpness `max_void_beta=16` and allowed remaining fraction
`max_void_tolerance=0.001` are independently adjustable. Their gradients,
MMA/GCMMA trial checks, convergence, final feasibility, live summaries and
exports include the void constraint. These are relaxed size targets with the
same finite-sharpness and mesh limitations as the solid option.

```powershell
python -m ml_opt --filter-radius 0.04 --projection-beta 8 --max-void-size 0.8
# Both phases, with independently chosen diameters:
python -m ml_opt --filter-radius 0.04 --projection-beta 8 --max-solid-size 0.45 --max-void-size 0.8
```

### Helmholtz density filter

For a positive `filter_radius = ℓ`, the smoothed density solves
`rho_smooth - ℓ² ∇²rho_smooth = x` with zero normal flux on every domain boundary. Here `x`
is the bounded element design variable. A cell-centred finite-volume
discretisation uses the actual x/y mesh spacings, including rectangular cells.
The formulation preserves constant densities and material volume and keeps
filtered densities within the input range, up to floating-point roundoff.
The material interpolation, volume constraint, saved density plots, and live topology frames all
use the final physical density after any projection. Saved `design` arrays remain the raw
optimiser variables, while saved `density` arrays hold the physical result.

PETSc factors the fixed filter matrix once per problem and reuses those
factors for the forward filter and its transpose adjoint. PyTorch propagates
compliance and volume gradients back through the filter to the design
variables. As with the elasticity bridge, this supports first derivatives on
CPU float64 tensors.

### Heaviside projection

The pointwise normalised smooth Heaviside projection follows Helmholtz
smoothing, using the formulation of
[Wang, Lazarov, and Sigmund (2011)](https://doi.org/10.1007/s00158-010-0602-y):

```text
rho_physical = [tanh(beta * eta) + tanh(beta * (rho_smooth - eta))]
               / [tanh(beta * eta) + tanh(beta * (1 - eta))]
```

`projection_beta >= 0` controls sharpness; 0 bypasses the projection exactly.
`0 < projection_eta < 1` sets its threshold. The UI starts at β = 2, η = 0.5;
the Python/CLI default β = 0 preserves the previous benchmark. Fixed β is
the default; eta stays fixed throughout the run.

In **Filters → β schedule**, select **Scheduled increase** to multiply β after
each full interval of accepted optimisation iterations, capped at the chosen
maximum. Defaults are an interval of 25, multiplier 2, and maximum β = 16;
the sharpness input gives the starting β and must be positive for scheduling.
For a start of 2, iterations 0–25 use β = 2, 26–50 use 4, 51–75 use 8, and
iteration 76 onward uses 16. A short iteration limit can stop before the
maximum is reached; increase the iteration limit to allow the schedule to finish.

At each increase the current design is re-evaluated at the new β before
stepping, and MMA/GCMMA asymptotes, scaling, and convergence patience restart.
Convergence stopping is enabled only at the maximum β. Robust optimisation
updates all three projections together and reinitialises its epigraph bound.
The active sharpness is recorded in each history row and in `summary.json`
as `final_projection_beta`; the saved config retains
the starting β. Scheduling does not modify the Helmholtz length or penalty.

```powershell
python -m ml_opt --material-model ramp --penalisation 3 --filter-radius 0.1 `
  --projection-beta 2 --projection-schedule geometric --projection-beta-interval 25 `
  --projection-beta-multiplier 2 --projection-beta-max 16 --max-iterations 100
```

Projection is generally not volume preserving, so the volume constraint uses
`mean(rho_physical)`, not the raw or smoothed density mean. PyTorch carries
both compliance and volume sensitivities through the projection and the
Helmholtz adjoint. The live frames and saved `density` array contain exactly
the physical density used for the state solve.

ℓ is the coefficient length in the PDE. The displayed circle has radius ℓ;
for ℓ = 0.1 it shows radius 0.1 and diameter 0.2. This is a smoothing-length
reference, not a guaranteed member thickness. The Helmholtz influence extends
beyond the circle. For comparison with a classical finite-support linear
filter, its conventional approximate equivalent radius is `R ≈ 2√3 ℓ`;
this larger R is not used for the visual circle. See
[the DTU filter comparison, Eq. 17](https://www.topopt.mek.dtu.dk/-/media/subsites/topopt/apps/dokumenter-og-filer-til-apps/topopt88.pdf).
The Helmholtz PDE with homogeneous
Neumann boundary conditions follows the approach of
[Lazarov and Sigmund (2011)](https://orbit.dtu.dk/en/publications/filters-in-topology-optimization-based-on-helmholtztype-different/).

Element contributions update a preallocated PETSc sparse matrix through COO
assembly and use PETSc's sequential LU solver. PyTorch differentiates the SIMP/RAMP material law and
volume constraint. The custom compliance operator supplies the implicit,
self-adjoint derivative `dC/dE_e = -u_e^T K_e(E=1) u_e` without differentiating
through LU. It supports first derivatives on CPU float64 tensors.

### Run

After installing the project and dependencies:

```powershell
.\.venv\Scripts\python.exe -m ml_opt --optimizer mma
.\.venv\Scripts\python.exe -m ml_opt --optimizer gcmma
.\.venv\Scripts\python.exe -m ml_opt --optimizer mma --filter-radius 0.1
.\.venv\Scripts\python.exe -m ml_opt --optimizer gcmma --filter-radius 0.1 --projection-beta 4 --projection-eta 0.5
```

The equivalent console command is `toptvis-cantilever --optimizer mma` with
the environment activated. For a smaller mesh:

```powershell
.\.venv\Scripts\python.exe examples\cantilever.py --nx 30 --ny 10 --optimizer gcmma --max-iterations 100
```

The load can also be changed from the command line, for example:

```powershell
.\.venv\Scripts\python.exe -m ml_opt --load 2 --load-angle -60 --load-node-x 45 --load-node-y 12
```

Each run writes `result.npz` (design, density, mesh, displacement, force),
`history.csv`, `summary.json`, `density.png`, and `convergence.png` to
`outputs/cantilever-mma` or `outputs/cantilever-gcmma`. Override the destination
with `--output PATH`. Run `python -m ml_opt --help` for the remaining parameters.
The history includes iteration zero and records quantities at the same design.

The default move limit is 0.2 per iteration. Stopping requires stable maximum
density change, relative objective change, and volume feasibility over five
consecutive iterations after iteration 20. A run can also reach its iteration
limit; `summary.json` reports the actual stop reason and feasibility separately.
This stopping test does not certify KKT stationarity or a global optimum.

### Three-field robust formulation

Enable `robust=True` in `CantileverConfig`, or `--robust` on the command line.
One Helmholtz-filtered design is projected at three thresholds:
`eta_eroded = projection_eta + robust_eta_offset`,
`eta_nominal = projection_eta`, and
`eta_dilated = projection_eta - robust_eta_offset`. The default offset is 0.1.
All thresholds must lie strictly inside (0, 1), and robust mode requires
`projection_beta > 0`. A positive filter length is recommended for spatial
regularisation. Threshold offsets are density thresholds, not geometric
manufacturing tolerances in model units.

```text
minimise_x,t  t
subject to C_i(x) <= t, i = eroded, nominal, dilated
           mean(rho_dilated(x)) / volume_fraction - 1 <= 0
           K(rho_i(x)) u_i = f, i = eroded, nominal, dilated
           0 <= x <= 1, t >= 0
```

For the same fixed load and monotone SIMP/RAMP material law, the eroded field is
the worst compliance case. All three states are nevertheless evaluated and
reported. MMA and GCMMA solve the epigraph using mmapy's native nonnegative
auxiliary variable `z`, with `t = C_ref * z`. `C_ref` is the initial worst
compliance at the start of each β stage and stays fixed within that stage. The density-dependent objective is zero,
`a0 = 1`, and the four rows are `[C_eroded/C_ref, C_nominal/C_ref,
C_dilated/C_ref, volume_constraint]` with `a = [1, 1, 1, 0]`.
Thus `z` couples to every compliance constraint and cannot relax the volume
limit. mmapy's usual penalised feasibility slacks remain in the subproblems;
the evaluated physical volume and epigraph residual are checked separately.
All three smooth compliance gradients pass through the projection and shared
Helmholtz adjoint, with no extra elasticity solve in backward. GCMMA checks
conservativeness for every compliance row and the volume row at each trial.
The maximum compliance remains a reporting metric, and its active gradient
is available for diagnostics; it is not the gradient used to optimise the
epigraph. Thresholds and the dilated volume limit remain fixed during the run;
optional β continuation follows the schedule described above, without volume rescaling.

```powershell
# Run from this checkout, using its source when sharing another worktree's environment.
$env:PYTHONPATH = Join-Path $PWD 'src'
python -m ml_opt --robust --filter-radius 0.1 --projection-beta 4 --field-execution sequential
python -m ml_opt --robust --filter-radius 0.1 --projection-beta 4 --field-execution parallel
```

In the interface, open **Filters**, select **Three fields · robust**, and choose
**Sequential** or **Parallel · 3 processes**. The live material selector shows
each field and its compliance and volume. The headline metrics always show
worst compliance and dilated volume, whichever field is displayed. The volume
target therefore permits less nominal material than a single-field run.

`Evaluation.density`, the ordinary snapshot callback, `density.png`, and the
saved `density` array use nominal material. `Evaluation.displacement` and the
saved `displacement` use the worst field; `volume_fraction` is the dilated
volume used in the constraint. The evaluation also exposes `field_densities`,
`field_compliances`, `field_volumes`, `field_displacements`, and `worst_field`.
The optional state snapshot callback receives a matching displacement/density
pair for the worst field in robust mode, suitable for equilibrium snapshots.
Robust outputs include `density_eroded/nominal/dilated` and corresponding
`displacement_*` arrays in `result.npz`, a `robust-fields.png` comparison,
per-field history columns, and field metadata in `summary.json`. History and
summary also record `epigraph_compliance` (t) and the normalised residual
`(max(C_i) - t) / C_ref`. The stopping check requires this residual and the
volume constraint to meet the feasibility tolerance. Relative objective change
uses `t` in robust mode and compliance in single-field mode. MMA iterates may violate
the true epigraph inequality before convergence; GCMMA uses conservative trial
approximations. These checks do not certify KKT stationarity. State-solve
counts include every field and all GCMMA trial evaluations; residuals report
the maximum across fields and linear iterations sum across fields.

**Parallelisation:** sequential mode reuses one PETSc solver for the three
states. Parallel mode uses three persistent spawned processes, each owning
its own `COMM_SELF` matrix, vectors, KSP, and LU factors. This works with the
bundled Windows MPIUNI build. Only NumPy moduli, states, and element energies
cross process boundaries; the PyTorch graph stays in the parent. The filter
is solved once per evaluation in the parent. Workers limit PyTorch threads
to one to reduce oversubscription, following the
[PyTorch multiprocessing guidance](https://docs.pytorch.org/docs/stable/notes/multiprocessing.html).
User scripts selecting parallel mode must call optimisation inside an
`if __name__ == "__main__":` guard. Prefer sequential mode for small meshes:
process startup, array transfers, and three copies of the solver consume time
and memory, so parallel mode is not an automatic speedup. Both bridges still
require CPU float64 and support first derivatives only.

Reproduce cold and warm evaluation timings, including all field gradients and
checking mode agreement, with `python benchmarks/robust_fields.py --nx 60 --ny 20`.

For larger distributed meshes, a natural next implementation is an MPI-enabled
PETSc build with three communicator groups, one per material field, and a
distributed matrix/KSP within each group. PETSc provides
[subcommunicators](https://petsc.org/release/manualpages/Sys/PetscSubcommType/)
and [parallel KSP/AMG solvers](https://petsc.org/release/manual/ksp/).
That would require distributing the mesh, assembly, filter, and gradients;
the current process option parallelises independent fields, not the DOFs of
one state solve. PyTorch DDP does not distribute this custom PETSc operator.

### Optimiser choice

**MMA is the default** for this basic, single-load compliance problem. It takes
one moving-asymptote step per outer iteration. **GCMMA** adds trial evaluations
and updates the approximation curvature until the objective and constraint
approximations are conservative, or the inner iteration limit is reached.
It is useful when more conservative updates
are wanted and can require more state solves. Both use the same volume
constraint, move limit, and initial-compliance scaling.

Both backends use [`mmapy` 0.3.1](https://github.com/arjendeetman/GCMMA-MMA-Python),
which is distributed under **GPL-3.0-or-later**; its implementation remains an
external dependency. The backend credits Krister Svanberg's original MMA/GCMMA
methods and Arjen Deetman's Python implementation. If GCMMA reaches its inner
iteration limit without a conservative approximation, it accepts the last
evaluated trial and continues. That step is not certified conservative. History
records `gcmma_inner_limit_reached`, the interface reports the fallback, and the
CLI marks its inner count with `*`. Feasibility and convergence still use the
actual evaluated objective and constraints.
In the interface, select GCMMA in **Optimiser** to set its **inner iteration
limit** (default 30). This counts retries after the initial trial, so a limit
of 30 permits up to 31 trial solves per outer iteration. The minimum is 1;
MMA does not use this setting.

For the default 60 x 20 mesh, a local run gave the following results. Both
met the documented stability criteria and retained a physical volume fraction
of 0.40000; these are local designs, not certified global optima.

| Optimiser | Initial compliance | Final compliance | Outer iterations | State solves |
| --- | ---: | ---: | ---: | ---: |
| MMA | 1841.457 | 232.349 | 89 | 90 |
| GCMMA | 1841.457 | 240.175 | 50 | 121 |

## Layout

```text
src/ml_opt/config.py          Benchmark and optimisation parameters
src/ml_opt/fem/               Mesh, Q4 integration, PETSc solve/autograd bridge
src/ml_opt/problem.py         SIMP objective, volume constraint, sensitivities
src/ml_opt/projection.py      Differentiable smooth Heaviside projection
src/ml_opt/optimizers/        MMA/GCMMA state and inner iterations
src/ml_opt/runner.py          Outer optimisation and stopping criteria
src/ml_opt/rom/               Offline snapshots, POD, and training artifacts
src/ml_opt/output.py          Result files and figures
src/ml_opt/cli.py             Command-line entry point
src/ml_opt/web/               Local HTTP server, solver worker, streaming API
src/ml_opt/web/static/        Visual interface and canvas mesh/topology views
src/ml_opt/web/static/convergence.js  Live convergence plots and iteration inspection
examples/cantilever.py        Runnable cantilever example
tests/                       Numerical and optimisation tests
scripts/                     Environment setup and validation
pyproject.toml               Package metadata and dependencies
```

## Development

The declared dependencies are NumPy, PyTorch (`torch`), PETSc's Python
bindings (`petsc4py`), SciPy, Matplotlib, and `mmapy`. SciPy provides the
optimiser backend's dense linear algebra; state solves and sparse assembly
are performed by PETSc.

Run the numerical tests with:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[dev]" --no-build-isolation
.\.venv\Scripts\python.exe -m pytest -q
```

When a Git worktree shares another checkout's environment, set
`$env:PYTHONPATH = Join-Path $PWD 'src'` before running Python or tests, so
the worktree's source is used. The launcher sets this automatically.

### Local Windows environment

Verified locally on 14 November 2026: Python 3.14.4, PyTorch 2.15.0+cpu,
NumPy 2.5.4, PETSc 3.26.1, and petsc4py 3.26.1. PETSc's bundled check example,
the Python environment check, and `pip check` passed. The local package
snapshot is saved in `.venv/environment-freeze.txt`.

The setup targets 64-bit Windows Python 3.14 and Visual Studio C++ Build Tools.
Run from PowerShell:

```powershell
.\scripts\setup_windows.ps1
```

The script installs Python dependencies in `.venv`, downloads local Cygwin
build tools, and compiles PETSc 3.26.1 with MSVC. Cygwin root registration and
shortcuts are disabled. PETSc uses static libraries, the MSVC DLL runtime,
real double precision, 32-bit indices, and MPIUNI for single-process execution.
BLAS/LAPACK is built from PETSc's `f2cblaslapack` package. PyTorch uses the CPU
build. Distributed MPI and GPU support require a separate configuration.

`stage_petsc_windows.py` copies the compiled headers and libraries into
`.venv/petsc` and converts the linker metadata for native Windows Python.
The original Cygwin build configuration is retained. The setup then builds
`petsc4py`, installs this project in editable mode, and checks the environment.

Activate the environment, or run its interpreter directly:

```powershell
.\.venv\Scripts\Activate.ps1
.\.venv\Scripts\python.exe scripts\check_environment.py
```

The check exercises PyTorch autograd and a PETSc conjugate-gradient solve.
Build logs are kept in `.venv/build-native-petsc.log` and
`.venv/build-petsc4py.log`. Source files and compiler configuration are retained
in `.venv/build/petsc-3.26.1`. After a successful PETSc build, use
`setup_windows.ps1 -SkipPetscBuild` to restage PETSc and rebuild the bindings.

The native build follows the [PETSc Windows installation guide](https://petsc.org/release/install/windows/)
and [petsc4py source installation instructions](https://petsc.org/release/petsc4py/install.html).

### Local macOS environment

Create a local environment using Python 3.11 or newer, then install the project
and interface dependencies. This requires real double-precision PETSc and
`petsc4py` for your platform, as described under Other environments below.

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -e ".[ui]"
```

Double-click **Start TOptVis.command** in Finder. It opens Terminal, starts the
local server, and opens your default browser when ready. Keep Terminal open
while using the app; press **Control-C** to stop the server. If the environment
is missing or startup fails, the launcher displays the error and waits for
Return before closing.

The launcher is executable in Git. If a downloaded copy has lost that permission,
restore it from the project folder:

```sh
chmod +x "Start TOptVis.command"
```

You can also run it from Terminal and pass launcher options:

```sh
./"Start TOptVis.command" --port 8001
```

### Other environments

Build or install real double-precision PETSc and `petsc4py` for your platform,
then install this package and its remaining dependencies:

```sh
python -m pip install -e .
```

Generated results belong in `outputs/` and model checkpoints in `checkpoints/`;
both directories are excluded from Git.
