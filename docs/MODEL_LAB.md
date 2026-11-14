# Model Lab

Open **ML Models** beside Cantilever 2D and ROM Training. This is a separate
workspace for training, comparing, and trying models. It shares the saved
full-order snapshot library with ROM Training, while model checkpoints and
experiment reports live under `outputs/model-lab/<model-id>/`. Existing ROM
bases and their validation are not overwritten. No pretrained weights or
external services are required.

## Models

Every entry has an implemented training and inference path:

| Family | Implementation |
| --- | --- |
| POD Galerkin | Euclidean displacement POD and exact reduced element assembly |
| Energy-weighted POD | Snapshot Gramian in the full-solid stiffness inner product, then Euclidean QR of the selected span for Galerkin assembly |
| Local POD | Deterministic clustering of log-modulus layouts, with a separate POD basis and packed operators in each populated cluster |
| Adaptive POD | Offline POD initialisation; residual-triggered FOM fallback, with basis enrichment at accepted outer optimisation states |
| POD + MLP | A small coefficient network and a fixed displacement basis fitted only on training cases |
| CNN | Spatial convolution blocks and a nodal displacement output |
| U-Net | A two-level encoder/decoder with spatial skip connections |
| FNO | Complex Fourier weights on positive/negative modes plus local convolution blocks |
| DeepONet | A convolutional branch encoder and a coordinate trunk network |
| Mesh transformer | Pooled field tokens, transformer encoding, and node-to-token cross-attention |

These are compact experimental reference architectures. The transformer is
not a pretrained Transolver or a claim to reproduce its published benchmarks.
Each trained model is tied to one mesh, domain, clamped-left support pattern,
Young/minimum modulus, Poisson ratio, and thickness. Loads, density layouts,
and material interpolation can vary, with accuracy determined by coverage.
The present workflow uses single-field elasticity. Models do not transfer
automatically across meshes or support/geometry changes.

## Train

1. Choose an existing snapshot dataset or expand **Generate a shared FOM
   dataset**. The generator uses the same FOM capture workflow as ROM Training.
   The defaults include multiple load heights, angles and volume fractions.
2. Choose a family, split fraction and random seed. At least two independent
   cases must actually be present in the snapshots. Duplicate physical cases
   stay in the same split; load magnitudes are grouped together because the
   neural target is normalised by load magnitude.
3. For neural models, set epochs, batch size, learning rate and width. Advanced
   controls expose applicable architecture parameters, early-stop patience,
   strain-energy and equilibrium loss weights. Displacement loss is always
   used. CPU and Auto work in the existing environment; CUDA is enabled only
   when the installed PyTorch build and hardware support it. Local evaluation
   and comparisons currently use CPU, including for GPU-trained checkpoints.
4. Train or build. The curve distinguishes the training objective from
   case-validation displacement MSE. Neural checkpoints are selected by the
   latter. Entire trajectories remain in one split; adjacent iterates are not
   randomly divided between training and validation.

Neural inputs contain nodal material-modulus and log-modulus fields, unit
load components, the clamped mask and normalised coordinates. Displacement
targets and the fitted scale use only training cases and unit load magnitude.
The output is multiplied by the load magnitude and clamped exactly at the
fixed edge. Neural gradients in raw mode use autograd through this complete
model and the material/filter/projection chain. The FEM energy adjoint is
not substituted for the derivative of an unconverged neural prediction.

The strain-energy loss compares prediction error in the current element
moduli; it differs from energy-weighted POD's fixed full-solid reference
metric. Equilibrium loss scatters internal element forces and excludes
clamped reactions. Validation checkpoint selection is not an independent
test result or an optimisation accuracy certificate.

Training preserves a best checkpoint. Stop retains that checkpoint when at
least one epoch has completed; its record is labelled cancelled and remains
usable. Stopping before a checkpoint exists leaves a record without an
inference model. Starting another fit creates another model ID; it does not
overwrite the inspected model. Automatic resume is not implemented.

## Compare against FOM

Check models in **Saved models**, then set independent load heights, angles,
volumes and error tolerances. **Compare checked models** applies the same
cases to every model; **Validate inspected model** tests one. Models must
share mesh, physics and filter/projection settings for a batch comparison.
The API rejects cases overlapping any selected model's source dataset,
including source cases used for checkpoint selection.

FOM MMA generates the unseen trajectories. At each sampled design, the
report compares displacement, compliance and raw-design gradient. Choose
raw, FEM-corrected, or both. The library shows errors, tolerance counts,
actual FOM solve counts, and a case-signature hash. Only rows with matching
case signatures represent the same comparison settings.

FEM-corrected mode uses the model displacement as the initial guess of a
PETSc CG/GAMG solve, checked by the existing full-equilibrium tolerance. If
the iterative configuration fails, it switches to LU and records the fallback.
Correction is a full-order physics solve; its iterations, calls and time
are included. A corrected model passing does not establish raw-model
accuracy or a speedup. Adaptive POD's raw mode can also use FOM fallback;
these calls are counted. Validation never enriches the adaptive basis or
updates weights.

Matching the force residual alone is not a certified displacement,
compliance or gradient error bound for these ill-conditioned material
layouts. The independent FOM comparisons remain the accuracy evidence.

## Try optimisation

Inspect a saved model and use **Try an optimisation**. Both MMA and GCMMA
are available. FEM correction is the default. Raw-model optimisation is
explicitly experimental; a nonpositive raw compliance stops the trial with
a message recommending more training or correction. A low raw-model
stopping metric does not certify the physical design.

Trials use one material field, the training mesh, fixed training projection
beta and filter, and no solid/void maximum-size constraints. Adaptive basis updates occur
only after accepted outer states, including the initial state; GCMMA trials
never mutate the basis. An enriched span includes the accepted FOM state,
so its state and physical gradient at that accepted design are preserved.
Local POD selection is discrete, with gradients valid for the selected
fixed basis away from cluster switching boundaries. No convergence guarantee
is claimed for switching raw surrogates.

Every final design, including the last design of a cancelled run, receives
an independent FOM displacement/compliance/gradient check. The report gives
optimisation time separately from the final FOM check, full-order volume,
prediction count, FOM solve count, correction iterations, LU fallbacks, and
adaptive basis updates. Raw neural histories identify the architecture and
mode rather than labelling their calls as FOM. Experimental models are tried
inside this workspace; Cantilever 2D retains its existing FOM/ROM selection.

## Persistence and limits

- `metadata.json`: architecture/options, source snapshot and case hashes,
  split membership, loss history, status, best epoch and report summaries.
- `model.pt`: tensor-only state dictionaries or bases/packed operators,
  plus configuration/provenance; loaded with `torch.load(weights_only=True)`.
  A SHA-256 is pinned before inference jobs. The portable download needs its
  metadata; keep both downloaded artifacts together for Python reuse.
- `validations/<id>/report.json` and `arrays.npz`: cases, errors, timings,
  modes, physical designs, both displacement fields and both gradients.
- `experiments/<id>/report.json` and `arrays.npz`: labelled optimisation
  history, final design/density, model and FOM results, and the enriched
  runtime basis for adaptive POD. Runtime enrichment does not overwrite the
  offline checkpoint or silently become training data.

The shared generator retains its limits of 24 cases, 512 snapshots and
128 MiB. Neural training is bounded to 6,000 nodes. Combined packed operators
are limited to 512 MiB per model; adaptive enrichment stops at its configured
rank/storage limit and can continue with counted FOM fallbacks. Validation
allows 16 models per batch, at most 128 comparison designs per mode and
128 MiB of saved arrays per model. All jobs use the same serialized worker
and Stop action as the existing workspaces; leaving a stream cancels its job.

Routes: `GET /api/ml/catalog`, `GET /api/ml/models`,
`GET /api/ml/models/<id>`, `POST /api/ml/train`,
`POST /api/ml/models/<id>/validate`, `POST /api/ml/compare`,
`POST /api/ml/models/<id>/experiment`,
`GET /api/ml/models/<id>/reports/{validation,experiment}`, and
`GET /api/ml/models/<id>/artifacts/{model,metadata,validation,validation_arrays,experiment,experiment_arrays}`.
Task responses are streamed NDJSON. Client requests use server-assigned IDs;
arbitrary checkpoint or dataset paths are not accepted.
