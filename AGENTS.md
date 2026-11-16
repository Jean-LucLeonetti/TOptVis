# Agent instructions

## Numerical backend

- PETSc and PyTorch are the established numerical backend. Extend them for
  feature work; do not replace them or introduce another numerical framework.
- Any change to this backend requires the user's explicit approval first.

## Feature work and parallel agents

- Start every new feature in a new Git worktree with its own branch before making changes.
- When agents work in parallel, give each agent a separate worktree and branch so their changes remain isolated.
- Preserve existing uncommitted changes in other worktrees.
- After merging a feature branch, push the target branch to its remote and verify that the merged commit is reachable from the remote target branch before cleanup.
- Once that verification succeeds, remove the feature's clean worktree and delete the local feature branch and any corresponding remote feature branch.
- Before cleanup, verify that no unmerged commits, uncommitted changes, or ongoing agent work remain. Never force removal or discard work; report any cleanup that cannot be completed safely.

## Git naming and attribution

- Use descriptive branch names such as `feature/...` or `fix/...`, without tool or assistant branding.
- Do not add tool or assistant references to branch names, commit messages, commit trailers, or other Git metadata.

## Public documentation and licensing

- Keep the main `README.md` concise, polished, and suitable for a public GitHub repository: a clear project description, key capabilities, essential setup, documentation links, and licensing.
- Keep detailed usage, numerical derivations, benchmark reports, and implementation notes in linked documentation rather than expanding the main README into a manual or changelog.
- Use a permissive license for original project code. Preserve third-party licenses and attribution, and clearly document any copyleft components and their distribution requirements; never describe the entire application as permissively licensed while those requirements apply.

## Product and interface

- TOptVis is an interactive demonstrator for topology optimisation and relevant
  optimisation and ML methods. Prioritise working experiments, clear visual
  comparisons, and links to the research behind each method.
- Keep the interface concise and professional. Use clear headings, readable
  labels, consistent alignment and spacing, and a restrained visual hierarchy.
- Do not add decorative microcopy, slogans, onboarding instructions, technology
  credits, or redundant badges and parameter summaries to the workspace.
- Keep solver names and technical parameters in their relevant settings. Do not
  repeat FOM, LU, full-order elasticity, projection values, continuation
  multipliers, or similar implementation details around the canvas or toolbar.
- Show parameter values once, in their controls. Use plain-language labels where
  possible. Put optional explanations in contextual help or documentation.
- Preserve meaningful results, legends, validation feedback, run status, errors,
  and the controls needed to use each method. Simplifying presentation must not
  silently change numerical behaviour or remove existing capabilities.
- Use readable text sizes and responsive layouts; do not shrink labels into
  tiny captions to make the interface fit. Preserve keyboard navigation and
  accessible names when removing explanatory text.
