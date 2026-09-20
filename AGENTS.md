# System-A Agent Boundaries

## Workspace Boundary

All project activity must remain inside this repository directory:

`/home/davidhatley/Projects/research/system_a`

- Do not read from or write to parent directories.
- Do not use `/tmp`, `/tmp/opencode`, or any other external temporary directory.
- Do not create ephemeral work that is discarded after the task.
- Put source checkouts and downloaded datasets under `data/`.
- Put dependency and model caches under `data/cache/`.
- Put virtual environments under `.venv/`.
- Put commands, measurements, logs, checksums, and generated results under `artifacts/`.
- Put durable research conclusions under `research/`.
- Put reusable project code under `scripts/` or the relevant experiment directory.

If a required tool cannot operate within this boundary, stop and document the blocker rather than requesting broader filesystem access.

## Research Record

Every substantive task must leave a durable record in the repository. Reports must distinguish verified measurements from inference and unknowns. Record commands and external source revisions needed to reproduce the work.

Do not silently delete intermediate evidence. Large downloaded model weights, datasets, caches, and virtual environments may remain local and ignored by Git, but their provenance and checksums must be recorded in tracked reports or manifests.

## Compute Boundary

- Do not start training beyond an explicitly bounded smoke test.
- Do not use cloud compute or paid services without user authorization.
- Do not run destructive cleanup commands.
- Preserve unrelated concurrent changes.
