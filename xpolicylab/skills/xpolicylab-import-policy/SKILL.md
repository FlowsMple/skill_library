---
name: xpolicylab-import-policy
description: Import, reproduce, and integrate external robot/RL/VLA policy repositories into this XPolicyLab project. Use when the user provides a policy Git repository or local source path, asks to add or adapt a policy under XPolicyLab/policy, needs model.py/deploy.py/deploy.yml/__init__.py wrappers, or wants policy reproduction, dependency capture, data/action/observation adaptation, and validation.
---

# XPolicyLab Import Policy

Turn an external policy repository or local source tree into a project-native policy under `XPolicyLab/policy/<PolicyName>/`. Preserve upstream source as a snapshot, add thin XPolicyLab adapters, keep policy-specific environments isolated, and validate the local debug deployment path before simulator use.

## Load References

Read these only when needed:

- `references/xpolicylab-policy-contract.md`: project import path, required files, environment contract, model interface, image/data/action standards, debug client gate.
- `references/import-workflow.md`: phase checklist for importing, adapting, documenting, validating, and reporting blockers.

Always inspect local code before editing:

- `XPolicyLab/model_template.py`
- `XPolicyLab/policy/setup_policy_server.py`
- `XPolicyLab/policy/DP` as the default reference policy unless the user names another reference
- `XPolicyLab/policy/demo_policy` for the minimal adapter shape

## Non-Negotiable Rules

- Do not overwrite an existing policy folder unless the user explicitly requests it.
- Commit the freshly scaffolded policy directory in the XPolicyLab git repository before adaptation edits. This is a hard stop gate: if the commit is blocked, git is unavailable, or the user declines committing, report the blocker and stop immediately. Do not edit adapters, scripts, configs, docs, or upstream code after this failure unless the user explicitly instructs you to continue.
- Do not use Docker, Docker Compose, dev containers, or container-only setup paths.
- Do not run dependency installation commands unless the user explicitly asks in the current turn. This includes `install.sh`, `uv sync`, `pip install`, `conda env create`, `conda create`, and equivalent commands.
- Do not download model weights, checkpoints, text encoders, or other large artifacts without explicit user approval in the current turn.
- Keep the policy environment separate from the XPolicyLab project/base environment and from the evaluation/debug client environment.
- Run policy-specific data conversion, training, inference, and model-server code in the policy environment.
- Never silently use the XPolicyLab environment as a fallback for policy preprocessing or inference dependencies.
- Keep generated docs and scripts portable: no absolute paths in examples, defaults, shell scripts, or installation docs.
- Standardize every policy image at training and deployment boundaries to RGB, width `320`, height `240`, HWC shape `(240, 320, 3)` before NCHW conversion, normalization, or upstream preprocessing.
- Report progress on multi-item conversion/encoding loops with a single `tqdm` bar per phase plus `set_postfix(...)` for live counters; do not emit per-episode or per-frame `print()` lines. Use `tqdm.write(...)` for end-of-phase summaries. See `references/import-workflow.md` for the pattern.
- Preserve real task language for language-conditioned policies. Placeholder embeddings may be used only for clearly named smoke tests, and real training must fail loudly until real embeddings are generated from the upstream text encoder.
- Treat `XPolicyLab/debug_env_client.py` through `XPolicyLab/utils/run_debug_env_client.sh` as the first end-to-end validation gate.

## Default Workflow

1. Identify policy name and upstream source. Preserve user-specified casing.
2. Inspect the local XPolicyLab interface and the DP/demo references.
3. Scaffold the policy workspace with the bundled script.
4. Remove the nested upstream `.git` directory **and every `.gitignore` file inside the upstream tree** (`find <upstream> -name .gitignore -delete`, recursive — upstream often vendors sub-projects that carry their own) so the source becomes a plain snapshot with no foreign rule files. Keeping any upstream `.gitignore` is a latent footgun: it may name upstream-tracked-but-ignored files (added before the rule, or via `git add -f`); on the first clone everything is fine, but any later transfer that respects `.gitignore` (rsync `--filter=':- .gitignore'`, VSCode/PyCharm remote sync with "exclude items by .gitignore", `tar --exclude-from`, etc.) silently drops those files on the destination host. Removing them is the only fix that survives re-clones across machines. Audit the contents first only so you can decide if any upstream-tracked-but-ignored file needs to be patched in the adapter (e.g. paths that hard-code the upstream maintainer's home dir).
5. Create `XPolicyLab/policy/<PolicyName>/.gitignore` mirroring `XPolicyLab/policy/DP/.gitignore` (`data/*`, `checkpoints/*`) plus any policy-specific large/generated outputs such as `runs/*`, `weights/*`, training logs, or cache directories. Do this before the commit gate so converted datasets, downloaded checkpoints, and run artifacts never enter the repository. The parent `XPolicyLab/.gitignore` already excludes `__pycache__/` and `*.egg-info`, so do not duplicate those.
6. Commit the freshly created policy directory in the XPolicyLab git repository before adaptation edits; if this is blocked, report the blocker before continuing.
7. Audit upstream docs and code for dependencies, model construction, checkpoint loading, observation keys, image preprocessing, state/action formats, reset/stateful rollout behavior, and language/text encoder requirements.
8. Add or adapt only the policy-local files needed for XPolicyLab: `model.py`, `deploy.py`, `deploy.yml`, `process_data.sh`, `train.sh`, `eval.sh`, `install.sh`, `INSTALLATION.md`, and upstream-local helper modules.
9. Keep helper scripts inside `XPolicyLab/policy/<PolicyName>/<upstream_project_dir>/` whenever practical; keep `XPolicyLab/policy/` tidy.
10. Do not create policy-root provenance files such as `IMPORT_NOTES.md` unless the user explicitly asks for an in-tree import record; keep source metadata in git commit messages, PR descriptions, or external project notes.
11. Validate in layers: syntax/import checks, config consistency, data conversion smoke checks when dependencies are available, then the debug client path.
12. If dependencies, checkpoints, embeddings, or artifacts are missing, report exactly what is blocked and what the user must install or download.

## Scaffold Command

Use the bundled script from the repository root:

```bash
python skills/xpolicylab-import-policy/scripts/scaffold_policy_import.py \
  --repo <git-url-or-local-path> \
  --name <PolicyName> \
  --project-root .
```

The script calls `bash create_policy.sh <PolicyName>` from `XPolicyLab/`, then clones or copies the upstream project into `XPolicyLab/policy/<PolicyName>/<upstream_project_dir>/`.

## Required Policy Contract

Every completed deployment integration must provide:

- `XPolicyLab/policy/<PolicyName>/__init__.py`: lightweight package marker.
- `XPolicyLab/policy/<PolicyName>/model.py`: defines `class Model(ModelTemplate)`.
- `XPolicyLab/policy/<PolicyName>/deploy.py`: deployment loop helpers, close to `demo_policy/deploy.py` unless policy-specific temporal/chunk logic requires changes.
- `XPolicyLab/policy/<PolicyName>/deploy.yml`: explicit config with `policy_name` equal to the folder/import name and `null` for values overridden by `eval.sh`.
- `XPolicyLab/policy/<PolicyName>/install.sh`: concrete policy-environment setup commands; unresolved dependency choices must be reported as blockers, not left as vague TODOs.
- `XPolicyLab/policy/<PolicyName>/INSTALLATION.md`: short, command-oriented installation/download instructions.
- `XPolicyLab/policy/<PolicyName>/process_data.sh`: DP-style data conversion entrypoint.
- `XPolicyLab/policy/<PolicyName>/train.sh`: upstream training entrypoint wrapper.
- `XPolicyLab/policy/<PolicyName>/eval.sh`: starts the policy model server in the policy environment and runs the current debug client path.

This means the full code path must be implemented even when local full training or real-model inference cannot be run. Do not skip `process_data.sh`, `train.sh`, `model.py`, `deploy.py`, or `eval.sh` because a later validation tier is unavailable.

## Tiered Validation

Run every validation tier that is locally feasible:

1. Always run static checks that do not require heavy policy dependencies: shell syntax checks, Python syntax compile for adapter files, `deploy.yml` consistency, `policy_name` import-path checks, and image-standard audits.
2. Run data-conversion smoke tests when the policy environment and required lightweight dependencies are available.
3. Run model import, server startup, and `eval.sh` single-environment debug-client tests when the policy environment, checkpoints, and runtime dependencies are available.
4. Run batch debug tests when batch support is implemented and single-environment debug has passed.
5. Run real training only when the user explicitly asks and the required compute, data, dependencies, and artifacts are available.

Do not stop at a half-finished scaffold. If dependencies, checkpoints, text encoders, embeddings, data, GPU, or environment issues block a validation tier, report the exact blocked command, missing requirement, last tier that passed, and next user action needed to unblock it.
