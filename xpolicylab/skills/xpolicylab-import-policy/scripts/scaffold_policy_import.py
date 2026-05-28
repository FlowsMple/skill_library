#!/usr/bin/env python3
"""Scaffold an external policy repository into XPolicyLab/policy/<name>.

This follows XPolicyLab/README.md: create the policy from demo_policy with
create_policy.sh, then place upstream source inside the generated folder.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


def run(cmd: list[str], cwd: Path | None = None) -> str:
    result = subprocess.run(cmd, cwd=cwd, check=True, text=True, capture_output=True)
    return result.stdout.strip()


def is_git_url(value: str) -> bool:
    return value.startswith(("http://", "https://", "git@")) or value.endswith(".git")


def derive_source_dir(repo: str) -> str:
    value = repo.rstrip("/")
    name = value.rsplit("/", 1)[-1]
    if name.endswith(".git"):
        name = name[:-4]
    return name or "upstream"


def write_if_missing(path: Path, text: str) -> None:
    if not path.exists():
        path.write_text(text, encoding="utf-8")


def remove_nested_git(source_target: Path) -> None:
    nested_git = source_target / ".git"
    if nested_git.exists():
        shutil.rmtree(nested_git)
    # Also strip every nested .gitignore inside the upstream tree (top-level
    # and any sub-project's). These only governed upstream's contributor
    # workflow; in our snapshot they become active filters for any sync tool
    # that respects nested .gitignore (rsync --filter=':- .gitignore',
    # VSCode/PyCharm remote sync, tar --exclude-from), silently dropping
    # upstream-tracked-but-ignored files on destination hosts. Recursive
    # because upstream sometimes vendors its own subprojects with their own
    # .gitignore. See xpolicylab-import-policy SKILL.md.
    for nested_gitignore in source_target.rglob(".gitignore"):
        nested_gitignore.unlink()


def commit_original_snapshot(xpolicylab_root: Path, policy_name: str) -> str:
    if not (xpolicylab_root / ".git").exists():
        return "Skipped commit: XPolicyLab is not a git repository."

    rel_policy = f"policy/{policy_name}"
    run(["git", "add", "--", rel_policy], cwd=xpolicylab_root)
    try:
        run(
            ["git", "commit", "-m", f"Import {policy_name} original policy source", "--", rel_policy],
            cwd=xpolicylab_root,
        )
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "").strip()
        raise SystemExit(
            "Failed to commit the original policy snapshot. "
            "Configure git user info or inspect repository state, then commit "
            f"{rel_policy} manually.\n{detail}"
        ) from exc

    commit_hash = run(["git", "rev-parse", "--short", "HEAD"], cwd=xpolicylab_root)
    return f"Committed original snapshot: {commit_hash}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True, help="Git URL or local source path")
    parser.add_argument("--name", required=True, help="Policy folder/import name")
    parser.add_argument("--project-root", default=".", help="Project root containing XPolicyLab/")
    parser.add_argument("--source-dir", default=None, help="Subdirectory for upstream source; defaults to repo basename")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing policy folder")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    xpolicylab_root = project_root / "XPolicyLab"
    policy_root = project_root / "XPolicyLab" / "policy"
    if not policy_root.is_dir():
        raise SystemExit(f"Cannot find policy root: {policy_root}")
    create_script = xpolicylab_root / "create_policy.sh"
    if not create_script.is_file():
        raise SystemExit(f"Cannot find create_policy.sh: {create_script}")

    target = policy_root / args.name
    if target.exists() and not args.force:
        raise SystemExit(f"Policy folder exists: {target}. Use --force to overwrite.")
    if target.exists() and args.force:
        shutil.rmtree(target)
    run(["bash", "create_policy.sh", args.name], cwd=xpolicylab_root)

    source_dir = args.source_dir or derive_source_dir(args.repo)
    source_target = target / source_dir
    if is_git_url(args.repo):
        run(["git", "clone", args.repo, str(source_target)])
        remove_nested_git(source_target)
    else:
        source = Path(args.repo).expanduser().resolve()
        if not source.exists():
            raise SystemExit(f"Source path does not exist: {source}")
        shutil.copytree(source, source_target, ignore=shutil.ignore_patterns(".git", "__pycache__"))

    write_if_missing(
        target / "INSTALLATION.md",
        f'''# {args.name} Installation

TODO:
- Add upstream dependency installation commands for this policy environment.
- Add checkpoint/model/text-encoder download commands or expected local filenames.
''',
    )

    commit_status = commit_original_snapshot(xpolicylab_root, args.name)

    # demo_policy already provides most project-native files. Only replace the
    # minimal import shim and add TODO-heavy adapter content if the template did
    # not create those files for some reason.
    (target / "__init__.py").write_text('"""XPolicyLab policy package."""\n', encoding="utf-8")

    # create_policy.sh copies demo_policy/eval.sh, setup_eval_policy_server.sh,
    # and setup_eval_env_client.sh verbatim. Those templates derive policy_name
    # from `basename "${SCRIPT_DIR}"`, so they already work for <PolicyName>
    # without per-policy substitution. Only fall back to writing them here if
    # the copy did not happen for some reason.
    write_if_missing(
        target / "eval.sh",
        '''#!/bin/bash
set -e

dataset_name=$1
task_name=$2
ckpt_name=$3
env_cfg_type=$4
expert_data_num=$5
action_type=$6
seed=$7
policy_gpu_id=$8
env_gpu_id=$9
policy_conda_env=${10}
eval_env_conda_env=${11}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
UTILS_DIR="${ROOT_DIR}/XPolicyLab/utils"

SERVER_SCRIPT="${SCRIPT_DIR}/setup_eval_policy_server.sh"
CLIENT_SCRIPT="${SCRIPT_DIR}/setup_eval_env_client.sh"

policy_server_port=$(bash "${UTILS_DIR}/get_free_port.sh")
policy_server_ip="localhost"

additional_info="ckpt_name=${ckpt_name},action_type=${action_type}"

cleanup() {
    if [[ -n "${SERVER_PID:-}" ]]; then
        echo "[MAIN] kill server ${SERVER_PID}"
        kill "${SERVER_PID}" 2>/dev/null || true
    fi
}
trap cleanup EXIT

echo "[MAIN] start server, policy_server_port=${policy_server_port}"

bash "${SERVER_SCRIPT}" \\
    "${dataset_name}" \\
    "${task_name}" \\
    "${ckpt_name}" \\
    "${env_cfg_type}" \\
    "${expert_data_num}" \\
    "${action_type}" \\
    "${seed}" \\
    "${policy_gpu_id}" \\
    "${policy_conda_env}" \\
    "${policy_server_port}" &

SERVER_PID=$!

sleep 3

echo "[MAIN] start client, server=${policy_server_ip}:${policy_server_port}"

bash "${CLIENT_SCRIPT}" \\
    "${dataset_name}" \\
    "${task_name}" \\
    "${ckpt_name}" \\
    "${env_cfg_type}" \\
    "${action_type}" \\
    "${seed}" \\
    "${env_gpu_id}" \\
    "${eval_env_conda_env}" \\
    "${additional_info}" \\
    "${policy_server_port}" \\
    "${policy_server_ip}"

echo "[MAIN] eval finished"
''',
    )

    write_if_missing(
        target / "setup_eval_policy_server.sh",
        '''#!/bin/bash
set -e

dataset_name=$1
task_name=$2
ckpt_name=$3
env_cfg_type=$4
expert_data_num=$5
action_type=$6
seed=$7
policy_gpu_id=$8
policy_conda_env=$9
policy_server_port=${10}
policy_server_host=${11:-"localhost"}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
UTILS_DIR="${ROOT_DIR}/XPolicyLab/utils"

policy_name="$(basename "${SCRIPT_DIR}")"
yaml_file="${ROOT_DIR}/XPolicyLab/policy/${policy_name}/deploy.yml"

action_dim=$(bash "${UTILS_DIR}/get_action_dim.sh" "${ROOT_DIR}" "${env_cfg_type}")

echo "[SERVER] policy=${policy_name}, task=${task_name}, ckpt=${ckpt_name}, policy_server_port=${policy_server_port}, action_dim=${action_dim}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${policy_conda_env}"

# TODO: append policy-specific overrides here (checkpoint_path, model_path, etc.).
exec env \\
    PYTHONWARNINGS=ignore::UserWarning \\
    CUDA_VISIBLE_DEVICES="${policy_gpu_id}" \\
    python "${ROOT_DIR}/XPolicyLab/setup_policy_server.py" \\
        --config_path "${yaml_file}" \\
        --overrides \\
            policy_server_port="${policy_server_port}" \\
            policy_server_host="${policy_server_host}" \\
            dataset_name="${dataset_name}" \\
            task_name="${task_name}" \\
            ckpt_name="${ckpt_name}" \\
            env_cfg_type="${env_cfg_type}" \\
            seed="${seed}" \\
            policy_name="${policy_name}" \\
            action_type="${action_type}" \\
            action_dim="${action_dim}"
''',
    )

    write_if_missing(
        target / "setup_eval_env_client.sh",
        '''#!/bin/bash
set -e

dataset_name=$1
task_name=$2
ckpt_name=$3
env_cfg_type=$4
action_type=$5
seed=$6
env_gpu_id=$7
eval_env_conda_env=$8
additional_info=$9
policy_server_port=${10}
policy_server_ip=${11:-"localhost"}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
UTILS_DIR="${ROOT_DIR}/XPolicyLab/utils"

policy_name="$(basename "${SCRIPT_DIR}")"
yaml_file="${ROOT_DIR}/XPolicyLab/policy/${policy_name}/deploy.yml"

echo "[CLIENT] policy=${policy_name}, task=${task_name}, server=${policy_server_ip}:${policy_server_port}"

bash "${UTILS_DIR}/setup_env_client.sh" \\
    "${UTILS_DIR}" \\
    "${yaml_file}" \\
    "${eval_env_conda_env}" \\
    "${policy_server_port}" \\
    "${dataset_name}" \\
    "${task_name}" \\
    "${env_cfg_type}" \\
    "${policy_name}" \\
    "${additional_info}" \\
    "${ROOT_DIR}" \\
    "${seed}" \\
    "${env_gpu_id}" \\
    "${policy_server_ip}"
''',
    )

    write_if_missing(
        target / "model.py",
        f'''import os

from XPolicyLab.model_template import ModelTemplate
from XPolicyLab.utils.process_data import (
    get_robot_action_dim_info,
    pack_robot_state,
    unpack_robot_state,
)


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
UPSTREAM_DIR = os.path.join(CURRENT_DIR, "{source_dir}")


class Model(ModelTemplate):
    def __init__(self, model_cfg):
        self.model_cfg = model_cfg
        self.action_type = model_cfg["action_type"]
        self.env_cfg_type = model_cfg["env_cfg_type"]
        self.robot_action_dim_info = get_robot_action_dim_info(self.env_cfg_type)
        self.model = self.get_model(model_cfg)
        self._last_obs = None

    def get_model(self, model_cfg):
        raise NotImplementedError("TODO: build/load the upstream {args.name} model here.")

    def update_obs(self, obs):
        self._last_obs = self.encode_obs(obs)

    def update_obs_batch(self, obs_list):
        self._last_obs = [self.encode_obs(obs) for obs in obs_list]

    def encode_obs(self, observation):
        state = pack_robot_state(
            observation,
            self.action_type,
            self.robot_action_dim_info,
            source_type="obs",
        )
        return {{
            "raw": observation,
            "state": state,
        }}

    def get_action(self):
        raise NotImplementedError("TODO: run upstream inference and return a list of action dicts.")

    def get_action_batch(self, env_idx_list):
        raise NotImplementedError("TODO: implement batch inference or explain why unsupported.")

    def reset(self):
        pass
''',
    )
    write_if_missing(
        target / "deploy.py",
        '''def eval_one_episode(TASK_ENV, model_client):
    while not TASK_ENV.is_episode_end():
        obs = TASK_ENV.get_obs()
        model_client.call(func_name="update_obs", obs=obs)
        actions = model_client.call(func_name="get_action")
        for action_idx, action in enumerate(actions):
            TASK_ENV.take_action(action)
            if action_idx != len(actions) - 1:
                obs = TASK_ENV.get_obs()
                model_client.call(func_name="update_obs", obs=obs)


def eval_one_episode_batch(TASK_ENV, model_client):
    while not TASK_ENV.is_episode_end():
        env_idx_list = TASK_ENV.get_running_env_idx_list()
        obs_list = TASK_ENV.get_obs_batch(env_idx_list)
        model_client.call(func_name="update_obs_batch", obs=obs_list)
        actions = model_client.call(func_name="get_action_batch", obs=env_idx_list)
        for action_idx in range(len(actions[0])):
            current_action_list = [env_actions[action_idx] for env_actions in actions]
            TASK_ENV.take_action_batch(current_action_list, env_idx_list)
            if action_idx != len(actions[0]) - 1:
                env_idx_list = TASK_ENV.get_running_env_idx_list()
                obs_list = TASK_ENV.get_obs_batch(env_idx_list)
                model_client.call(func_name="update_obs_batch", obs=obs_list)
''',
    )
    (target / "deploy.yml").write_text(
        f'''policy_name: {args.name}
dataset_name: null
task_name: null
ckpt_name: null
env_cfg_type: null
expert_data_num: null
action_type: null
seed: null
eval_batch: false        # flip to true only after update_obs_batch / get_action_batch are validated
eval_env: debug          # debug | sim | real -- selects the env client runner via setup_env_client.sh
checkpoint_path: null
model_path: null
upstream_dir: {source_dir}
sample_data_dir: data/RoboDojo/test_data/arx_x5
''',
        encoding="utf-8",
    )
    (target / "process_data.sh").write_text(
        '''#!/bin/bash
set -euo pipefail

dataset_name=${1}
task_name=${2}
env_cfg_type=${3}
expert_data_num=${4}
action_type=${5}

echo "TODO: convert data/${dataset_name}/${task_name}/${env_cfg_type} or data/RoboDojo/test_data/arx_x5 into upstream format"
echo "TODO: standardize every camera frame to RGB HWC shape (240, 320, 3), i.e. 320x240 width x height"
''',
        encoding="utf-8",
    )
    write_if_missing(
        target / "train.sh",
        '''#!/bin/bash
set -euo pipefail

dataset_name=${1}
task_name=${2}
env_cfg_type=${3}
expert_data_num=${4}
action_type=${5}
seed=${6}
gpu_id=${7}

echo "TODO: call upstream training entrypoint"
''',
    )
    write_if_missing(
        target / "install.sh",
        f'''#!/bin/bash
set -euo pipefail

policy_conda_env="${{1:-XPolicyLab-{args.name}}}"

echo "TODO: create/use an independent Conda or uv environment for this policy."
echo "TODO: for Conda, install upstream dependencies inside: ${{policy_conda_env}}"
echo "TODO: for uv, follow policy/Pi_05 style and activate <policy_uv_env_path>/.venv/bin/activate in eval.sh"
echo "Docker is not allowed for XPolicyLab policy integrations."
''',
    )
    print(f"Created policy scaffold: {target}")
    print(f"Upstream source: {source_target}")
    print(commit_status)


if __name__ == "__main__":
    main()
