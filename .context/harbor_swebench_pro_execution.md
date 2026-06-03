# Harbor + SWE-bench Pro Execution Notes

Date: 2026-04-30

This note summarizes how Harbor runs SWE-bench Pro tasks, what state/resume behavior it provides, and how it could help slime generalize across multiple software-engineering harnesses.

## Summary

Harbor turns each SWE-bench Pro instance into a normal Harbor task directory. The task contains a Docker environment, an instruction, an oracle solution script, and a verifier wrapper around the official SWE-bench Pro `run_script.sh` and `parser.py`.

For slime, the useful piece is not `harbor run` as a training loop replacement. The useful piece is Harbor as a task packaging, sandbox, harness, verifier, artifact, and job-resumption layer. Slime still needs to own trainable rollout generation because it needs token traces, loss masks, and logprobs.

## Harbor Task Shape

The SWE-bench Pro adapter generates one Harbor task per instance:

```text
task/
  task.toml
  instruction.md
  environment/Dockerfile
  solution/solve.sh
  tests/test.sh
  tests/run_script.sh
  tests/parser.py
  tests/config.json
```

Relevant Harbor references:

- Task layout and execution flow: `/Users/warnold/proj/harbor/adapters/swebenchpro/README.md:54`
- Dataset loading from `ScaleAI/SWE-bench_Pro`: `/Users/warnold/proj/harbor/adapters/swebenchpro/src/swebenchpro/adapter.py:62`
- Generated task writing: `/Users/warnold/proj/harbor/adapters/swebenchpro/src/swebenchpro/adapter.py:286`
- Generated Dockerfile template: `/Users/warnold/proj/harbor/adapters/swebenchpro/src/swebenchpro/task-template/Dockerfile:1`
- Generated verifier wrapper: `/Users/warnold/proj/harbor/adapters/swebenchpro/src/swebenchpro/task-template/test.sh:1`
- Oracle patch script template: `/Users/warnold/proj/harbor/adapters/swebenchpro/src/swebenchpro/task-template/solve.sh:1`

The adapter writes `tests/config.json` with the full SWE-bench Pro record. It copies official per-instance `run_script.sh` and `parser.py`, then applies a small Jest reliability patch by adding `--maxWorkers=1 --forceExit`.

## Execution Flow

`harbor run -d cais/swebenchpro` resolves a dataset package, creates trial configs, and runs each trial through Harbor's generic trial runner.

The trial flow is:

1. Load the task.
2. Start the configured environment.
3. Install/setup the selected agent harness.
4. Run the agent against `instruction.md`.
5. Upload `tests/` into the environment.
6. Execute the verifier script.
7. Read `/logs/verifier/reward.txt` or `/logs/verifier/reward.json`.
8. Save trial artifacts and `result.json`.

Relevant Harbor references:

- CLI config resolution for `harbor run`: `/Users/warnold/proj/harbor/src/harbor/cli/jobs.py:984`
- Dataset/package task resolution: `/Users/warnold/proj/harbor/src/harbor/models/job/config.py:124`
- Job creates trial configs: `/Users/warnold/proj/harbor/src/harbor/job.py:248`
- Trial lifecycle: `/Users/warnold/proj/harbor/src/harbor/trial/trial.py:126`
- Environment start: `/Users/warnold/proj/harbor/src/harbor/trial/trial.py:305`
- Agent execution: `/Users/warnold/proj/harbor/src/harbor/trial/trial.py:362`
- Verification: `/Users/warnold/proj/harbor/src/harbor/trial/trial.py:385`
- Final result writing: `/Users/warnold/proj/harbor/src/harbor/trial/trial.py:421`
- Generic verifier implementation: `/Users/warnold/proj/harbor/src/harbor/verifier/verifier.py:123`

For SWE-bench Pro specifically, `tests/test.sh` does the benchmark work:

1. `cd /app` or `/testbed`.
2. Run the final line of `before_repo_set_cmd` to check out the gold test files.
3. Select the configured test files.
4. Run official `run_script.sh`.
5. Run official `parser.py` into `/tmp/output.json`.
6. Compare passed tests against `fail_to_pass | pass_to_pass`.
7. Write scalar Harbor reward to `/logs/verifier/reward.txt`.

Reference: `/Users/warnold/proj/harbor/adapters/swebenchpro/src/swebenchpro/task-template/test.sh:26`

## Containers and Environments

The default SWE-bench Pro config uses Docker:

- `/Users/warnold/proj/harbor/adapters/swebenchpro/swebenchpro.yaml:8`

The Docker environment is built from each task's `environment/Dockerfile`. For SWE-bench Pro, that Dockerfile extends a prebuilt official SWE-bench Pro image and resets the repo to the base commit.

Harbor's Docker backend uses Docker Compose. It bind-mounts host trial directories into the container:

- `/logs/agent`
- `/logs/verifier`
- `/logs/artifacts`

Relevant Harbor references:

- Environment backend registry: `/Users/warnold/proj/harbor/src/harbor/environments/factory.py:24`
- Docker environment construction: `/Users/warnold/proj/harbor/src/harbor/environments/docker/docker.py:145`
- Compose files used for Docker tasks: `/Users/warnold/proj/harbor/src/harbor/environments/docker/docker.py:253`
- Docker start/build/up flow: `/Users/warnold/proj/harbor/src/harbor/environments/docker/docker.py:462`
- Docker cleanup behavior: `/Users/warnold/proj/harbor/src/harbor/environments/docker/docker.py:520`
- Log/artifact bind mounts: `/Users/warnold/proj/harbor/src/harbor/environments/docker/docker-compose-base.yaml:1`

Harbor is not limited to Docker. The environment factory includes Docker, Daytona, E2B, GKE, Modal, Runloop, Singularity, Tensorlake, and Apple Container backends. That is useful if we want the same task packaging and verifier logic but different sandbox providers.

## State Management

During a trial, task state is the mutable filesystem inside the live environment. The selected agent edits the checked-out repository in that environment. The verifier then runs in the same environment against those edits.

Persistent state after a trial is mostly filesystem output under the trial directory:

- `agent/`
- `verifier/`
- `artifacts/`
- `config.json`
- `result.json`
- `trial.log`

Relevant Harbor references:

- Trial output structure: `/Users/warnold/proj/harbor/src/harbor/models/trial/paths.py:76`
- Environment-side paths: `/Users/warnold/proj/harbor/src/harbor/models/trial/paths.py:9`
- Logs downloaded or chowned after agent execution: `/Users/warnold/proj/harbor/src/harbor/trial/trial.py:444`
- Artifact collection: `/Users/warnold/proj/harbor/src/harbor/trial/trial.py:760`

For default Docker runs with `delete: true`, the container and volumes are cleaned up after the trial. The final edited repo state is not preserved unless captured as an artifact or left in logs by the agent/harness.

## Resumption Behavior

Harbor supports job-level resumption, not mid-trial checkpointing.

`harbor job resume -p <job_dir>` reloads the saved job config, scans completed trial directories, keeps trials that already have `result.json`, removes incomplete trial dirs, and schedules the remaining trial configs.

Relevant Harbor references:

- Resume CLI: `/Users/warnold/proj/harbor/src/harbor/cli/jobs.py:1271`
- Existing job detection: `/Users/warnold/proj/harbor/src/harbor/job.py:68`
- Existing trial scan: `/Users/warnold/proj/harbor/src/harbor/job.py:168`
- Remaining trial filtering: `/Users/warnold/proj/harbor/src/harbor/job.py:226`
- Result update on trial completion: `/Users/warnold/proj/harbor/src/harbor/job.py:405`

This helps large evaluation batches. If a 731-task SWE-bench Pro run is interrupted, completed trials do not need to run again. It does not resume an agent halfway through an individual task. A failed or cancelled trial restarts from a fresh environment.

## Harness Generalization

Harbor's main value for our use case is that it can run the same Harbor task through many agent harnesses while preserving the same verifier and result format.

Built-in agent adapters include:

- `oracle`
- `nop`
- `terminus-2`
- `claude-code`
- `codex`
- `opencode`
- `swe-agent`
- `mini-swe-agent`
- `openhands`
- `openhands-sdk`
- `aider`
- `gemini-cli`
- `goose`
- `qwen-coder`
- others

Relevant Harbor references:

- Agent factory registry: `/Users/warnold/proj/harbor/src/harbor/agents/factory.py:33`
- Agent enum names: `/Users/warnold/proj/harbor/src/harbor/models/agent/name.py:4`
- Installed agent base class: `/Users/warnold/proj/harbor/src/harbor/agents/installed/base.py:139`
- SWE-agent adapter command construction: `/Users/warnold/proj/harbor/src/harbor/agents/installed/swe_agent.py:388`
- Mini-SWE-agent trajectory conversion: `/Users/warnold/proj/harbor/src/harbor/agents/installed/mini_swe_agent.py:133`

This means Harbor can serve as a shared evaluation substrate for comparing harnesses on the same tasks:

```bash
harbor run -d cais/swebenchpro -a codex -m openai/...
harbor run -d cais/swebenchpro -a swe-agent -m openai/...
harbor run -d cais/swebenchpro -a opencode -m openai/...
harbor run -d cais/swebenchpro -a claude-code -m anthropic/...
```

That is valuable even if slime does not use Harbor directly for RL rollouts.

## Relationship to Current Slime SWE-bench Pro Setup

Current slime already has a custom SWE-bench Pro pipeline:

- Dataset conversion: `examples/swebench-pro/prepare_swebench_pro_data.py:89`
- Custom generation/reward hooks: `examples/swebench-pro/generate_with_swebench_pro.py:520`
- Patch metadata and NATS eval request: `examples/swebench-pro/generate_with_swebench_pro.py:1000`
- Session worker: `docker/swepro-session/runner.py:276`
- Eval worker: `docker/swepro-eval/runner.py:82`
- Custom per-sample rollout hook path: `slime/rollout/sglang_rollout.py:392`

The current slime path is RL-native: it preserves model completions, tokens, loss masks, and logprobs. Harbor's normal `harbor run` path is harness-native: it runs an installed agent and records trial-level results/artifacts.

That distinction matters. A Harbor agent run is excellent for evaluation and harness comparison, but it is not directly a slime training sample unless the harness is modified to call the slime model endpoint and return trainable token/logprob traces.

## What Harbor Could Be Used For

### 1. Canonical Task Source

Use Harbor package/dataset resolution as the canonical source of SWE-bench Pro task metadata, task assets, Dockerfiles, official `run_script.sh`, `parser.py`, and verification config.

Benefit: less bespoke SWE-bench Pro normalization in slime, easier upgrades when Harbor's `cais/swebenchpro` package changes.

### 2. Harness Comparison Runner

Use `harbor run` outside the RL loop to compare Codex, SWE-agent, OpenCode, Claude Code, mini-SWE-agent, OpenHands, and any custom agent against the same SWE-bench Pro task package.

Benefit: a common eval matrix for agent harness decisions before wiring a harness into slime.

### 3. Harbor-Backed Eval Worker

Replace or supplement `docker/swepro-eval/runner.py` with a Harbor verifier path:

1. Given `instance_id` and patch/diff, locate/download the Harbor task.
2. Start Harbor environment.
3. Apply the patch to the repo.
4. Run Harbor verifier.
5. Return reward and verifier artifacts.

Benefit: reuse Harbor's official task wrapper and result handling while keeping slime's RL-native generation.

### 4. Harbor-Backed Session Worker

Build a slime rollout worker that uses Harbor environments instead of the current custom SWE-agent session Docker setup. The worker would start a Harbor task environment, expose command execution to the model, and call Harbor verification at the end.

Benefit: generalizes beyond SWE-agent and beyond SWE-bench Pro if implemented against Harbor's task/environment interfaces.

Cost: more work than a verifier-only integration because the slime rollout still needs to collect token-level training data.

### 5. Artifact and Batch Resume Layer

Use Harbor's job directory conventions and resume semantics for large eval sweeps, especially non-training harness comparisons.

Benefit: interrupted multi-harness runs can skip completed trials. This is batch-level resume only; it does not checkpoint individual agent sessions.

## What Harbor Should Not Replace Initially

Do not replace slime's rollout/training loop with `harbor run` directly.

Reason: slime needs trainable samples with token IDs, response lengths, loss masks, and rollout logprobs. Generic Harbor harnesses usually return trial artifacts and reward, not a training-ready `Sample`.

The safer split is:

- Harbor: task packaging, sandbox, official verifier, harness comparison, batch eval resume.
- Slime: model serving, rollout sampling, logprob capture, training sample construction, RL optimization.

## Suggested Integration Path

### Phase 1: Read Harbor Tasks Into Slime

Add a small loader that resolves `cais/swebenchpro` or local Harbor task dirs and emits slime prompt records with:

- `instance_id`
- `instruction.md`
- `task_dir`
- Docker image/base commit metadata from `tests/config.json`
- `fail_to_pass`
- `pass_to_pass`
- `selected_test_files_to_run`
- path to official `run_script.sh`
- path to official `parser.py`

This is low risk and keeps the current slime rollout intact.

### Phase 2: Harbor Verifier Adapter

Add an optional reward/eval backend that evaluates a generated patch via Harbor's verifier contract instead of the current bespoke eval worker.

This would validate whether Harbor parity holds in our infrastructure while preserving all training behavior.

### Phase 3: Harness Eval Matrix

Run `harbor run` separately for selected harnesses and task subsets. Store results alongside slime eval outputs to compare harness behavior and identify which harnesses are worth making trainable.

### Phase 4: Trainable Harbor Environment Rollout

Only after verifier parity is solid, build a Harbor-backed rollout worker for slime. This would use Harbor environments for state and tools, but still use slime's model server and sampling path so it can return trainable `Sample` objects.

## Open Questions

- Does `cais/swebenchpro` expose all 731 tasks with the same files as the local adapter output?
- Are Harbor package task names stable enough to use as slime dataset IDs?
- Can Harbor environments be started cheaply enough for rollout-scale training, or should we keep the existing NATS session pool for RL and use Harbor mainly for eval?
- Which harnesses can expose token/logprob traces suitable for RL, versus only final trial artifacts?
- Should slime store Harbor trial directories as rollout artifacts for failed samples?

