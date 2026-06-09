# Reproducible SWE-Pro/Dynamo Experiments

This directory is the simple interface for this branch:

- `cluster.yaml` describes Ray, GPU allocation, Dynamo endpoints, and small runtime-environment details.
- `experiment_config.yaml` describes the experiment as ordered groups of normal slime CLI arguments.

The runner only translates those two files into a stock command shaped like:

```bash
ray job submit ... -- python3 train_async.py <normal slime args>
```

## Validate The Command

From the repository root:

```bash
python3 examples/swebench-pro/run_experiment.py \
  --cluster examples/swebench-pro/reproducible/cluster.yaml \
  --experiment examples/swebench-pro/reproducible/experiment_config.yaml \
  --dry-run
```

For deterministic trace-replay performance tests:

```bash
python3 examples/swebench-pro/run_experiment.py \
  --cluster examples/swebench-pro/reproducible/cluster.yaml \
  --experiment examples/swebench-pro/reproducible/experiment_config.yaml \
  --mode perf-test \
  --dry-run
```

The dry-run output prints the resolved `train_async.py` command, the final Ray
submission command, and the runtime env JSON that Ray will receive.

## Run

Once the cluster services in `cluster.yaml` are reachable and Ray is listening:

```bash
python3 examples/swebench-pro/run_experiment.py \
  --cluster examples/swebench-pro/reproducible/cluster.yaml \
  --experiment examples/swebench-pro/reproducible/experiment_config.yaml
```

Run the performance-test mode with the same command plus `--mode perf-test`.

Each submitted run snapshots both YAML files plus the generated commands under
`.context/swepro-runs/<run-id>/`.

## Editing Rules

Keep cluster-owned launch details in `cluster.yaml`:

- actor and rollout GPU counts
- Ray address and job behavior
- Dynamo frontend, NATS, and worker-system port
- runtime environment required by infrastructure

The Dynamo frontend is externally managed. Configure its router mode and KV
event behavior in the frontend deployment itself; the experiment runner only
attaches slime to the declared frontend URL.

Keep algorithm and trainer behavior in `experiment_config.yaml` as literal slime
CLI tokens. That includes GRPO/PPO/GSPO choices, losses, delayed weight-update
intervals, rollout function paths, optimizer settings, and performance knobs.

The runner rejects duplicate cluster-owned flags in experiment argument groups,
so `--actor-num-nodes` and `--dynamo-frontend-url` do not drift between files.
