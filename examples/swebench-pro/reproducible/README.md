# SWE-Pro/Dynamo Experiment Interface

This directory is the customer-facing Slynamo 3.0 interface:

- `cluster.yaml` describes Ray, GPU allocation, Dynamo endpoints, and small runtime-environment details.
- `experiment_config.yaml` describes the experiment as ordered groups of normal slime CLI arguments.

The runner only translates those two files into a stock command shaped like:

```bash
ray job submit ... -- python3 train_async.py ...
```

## Prerequisites

The included launch configuration expects:

- x86_64 trainer and Dynamo hosts
- a fixed-revision checkout of `SWE-bench_Pro-os`
- a Docker-compatible socket available to the session and evaluation workers
- a model checkpoint plus its Torch Distributed conversion
- prepared SWE-bench Pro prompt data
- an S3 bucket reachable from both Dynamo workers and Ray workers
- an externally managed Ray cluster with the resources declared in `cluster.yaml`

Check out the tested benchmark revision and its pinned submodules, then set
the remaining host-side values:

```bash
export SWEPRO_SOURCE_ROOT=/absolute/path/to/SWE-bench_Pro-os
git clone https://github.com/scaleapi/SWE-bench_Pro-os.git "$SWEPRO_SOURCE_ROOT"
git -C "$SWEPRO_SOURCE_ROOT" checkout ca10a60a5fcae51e6948ffe1485d4153d421e6c5
git -C "$SWEPRO_SOURCE_ROOT" submodule update --init --recursive

export SWEPRO_DOCKER_SOCKET=/var/run/docker.sock
export SWEPRO_DOCKERHUB_USERNAME=YOUR_DOCKERHUB_USERNAME
```

The session image pins the matching SWE-agent submodule revision
`402a7b8fdac8193f3f255bb53859ba274234f596`. The session and evaluation
workers execute benchmark-owned helpers and Dockerfiles from this checkout.

## Build

From the repository root, build the x86_64 trainer/Dynamo image and both
required SWE-Pro workers:

```bash
DOCKER_BUILDKIT=1 docker build -f docker/Dockerfile.dynamo -t slynamo:3.0-dynamo .
docker build -f docker/swepro-session/Dockerfile -t slynamo-swepro-session:3.0 .
docker build -f docker/swepro-eval/Dockerfile -t slynamo-swepro-eval:3.0 .
```

The images pin the Slime base digest, combined Dynamo revision, SWE-agent
revision, Rust toolchain, and Python dependency sets. The trainer image installs
the `fsspec`, `s3fs`, `zstandard`, and `msgspec` dependencies used by rollout
metadata.

Prepare prompt data with explicit inputs:

```bash
python3 examples/swebench-pro/prepare_swebench_pro_data.py \
  --input /absolute/path/to/source.jsonl \
  --output /absolute/path/to/swebench_pro_train.jsonl \
  --source-root "$SWEPRO_SOURCE_ROOT" \
  --dockerhub-username "$SWEPRO_DOCKERHUB_USERNAME"
```

Replace every `YOUR_*` value in `cluster.yaml` and `experiment_config.yaml`:

- `YOUR_DYNAMO_FRONTEND_HOST`
- `YOUR_NATS_HOST`
- `YOUR_S3_BUCKET`
- `YOUR_HF_CHECKPOINT`
- `YOUR_TORCH_DIST_CHECKPOINT`
- `YOUR_SWEBENCH_PRO_TRAIN_JSONL`

Dry runs preserve placeholders so the generated command can be inspected. A
real submission rejects unresolved placeholders.

## Required Services

SWE-Pro session rollouts require NATS, a session worker, and an evaluation
worker. The Dynamo frontend and SGLang worker additionally require etcd. The
following single-host recipe shows every required process; use equivalent
services in Kubernetes and inject the same environment variables. It publishes
the control ports so Ray workers outside the Docker network can reach them.
These services are unauthenticated; expose the ports only on a trusted network
and restrict ingress to the Ray and Dynamo nodes.

```bash
docker network create slynamo

docker run -d --name nats --network slynamo -p 4222:4222 \
  slynamo:3.0-dynamo nats-server -js

docker run -d --name etcd --network slynamo -p 2379:2379 \
  slynamo:3.0-dynamo etcd \
  --name etcd \
  --listen-client-urls http://0.0.0.0:2379 \
  --advertise-client-urls http://etcd:2379

docker run -d --name swepro-session --network slynamo \
  -e SWEPRO_NATS_URL=nats://nats:4222 \
  -e SWEPRO_DOCKERHUB_USERNAME="$SWEPRO_DOCKERHUB_USERNAME" \
  --mount type=bind,src="$SWEPRO_SOURCE_ROOT",dst=/code/SWE-bench_Pro-os,readonly \
  --mount type=bind,src="$SWEPRO_DOCKER_SOCKET",dst=/var/run/docker-swepro.sock \
  slynamo-swepro-session:3.0

docker volume create swepro-workspaces
docker run -d --name swepro-eval --network slynamo \
  -e SWEPRO_NATS_URL=nats://nats:4222 \
  -e SWEPRO_DOCKERHUB_USERNAME="$SWEPRO_DOCKERHUB_USERNAME" \
  --mount type=bind,src="$SWEPRO_SOURCE_ROOT",dst=/opt/SWE-bench_Pro-os,readonly \
  --mount type=bind,src="$SWEPRO_DOCKER_SOCKET",dst=/var/run/docker-swepro.sock \
  --mount type=volume,src=swepro-workspaces,dst=/swepro-workspaces \
  slynamo-swepro-eval:3.0
```

Start the Dynamo frontend and an SGLang worker from the pinned image. Both must
see `NATS_SERVER` and `ETCD_ENDPOINTS`. The worker must also receive the model
mount, S3 credentials or workload identity, `--enable-rl` for metadata uploads,
and `--enable-return-routed-experts` for routing replay:

```bash
docker run -d --name dynamo-frontend --network slynamo -p 3000:3000 -p 20390:20390 \
  -e NATS_SERVER=nats://nats:4222 \
  -e DYN_AGENT_TRACE_TOOL_EVENTS_ZMQ_ENDPOINT=tcp://0.0.0.0:20390 \
  -e ETCD_ENDPOINTS=http://etcd:2379 \
  slynamo:3.0-dynamo python -m dynamo.frontend --http-port 3000

docker run -d --name dynamo-worker --network slynamo --gpus all \
  -e NATS_SERVER=nats://nats:4222 \
  -e ETCD_ENDPOINTS=http://etcd:2379 \
  --env-file /absolute/path/to/s3-credentials.env \
  --mount type=bind,src=/absolute/path/to/model,dst=/models/model,readonly \
  slynamo:3.0-dynamo python -m dynamo.sglang \
  --model-path /models/model \
  --tp YOUR_TENSOR_PARALLEL_SIZE \
  --enable-rl \
  --enable-return-routed-experts
```

Set `YOUR_DYNAMO_FRONTEND_HOST` and `YOUR_NATS_HOST` in `cluster.yaml` to
DNS names or IP addresses reachable from every Ray node. Do not use the Docker-only
service names unless the Ray containers also join the `slynamo` network.

For AWS S3, the credentials file uses the standard `AWS_ACCESS_KEY_ID`,
`AWS_SECRET_ACCESS_KEY`, optional `AWS_SESSION_TOKEN`, and region variables.
For MinIO, also provide the endpoint expected by `s3fs`. Ray workers need the
same S3 access through workload identity or `cluster.yaml` runtime environment.

The example selects stateful `/v1/responses`; set `swepro.api_mode` to
`completions` to exercise `/v1/completions` instead. Core Slime rollouts
support partial-rollout cancellation and metadata recovery in either mode. The
SWE-Pro agent rollout is sessionful and intentionally rejects partial rollout.

## Validate The Command

From the repository root:

```bash
python3 examples/swebench-pro/run_experiment.py \
  --cluster examples/swebench-pro/reproducible/cluster.yaml \
  --experiment examples/swebench-pro/reproducible/experiment_config.yaml \
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

Each submitted run snapshots both YAML files plus the generated commands under
`.artifacts/swepro-runs/<run-id>/`.

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
