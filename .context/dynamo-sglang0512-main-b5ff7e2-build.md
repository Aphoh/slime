# Dynamo SGLang 0.5.12 Runtime Image Provenance

This records the engine/frontend image lineage for:

```text
aphoh/slime:swepro-sglang0512-main-b5ff7e2-20260606T1550Z
```

It is not a Slime trainer image. It is a Dynamo-rendered SGLang runtime image.

## Source

Transcript evidence from thread `019e8c6e-d7d9-7890-8bcd-6f80c91e6c91` says the image was built from local Dynamo main at:

```text
b5ff7e26ba7e
```

The successful build path was:

```text
cd /Users/warnold/dev/dynamo
container/render.py --framework=sglang --target=runtime --platform=linux/arm64 --output-short-filename
docker build --platform linux/arm64 -f container/rendered.Dockerfile -t aphoh/slime:swepro-sglang0512-main-b5ff7e2-20260606T1550Z .
docker push aphoh/slime:swepro-sglang0512-main-b5ff7e2-20260606T1550Z
```

The copied local reproduction Dockerfile is:

```text
/Users/warnold/dev/slime/.context/Dockerfile.dynamo-sglang0512-main-b5ff7e2
```

Build it with the Dynamo repo as the context:

```text
docker build --platform linux/arm64 \
  -f /Users/warnold/dev/slime/.context/Dockerfile.dynamo-sglang0512-main-b5ff7e2 \
  -t aphoh/slime:swepro-sglang0512-main-b5ff7e2-repro \
  /Users/warnold/dev/dynamo
```

## Base Images

The rendered Dockerfile uses:

```text
ARG BASE_IMAGE=nvcr.io/nvidia/cuda-dl-base
ARG BASE_IMAGE_TAG=25.11-cuda13.0-devel-ubuntu24.04
ARG RUNTIME_IMAGE=lmsysorg/sglang
ARG RUNTIME_IMAGE_TAG=v0.5.12.post1-cu130-runtime
```

The final runtime stage starts from:

```text
lmsysorg/sglang:v0.5.12.post1-cu130-runtime
```

## Important Difference From Trainer Images

This image has Dynamo + SGLang runtime content and is appropriate for frontend/engine pods.
It does not include the Slime trainer stack:

```text
/root/src/Megatron-LM
/code/slime
/code/SWE-bench_Pro-os
```

The trainer-capable images were built from `aphoh/slime:0.2.4-arm64-gb200` with:

```text
/Users/warnold/dev/slime/examples/swebench-pro/Dockerfile.arm64-gb200
```

Those runs also depended on the `code-cache` PVC mounted at `/code` for SWE-Pro/SWE-agent files.
