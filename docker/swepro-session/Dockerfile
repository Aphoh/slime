FROM python:3.11

RUN apt-get update \
    && apt-get install -y --no-install-recommends git ca-certificates docker.io docker-cli \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir nats-py docker pyyaml

COPY runner.py /opt/swepro-session/runner.py

ENV SWE_AGENT_CONFIG_ROOT=/code/SWE-bench_Pro-os/SWE-agent
ENV SWEPRO_EVAL_ROOT=/code/SWE-bench_Pro-os
ENV DOCKER_HOST=unix:///var/run/docker-swepro.sock

ENTRYPOINT ["python", "/opt/swepro-session/runner.py"]
