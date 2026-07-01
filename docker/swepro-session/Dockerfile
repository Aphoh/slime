FROM python:3.11@sha256:7fca2df201542db9232b551db50cf4c4930d4baad691dbc973e1720cbf1b3756

RUN apt-get update \
    && apt-get install -y --no-install-recommends git ca-certificates docker.io docker-cli \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir nats-py==2.15.0 docker==7.1.0 pyyaml==6.0.3 pyzmq==27.1.0 msgpack==1.2.1

COPY docker/swepro-session/runner.py /opt/swepro-session/runner.py
COPY examples/swebench-pro/dynamo_agent_trace.py /opt/swepro-session/dynamo_agent_trace.py

ENV SWE_AGENT_CONFIG_ROOT=/code/SWE-bench_Pro-os/SWE-agent
ENV SWEPRO_EVAL_ROOT=/code/SWE-bench_Pro-os
ENV DOCKER_HOST=unix:///var/run/docker-swepro.sock

ENTRYPOINT ["python", "/opt/swepro-session/runner.py"]
