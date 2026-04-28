FROM python:3.11-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends git ca-certificates docker.io \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir docker nats-py pandas tqdm

COPY runner.py /opt/swepro-eval/runner.py

ENV DOCKER_HOST=unix:///var/run/docker-swepro.sock
ENTRYPOINT ["python", "/opt/swepro-eval/runner.py"]
