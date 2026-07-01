FROM python:3.11-slim@sha256:b27df5841f3355e9473f9a516d38a6783b6c8dfeacaf2d14a240f443b368ddb6

RUN apt-get update \
    && apt-get install -y --no-install-recommends git ca-certificates docker.io \
    && rm -rf /var/lib/apt/lists/*

COPY docker/swepro-eval/requirements.lock /tmp/swepro-eval-requirements.lock
RUN pip install --no-cache-dir -r /tmp/swepro-eval-requirements.lock && \
    rm /tmp/swepro-eval-requirements.lock

COPY docker/swepro-eval/runner.py /opt/swepro-eval/runner.py

ENV DOCKER_HOST=unix:///var/run/docker-swepro.sock
ENTRYPOINT ["python", "/opt/swepro-eval/runner.py"]
