#!/usr/bin/env python3
"""Manual smoke client for the SWE-bench Pro session worker."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path

from sweagent_session import SweAgentSessionClient


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nats-url", default=os.getenv("SWEPRO_NATS_URL", "nats://warnold-swepro-nats:4222"))
    parser.add_argument("--instance-id", required=True)
    parser.add_argument("--image-name", required=True)
    parser.add_argument("--base-commit", required=True)
    parser.add_argument("--repo-name", default="app")
    args = parser.parse_args()

    client = SweAgentSessionClient(args.nats_url)
    started = await client.start(
        instance_id=args.instance_id,
        image_name=args.image_name,
        base_commit=args.base_commit,
        repo_name=args.repo_name,
    )
    session_id = started["session_id"]
    print(json.dumps({"started": started}, indent=2))
    try:
        tool_call = {
            "id": "call_smoke_bash",
            "type": "function",
            "function": {
                "name": "bash",
                "arguments": json.dumps({"command": "pwd && echo ROOT=$ROOT && which str_replace_editor && which submit"}),
            },
        }
        step = await client.step(session_id, tool_call)
        print(json.dumps({"bash": step}, indent=2))

        submit_call = {
            "id": "call_smoke_submit",
            "type": "function",
            "function": {"name": "submit", "arguments": "{}"},
        }
        submitted = await client.step(session_id, submit_call)
        print(json.dumps({"submit": submitted}, indent=2))
    finally:
        closed = await client.close(session_id)
        print(json.dumps({"closed": closed}, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
