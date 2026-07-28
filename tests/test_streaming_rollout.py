import asyncio
import json
import sys
from contextlib import asynccontextmanager, nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import sglang_router  # noqa: F401
except ImportError:
    sys.modules["sglang_router"] = SimpleNamespace(__version__="0.3.0")

try:
    import transformers  # noqa: F401
except ImportError:
    sys.modules["transformers"] = SimpleNamespace(
        AutoProcessor=object,
        AutoTokenizer=object,
        PreTrainedTokenizerBase=object,
        ProcessorMixin=object,
    )

from slime.rollout import sglang_rollout, sglang_streaming_rollout as streaming
from slime.rollout.streaming_utils import merge_stream_chunk
from slime.utils.types import Sample

NUM_GPUS = 0


def _chunk(text, pairs, output_length):
    return {
        "text": text,
        "meta_info": {
            "output_token_logprobs": pairs,
            "output_token_logprobs_length": output_length,
        },
    }


@pytest.mark.parametrize(
    ("initial", "chunks", "expected"),
    [
        (
            ([], [], ""),
            [
                _chunk("a", [[-0.1, 11, None]], 1),
                _chunk("ab", [[-0.1, 11, None], [-0.2, 12, None]], 2),
            ],
            ([11, 12], [-0.1, -0.2], "ab"),
        ),
        (
            ([], [], ""),
            [
                _chunk("a", [[-0.1, 11, None]], 1),
                _chunk("b", [[-0.2, 12, None]], 2),
                _chunk("", [], 2),
            ],
            ([11, 12], [-0.1, -0.2], "ab"),
        ),
        (
            ([11], [-0.1], "a"),
            [_chunk(None, [[-0.1, 11, None]], 1)],
            ([11], [-0.1], "a"),
        ),
    ],
    ids=["cumulative", "disjoint", "null-text"],
)
def test_merge_stream_chunks(initial, chunks, expected):
    tokens, log_probs, text = initial
    for chunk in chunks:
        tokens, log_probs, text = merge_stream_chunk(
            tokens=tokens,
            log_probs=log_probs,
            text=text,
            chunk=chunk,
        )

    assert (tokens, log_probs, text) == expected


def test_merge_stream_rejects_inconsistent_length():
    with pytest.raises(ValueError, match="output_token_logprobs_length"):
        merge_stream_chunk(
            tokens=[11],
            log_probs=[-0.1],
            text="a",
            chunk=_chunk("bc", [[-0.2, 12, None], [-0.3, 13, None]], 4),
        )


def test_stream_cancellation_closes_request_and_keeps_prefix(monkeypatch):
    first_chunk_seen = asyncio.Event()
    request_closed = False
    state = SimpleNamespace(
        tokenizer=SimpleNamespace(
            decode=lambda token_ids, skip_special_tokens=False: "".join(f"<{token_id}>" for token_id in token_ids)
        ),
        processor=None,
        aborted=False,
        streaming_generation=False,
        streaming_tasks=set(),
    )

    async def lines():
        yield "data: " + json.dumps(
            {
                "text": None,
                "meta_info": {
                    "output_token_logprobs": [[-0.1, 11, None]],
                    "output_token_logprobs_length": 1,
                },
            }
        )
        first_chunk_seen.set()
        await asyncio.Event().wait()

    response = SimpleNamespace(raise_for_status=lambda: None, aiter_lines=lines)

    @asynccontextmanager
    async def stream(method, url, json, headers):
        nonlocal request_closed
        assert (method, url, json["stream"]) == ("POST", "http://frontend:8000/generate", True)
        try:
            yield response
        finally:
            request_closed = True

    monkeypatch.setattr(streaming, "GenerateState", lambda _args: state)
    monkeypatch.setattr(streaming, "_prepare_prompt_ids", lambda *_args: [1, 2])
    monkeypatch.setattr(streaming.http_utils, "_http_client", SimpleNamespace(stream=stream))
    monkeypatch.setattr(
        streaming,
        "trace_span",
        lambda *_args, **_kwargs: nullcontext(SimpleNamespace(update=lambda *_args, **_kwargs: None)),
    )
    args = SimpleNamespace(
        ci_test=False,
        sglang_router_ip="frontend",
        sglang_router_port=8000,
        use_rollout_routing_replay=False,
        router_policy=None,
    )

    async def exercise():
        task = asyncio.create_task(streaming.generate_streaming(args, Sample(prompt="hello"), {"max_new_tokens": 8}))
        await first_chunk_seen.wait()
        state.aborted = True
        task.cancel()
        return await task

    result = asyncio.run(exercise())

    assert result.status == Sample.Status.ABORTED
    assert (result.tokens, result.rollout_log_probs, result.response) == ([1, 2, 11], [-0.1], "<11>")
    assert request_closed is True
    assert state.streaming_generation is True
    assert not state.streaming_tasks


def test_partial_abort_buffers_only_nonempty_fully_aborted_groups(monkeypatch):
    partial = Sample(response="x", response_length=1, status=Sample.Status.ABORTED)
    terminal = Sample(response="done", response_length=2, status=Sample.Status.TRUNCATED)
    empty = Sample(response="", response_length=0, status=Sample.Status.ABORTED)

    async def exercise():
        state = SimpleNamespace(
            aborted=False,
            streaming_generation=True,
            streaming_tasks=set(),
            pendings={
                asyncio.create_task(asyncio.sleep(0, result=group))
                for group in ([partial], [terminal], [empty])
            },
        )
        monkeypatch.setattr(sglang_rollout, "GenerateState", lambda _args: state)
        return await sglang_rollout.abort(SimpleNamespace(partial_rollout=True), rollout_id=7)

    assert asyncio.run(exercise()) == [[partial]]
    assert partial.metadata["start_rollout_id"] == 7


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
