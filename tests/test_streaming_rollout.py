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

from slime.rollout import sglang_rollout
from slime.rollout import sglang_streaming_rollout as streaming
from slime.rollout.streaming_utils import SGLangStreamAccumulator
from slime.utils.types import Sample

NUM_GPUS = 0


def _chunk(text, pairs, output_length, **meta_info):
    return {
        "text": text,
        "meta_info": {
            "output_token_logprobs": pairs,
            "output_token_logprobs_length": output_length,
            **meta_info,
        },
    }


@pytest.mark.parametrize(
    ("initial", "chunks", "expected"),
    [
        (
            ("cumulative", [], [], ""),
            [
                _chunk("a", [[-0.1, 11, None]], 1),
                _chunk("ab", [[-0.1, 11, None], [-0.2, 12, None]], 2),
            ],
            ([11, 12], [-0.1, -0.2], "ab"),
        ),
        (
            ("incremental", [], [], ""),
            [
                _chunk("a", [[-0.1, 11, None]], 1),
                _chunk("b", [[-0.2, 12, None]], 2),
                _chunk("", [], 2),
            ],
            ([11, 12], [-0.1, -0.2], "ab"),
        ),
        (
            ("cumulative", [11], [-0.1], "a"),
            [_chunk(None, [[-0.1, 11, None]], 1)],
            ([11], [-0.1], "a"),
        ),
    ],
    ids=["cumulative", "disjoint", "null-text"],
)
def test_merge_stream_chunks(initial, chunks, expected):
    output_mode, tokens, log_probs, text = initial
    accumulator = SGLangStreamAccumulator(
        output_mode=output_mode,
        output_length=len(tokens),
        tokens=list(tokens),
        text=text,
    )
    for chunk in chunks:
        update = accumulator.add(
            chunk=chunk,
            decode=lambda _tokens, current_text=text: current_text,
        )
        tokens, text = accumulator.tokens, accumulator.text
        log_probs = update.log_probs if update.replace_call_state else log_probs + update.log_probs

    assert (tokens, log_probs, text) == expected


def test_merge_stream_rejects_inconsistent_length():
    accumulator = SGLangStreamAccumulator(
        output_mode="incremental",
        output_length=1,
        tokens=[11],
        text="a",
    )
    with pytest.raises(ValueError, match="output_token_logprobs_length"):
        accumulator.add(
            chunk=_chunk("bc", [[-0.2, 12, None], [-0.3, 13, None]], 4),
            decode=lambda _tokens: "",
        )


def _generation_state(*, abort_mode="request", stream_output_mode="incremental"):
    return SimpleNamespace(
        tokenizer=SimpleNamespace(
            decode=lambda token_ids, skip_special_tokens=False: "".join(f"<{token_id}>" for token_id in token_ids)
        ),
        processor=None,
        aborted=False,
        abort_mode=abort_mode,
        stream_output_mode=stream_output_mode,
        cancellable_tasks=set(),
    )


def _streaming_args():
    return SimpleNamespace(
        ci_test=False,
        sglang_router_ip="frontend",
        sglang_router_port=8000,
        use_rollout_routing_replay=False,
        router_policy=None,
        sglang_speculative_algorithm=False,
    )


def _patch_streaming(monkeypatch, state, stream):
    monkeypatch.setattr(streaming, "GenerateState", lambda _args: state)
    monkeypatch.setattr(streaming, "_prepare_prompt_ids", lambda *_args: [1, 2])
    monkeypatch.setattr(streaming.http_utils, "_http_client", SimpleNamespace(stream=stream))
    monkeypatch.setattr(
        streaming,
        "trace_span",
        lambda *_args, **_kwargs: nullcontext(SimpleNamespace(update=lambda *_args, **_kwargs: None)),
    )


def test_streaming_generator_rejects_server_abort_mode(monkeypatch):
    args = SimpleNamespace(ci_test=False)
    monkeypatch.setattr(streaming, "GenerateState", lambda _args: _generation_state(abort_mode="server"))
    with pytest.raises(RuntimeError, match="must be the globally configured"):
        asyncio.run(streaming.generate_streaming(args, Sample(prompt="hello"), {"max_new_tokens": 1}))


def test_stream_accumulator_requires_reported_length():
    with pytest.raises(ValueError, match="must include output_token_logprobs_length"):
        SGLangStreamAccumulator(output_mode="incremental").add(
            {"text": "x", "meta_info": {"output_token_logprobs": [[-0.1, 11, None]]}},
            decode=lambda _tokens: "",
        )


def test_stream_accumulator_rejects_output_incompatible_with_configured_mode():
    accumulator = SGLangStreamAccumulator(output_mode="incremental")
    accumulator.add(_chunk("a", [[-0.1, 11, None]], 1), decode=lambda _tokens: "")

    with pytest.raises(ValueError, match="incremental streaming output has inconsistent"):
        accumulator.add(
            _chunk(
                "ab",
                [[-0.1, 11, None], [-0.2, 12, None]],
                2,
            ),
            decode=lambda _tokens: "",
        )


@pytest.mark.parametrize("stream_interval", [1, 20, 64])
@pytest.mark.parametrize("incremental", [True, False], ids=["incremental", "cumulative"])
@pytest.mark.parametrize("top_p_layout", ["final_full", "per_chunk"])
def test_generate_streaming_preserves_metadata_across_stream_intervals(
    monkeypatch, stream_interval, incremental, top_p_layout
):
    # Keep the response longer than the largest interval so the second chunk
    # makes cumulative versus incremental output observable on the wire.
    response_tokens = list(range(11, 76))
    chunks = []
    previous = 0
    for end in range(stream_interval, len(response_tokens) + stream_interval, stream_interval):
        end = min(end, len(response_tokens))
        start = previous if incremental else 0
        token_slice = response_tokens[start:end]
        is_final = end == len(response_tokens)
        top_p_tokens = token_slice if top_p_layout == "per_chunk" else response_tokens
        top_p_metadata = (
            {
                "top_p_token_ids": [token_id + 1000 for token_id in top_p_tokens],
                "top_p_token_offsets": list(range(len(top_p_tokens) + 1)),
            }
            if top_p_layout == "per_chunk" or is_final
            else {}
        )
        chunks.append(
            _chunk(
                "".join(f"<{token_id}>" for token_id in token_slice),
                [[-float(token_id), token_id, None] for token_id in token_slice],
                end,
                finish_reason={"type": "stop"} if is_final else None,
                **top_p_metadata,
            )
        )
        previous = end
        if end == len(response_tokens):
            break

    async def lines():
        for chunk in chunks:
            yield "data: " + json.dumps(chunk)

    response = SimpleNamespace(raise_for_status=lambda: None, aiter_lines=lines)

    @asynccontextmanager
    async def stream(method, url, json, headers):
        assert (method, url, json["stream"]) == ("POST", "http://frontend:8000/generate", True)
        yield response

    state = _generation_state(stream_output_mode="incremental" if incremental else "cumulative")
    _patch_streaming(monkeypatch, state, stream)
    args = _streaming_args()

    result = asyncio.run(
        streaming.generate_streaming(
            args,
            Sample(prompt="hello"),
            {"max_new_tokens": len(response_tokens), "skip_special_tokens": False},
        )
    )

    assert result.status == Sample.Status.COMPLETED
    assert result.tokens == [1, 2, *response_tokens]
    assert result.response_length == len(response_tokens)
    assert result.rollout_log_probs == [-float(token_id) for token_id in response_tokens]
    assert result.rollout_top_p_token_ids.tolist() == [token_id + 1000 for token_id in response_tokens]
    assert result.rollout_top_p_token_offsets.tolist() == list(range(len(response_tokens) + 1))


def test_stream_cancellation_closes_request_and_keeps_prefix(monkeypatch):
    first_chunk_seen = asyncio.Event()
    request_closed = False
    state = _generation_state()

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

    _patch_streaming(monkeypatch, state, stream)
    args = _streaming_args()

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
    assert not state.cancellable_tasks


def test_unrelated_stream_cancellation_propagates(monkeypatch):
    request_started = asyncio.Event()
    state = _generation_state()

    async def lines():
        request_started.set()
        await asyncio.Event().wait()
        yield "unreachable"

    response = SimpleNamespace(raise_for_status=lambda: None, aiter_lines=lines)

    @asynccontextmanager
    async def stream(method, url, json, headers):
        yield response

    _patch_streaming(monkeypatch, state, stream)
    args = _streaming_args()

    async def exercise():
        task = asyncio.create_task(
            streaming.generate_streaming(
                args,
                Sample(prompt="hello"),
                {"max_new_tokens": 8},
            )
        )
        await request_started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(exercise())
    assert not state.cancellable_tasks


def test_partial_abort_buffers_mixed_group_with_nonempty_aborted_sample(monkeypatch):
    partial = Sample(response="x", response_length=1, status=Sample.Status.ABORTED)
    terminal = Sample(response="done", response_length=2, status=Sample.Status.TRUNCATED)
    empty = Sample(response="", response_length=0, status=Sample.Status.ABORTED)
    mixed_group = [terminal, partial]

    async def exercise():
        state = SimpleNamespace(
            aborted=False,
            abort_mode="request",
            cancellable_tasks=set(),
            pendings={
                asyncio.create_task(asyncio.sleep(0, result=group)) for group in (mixed_group, [terminal], [empty])
            },
        )
        monkeypatch.setattr(sglang_rollout, "GenerateState", lambda _args: state)
        return await sglang_rollout.abort(SimpleNamespace(partial_rollout=True), rollout_id=7)

    assert asyncio.run(exercise()) == [mixed_group]
    assert partial.metadata["start_rollout_id"] == 7
    assert terminal.metadata["start_rollout_id"] == 7


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
