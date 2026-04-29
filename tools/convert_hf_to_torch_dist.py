import gc
import os
from pathlib import Path

import torch
import torch.distributed as dist
from megatron.core.enums import ModelType
from megatron.training.arguments import parse_args, validate_args
from megatron.training.checkpointing import get_checkpoint_name, get_checkpoint_tracker_filename, save_checkpoint
from megatron.training.training import get_model

import slime_plugins.mbridge  # noqa: F401
from mbridge import AutoBridge
from slime.backends.megatron_utils.arguments import set_default_megatron_args
from slime.backends.megatron_utils.initialize import init
from slime.backends.megatron_utils.model_provider import get_model_provider_func
from slime.utils.logging_utils import configure_logger
from slime.utils.memory_utils import print_memory


def add_convertion_args(parser):
    """Add conversion arguments to the parser"""
    parser.add_argument("--hf-checkpoint", type=str, required=True, help="HuggingFace model path")
    parser.add_argument(
        "--megatron-to-hf-mode",
        choices=["raw", "bridge"],
        default="raw",
        help="The method to convert megatron weights to hugging face weights for SGLang.",
    )
    try:
        parser.add_argument("--padded-vocab-size", type=int, default=None)
    except Exception:
        pass
    return parser


def get_args():
    args = parse_args(add_convertion_args)
    args = set_default_megatron_args(args)

    # set to pass megatron validate_args
    args.save_interval = 1
    args.micro_batch_size = 1
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    args.global_batch_size = int(os.environ.get("WORLD_SIZE", "1"))

    assert world_size <= args.num_layers, (
        f"World size {world_size} must be less than or equal to number of layers {args.num_layers}. "
        "You are using too many GPUs for this conversion."
    )

    def ceildiv(a, b):
        return -(a // -b)

    auto_pipeline = os.environ.get("SLIME_CONVERT_AUTO_PIPELINE", "1") not in {"0", "false", "False", "no", "No"}
    if auto_pipeline and args.pipeline_model_parallel_size == 1 and world_size > 1:
        pp_size = world_size
        while True:
            args.pipeline_model_parallel_size = pp_size
            args.decoder_last_pipeline_num_layers = args.num_layers - ceildiv(
                args.num_layers, args.pipeline_model_parallel_size
            ) * (args.pipeline_model_parallel_size - 1)

            if args.decoder_last_pipeline_num_layers > 0:
                break

            if pp_size % 2 == 0:
                pp_size //= 2
            else:
                raise ValueError(
                    f"Cannot find a valid pipeline model parallel size for {args.num_layers} layers and {world_size} GPUs."
                )
    print(
        f"Using pipeline model parallel size: {args.pipeline_model_parallel_size}, decoder last pipeline num layers: {args.decoder_last_pipeline_num_layers}"
    )

    validate_args(args)
    return args


def _env_true(name: str) -> bool:
    return os.environ.get(name, "0") in {"1", "true", "True", "yes", "Yes"}


def _broadcast_file_from_rank0(path: Path, local_rank: int, global_rank: int) -> None:
    if global_rank == 0:
        payload = path.read_bytes() if path.exists() else None
        size = torch.tensor([-1 if payload is None else len(payload)], dtype=torch.long)
    else:
        payload = None
        size = torch.tensor([-1], dtype=torch.long)

    dist.broadcast(size, src=0)
    if size.item() < 0:
        return

    if global_rank == 0:
        data = torch.frombuffer(bytearray(payload), dtype=torch.uint8)
    else:
        data = torch.empty(size.item(), dtype=torch.uint8)
    dist.broadcast(data, src=0)

    if local_rank == 0 and global_rank != 0:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data.numpy().tobytes())


def main():
    if torch.version.hip:
        import megatron.core.dist_checkpointing.strategies.filesystem_async as filesystem_async_module
        from slime.utils.rocm_checkpoint_writer import ROCmFileSystemWriterAsync

        filesystem_async_module.FileSystemWriterAsync = ROCmFileSystemWriterAsync
        print("[ROCm] Applied FileSystemWriterAsync patch for HIP compatibility")

    configure_logger()

    # Initialize distributed environment
    world_size = int(os.getenv("WORLD_SIZE") or os.getenv("SLURM_NTASKS") or 1)
    local_rank = int(os.getenv("LOCAL_RANK") or os.getenv("SLURM_LOCALID") or 0)
    global_rank = int(os.getenv("RANK") or os.getenv("SLURM_PROCID") or 0)

    torch.cuda.set_device(local_rank)
    os.environ.setdefault("WORLD_SIZE", str(world_size))
    os.environ.setdefault("RANK", str(global_rank))
    os.environ.setdefault("LOCAL_RANK", str(local_rank))
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "12355")
    process_group_backend = os.environ.get("SLIME_CONVERT_PROCESS_GROUP_BACKEND", "nccl")
    init_kwargs = {
        "backend": process_group_backend,
        "world_size": world_size,
        "rank": global_rank,
    }
    if process_group_backend == "nccl":
        init_kwargs["device_id"] = torch.device(f"cuda:{local_rank}")
    dist.init_process_group(**init_kwargs)
    args = get_args()
    init(args)

    # if using AMD gpus, we have to do the conversion in cpu
    if hasattr(torch.version, "hip") and torch.version.hip is not None:
        assert args.use_cpu_initialization, "AMD GPU requires --use_cpu_initialization=True"

    model = get_model(get_model_provider_func(args), ModelType.encoder_or_decoder, wrap_with_ddp=False)

    # Load model
    hf_model_path = args.hf_checkpoint
    bridge = AutoBridge.from_pretrained(hf_model_path, trust_remote_code=True)
    bridge.load_weights(model, hf_model_path, memory_efficient=True)
    print(f"Model loaded: {hf_model_path}")

    if args.use_cpu_initialization:
        model[0] = model[0].cpu()

    print_memory("after loading model")
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()

    save_checkpoint(1, model, None, None, 0, release=True)

    if _env_true("SLIME_CONVERT_LOCAL_CHECKPOINT"):
        checkpoint_dir = Path(get_checkpoint_name(args.save, -1, True, return_base_dir=True))
        for filename in ("common.pt", ".metadata"):
            _broadcast_file_from_rank0(checkpoint_dir / filename, local_rank, global_rank)
        if local_rank == 0:
            tracker_filename = get_checkpoint_tracker_filename(args.save)
            with open(tracker_filename, "w") as f:
                f.write("release")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
