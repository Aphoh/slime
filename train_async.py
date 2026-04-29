import logging
import os

import ray

from slime.ray.placement_group import create_placement_groups, create_rollout_manager, create_training_models
from slime.utils.arguments import parse_args
from slime.utils.logging_utils import configure_logger, finish_tracking, init_tracking
from slime.utils.misc import should_run_periodic_action
from slime.utils.speedscope_trace import trace_span

logger = logging.getLogger(__name__)


# The framework supports other asynchronous approaches such as fully async (which is shown in examples/full_async).
def train(args):
    assert not args.colocate, "Colocation is not supported for async training."
    configure_logger()
    # allocate the GPUs
    pgs = create_placement_groups(args)
    init_tracking(args)

    # create the rollout manager, with sglang engines inside.
    # need to initialize rollout manager first to calculate num_rollout
    rollout_manager, num_rollout_per_epoch = create_rollout_manager(args, pgs["rollout"])

    # create the actor and critic models
    actor_model, critic_model = create_training_models(args, pgs, rollout_manager)

    # Always push actor weights to rollout once weights are loaded.
    with trace_span("driver", "weights.initial_update"):
        actor_model.update_weights()

    if args.check_weight_update_equal:
        ray.get(rollout_manager.check_weights.remote(action="compare"))

    # async train loop.
    allow_stale_rollouts = os.getenv("SLIME_ASYNC_ALLOW_STALE_ROLLOUTS", "0") == "1"
    if allow_stale_rollouts:
        logger.info("SLIME_ASYNC_ALLOW_STALE_ROLLOUTS=1: weight updates will not wait for prefetched rollout data.")

    with trace_span("driver", "rollout.submit", rollout_id=args.start_rollout_id):
        rollout_data_next_future = rollout_manager.generate.remote(args.start_rollout_id)
    for rollout_id in range(args.start_rollout_id, args.num_rollout):
        # Sync the last generation
        if rollout_data_next_future is not None:
            with trace_span("driver", "rollout.wait", rollout_id=rollout_id):
                rollout_data_curr_ref = ray.get(rollout_data_next_future)

        # Start the next rollout early.
        if rollout_id + 1 < args.num_rollout:
            with trace_span("driver", "rollout.submit", rollout_id=rollout_id + 1):
                rollout_data_next_future = rollout_manager.generate.remote(rollout_id + 1)

        if args.use_critic:
            actor_trains_this_step = rollout_id >= args.num_critic_only_steps
            with trace_span("driver", "trainer.critic.submit", rollout_id=rollout_id):
                value_refs = critic_model.async_train(rollout_id, rollout_data_curr_ref)
            if actor_trains_this_step:
                with trace_span("driver", "trainer.actor.wait", rollout_id=rollout_id):
                    ray.get(actor_model.async_train(rollout_id, rollout_data_curr_ref, external_data=value_refs))
            else:
                with trace_span("driver", "trainer.critic.wait", rollout_id=rollout_id):
                    ray.get(value_refs)
        else:
            with trace_span("driver", "trainer.actor.wait", rollout_id=rollout_id):
                ray.get(actor_model.async_train(rollout_id, rollout_data_curr_ref))

        if should_run_periodic_action(rollout_id, args.save_interval, num_rollout_per_epoch, args.num_rollout):
            if (not args.use_critic) or rollout_id >= args.num_critic_only_steps:
                actor_model.save_model(
                    rollout_id,
                    force_sync=rollout_id == args.num_rollout - 1,
                )
            if args.use_critic:
                critic_model.save_model(
                    rollout_id,
                    force_sync=rollout_id == args.num_rollout - 1,
                )
            if args.rollout_global_dataset:
                ray.get(rollout_manager.save.remote(rollout_id))

        if (rollout_id + 1) % args.update_weights_interval == 0:
            if allow_stale_rollouts:
                logger.info(
                    "Updating weights without draining prefetched rollout future at rollout_id=%s. "
                    "In-flight rollouts may train with stale-policy samples.",
                    rollout_id,
                )
            else:
                # sync generate before update weights to prevent update weight in the middle of generation
                rollout_data_curr_ref = ray.get(x) if (x := rollout_data_next_future) is not None else None
                rollout_data_next_future = None
            with trace_span("driver", "weights.update", rollout_id=rollout_id):
                actor_model.update_weights()

        if should_run_periodic_action(rollout_id, args.eval_interval, num_rollout_per_epoch):
            ray.get(rollout_manager.eval.remote(rollout_id))

    ray.get(rollout_manager.dispose.remote())
    finish_tracking(args)


if __name__ == "__main__":
    args = parse_args()
    train(args)
