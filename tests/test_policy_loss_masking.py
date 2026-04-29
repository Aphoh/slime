import pytest

torch = pytest.importorskip("torch")

from slime.utils.ppo_utils import compute_policy_loss


def test_policy_loss_masks_nonfinite_inactive_positions():
    ppo_kl = torch.tensor([float("nan"), float("-inf"), float("inf"), 0.2])
    advantages = torch.tensor([1.0, 1.0, 1.0, 1.0])
    loss_mask = torch.tensor([0, 0, 0, 1])

    pg_loss, clipfrac = compute_policy_loss(
        ppo_kl,
        advantages,
        eps_clip=0.2,
        eps_clip_high=0.2,
        loss_mask=loss_mask,
    )

    assert torch.isfinite(pg_loss).all()
    assert torch.isfinite(clipfrac).all()
    assert pg_loss[:3].tolist() == [0.0, 0.0, 0.0]
    assert clipfrac[:3].tolist() == [0.0, 0.0, 0.0]


def test_policy_loss_zero_advantage_does_not_form_zero_times_inf():
    ppo_kl = torch.tensor([float("-inf"), -1000.0, float("inf")])
    advantages = torch.zeros_like(ppo_kl)

    pg_loss, clipfrac = compute_policy_loss(
        ppo_kl,
        advantages,
        eps_clip=0.2,
        eps_clip_high=0.2,
    )

    assert torch.isfinite(pg_loss).all()
    assert torch.isfinite(clipfrac).all()
    assert pg_loss.tolist() == [0.0, 0.0, 0.0]
