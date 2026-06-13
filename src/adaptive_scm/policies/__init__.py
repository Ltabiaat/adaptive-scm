"""Inventory replenishment policies. EOQ, OrderUpTo, and PPO all implement the ``Policy`` ABC."""

from adaptive_scm.policies.base import Policy, State
from adaptive_scm.policies.eoq import EOQPolicy
from adaptive_scm.policies.order_up_to import OrderUpToPolicy

__all__ = ["Policy", "State", "EOQPolicy", "OrderUpToPolicy"]

try:
    from adaptive_scm.policies.ppo import PPOAgent, PPOHyperparams  # noqa: F401

    __all__.extend(["PPOAgent", "PPOHyperparams"])
except ImportError:  # pragma: no cover - exercised only without the deep extra
    pass
