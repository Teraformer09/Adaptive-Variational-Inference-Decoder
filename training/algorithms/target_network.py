"""Target Network Synchronization Manager for R2D2."""

from __future__ import annotations

from training.models.r2d2_network import R2D2Network


class TargetNetworkManager:
    """Manages periodic hard synchronization or Polyak soft synchronization of target network."""

    def __init__(
        self,
        online_net: R2D2Network,
        sync_interval: int = 25,
        tau: float = 1.0,
    ) -> None:
        self.online_net = online_net
        self.target_net = online_net.clone()
        self.sync_interval = int(sync_interval)
        self.tau = float(tau)
        self.step_count = 0

    def hard_sync(self) -> None:
        """Copy online network weights into target network."""
        self.target_net.load_state_dict(self.online_net.get_state_dict())

    def soft_sync(self, tau: float) -> None:
        """Polyak averaging: theta_target <- (1 - tau) * theta_target + tau * theta_online."""
        online_params = self.online_net.parameters()
        target_params = self.target_net.parameters()
        for k in online_params:
            target_params[k][:] = (
                (1.0 - tau) * target_params[k] + tau * online_params[k]
            )

    def step(self) -> bool:
        """Advance step counter and synchronize if sync_interval reached."""
        self.step_count += 1
        if self.step_count % self.sync_interval == 0:
            if self.tau >= 1.0:
                self.hard_sync()
            else:
                self.soft_sync(self.tau)
            return True
        return False
