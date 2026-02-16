"""Actor-Critic neural network for RL agents.

Architecture:
- Shared backbone: fully connected layers with ReLU activations
- Actor head: outputs action logits (one per possible action)
- Critic head: outputs state value estimate (single scalar)

The shared backbone extracts common features used by both the policy
(actor) and value function (critic), which improves sample efficiency.
"""

import torch
import torch.nn as nn
from typing import List, Optional, Tuple


class ActorCriticNetwork(nn.Module):
    """Actor-Critic network with shared backbone.

    Args:
        input_size: Size of input feature vector (e.g. 106)
        action_size: Number of possible actions (e.g. 100)
        hidden_sizes: List of hidden layer sizes for the backbone
    """

    def __init__(
        self,
        input_size: int,
        action_size: int,
        hidden_sizes: Optional[List[int]] = None,
    ):
        super().__init__()

        if hidden_sizes is None:
            hidden_sizes = [128, 64]

        # Build shared backbone
        backbone_layers = []
        prev_size = input_size
        for hidden_size in hidden_sizes:
            backbone_layers.append(nn.Linear(prev_size, hidden_size))
            backbone_layers.append(nn.ReLU())
            prev_size = hidden_size

        self.backbone = nn.Sequential(*backbone_layers)

        # Actor head: outputs action logits
        self.actor_head = nn.Linear(prev_size, action_size)

        # Critic head: outputs state value
        self.critic_head = nn.Linear(prev_size, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: Input features [batch_size, input_size]

        Returns:
            Tuple of (action_logits [batch_size, action_size],
                      state_value [batch_size, 1])
        """
        shared = self.backbone(x)
        logits = self.actor_head(shared)
        value = self.critic_head(shared)
        return logits, value

    def apply_action_mask(
        self, logits: torch.Tensor, action_mask: torch.Tensor
    ) -> torch.Tensor:
        """Mask invalid actions by setting their logits to -inf.

        Args:
            logits: Action logits [batch_size, action_size]
            action_mask: Boolean mask, True = valid [batch_size, action_size]

        Returns:
            Masked logits with invalid actions set to -inf
        """
        masked = logits.clone()
        masked[~action_mask] = float('-inf')
        return masked


def create_actor_critic_network(
    feature_size: int,
    action_size: int,
    hidden_sizes: Optional[List[int]] = None,
) -> ActorCriticNetwork:
    """Factory function to create an ActorCriticNetwork.

    Args:
        feature_size: Size of input feature vector
        action_size: Number of possible actions
        hidden_sizes: Optional list of hidden layer sizes

    Returns:
        Configured ActorCriticNetwork
    """
    return ActorCriticNetwork(
        input_size=feature_size,
        action_size=action_size,
        hidden_sizes=hidden_sizes,
    )
