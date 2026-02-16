"""Unit tests for neural networks - Actor-Critic architecture.

Tests the PyTorch neural network architectures used for RL agents.

Architecture:
- Shared feature extraction backbone (2+ hidden layers)
- Actor head: outputs action logits
- Critic head: outputs state value estimate

Key features tested:
- Network initialization and architecture
- Forward pass (features → logits + value)
- Correct output dimensions
- Action masking support
- Different hidden layer sizes
- Gradient flow
- Device placement (CPU/CUDA)

The network uses:
- Input: Feature vector (106 features)
- Hidden: 2-3 fully connected layers with ReLU
- Actor output: Action logits (100 actions)
- Critic output: State value (1 scalar)
"""

import pytest
import torch
import numpy as np
from agents.rl.networks import ActorCriticNetwork, create_actor_critic_network
from agents.rl.features import FeatureExtractor


class TestActorCriticNetworkBasics:
    """Test basic network creation and architecture."""

    def test_network_creation(self):
        """Can create an ActorCriticNetwork."""
        network = ActorCriticNetwork(
            input_size=106,
            action_size=100,
            hidden_sizes=[128, 64]
        )
        assert network is not None

    def test_network_has_correct_layers(self):
        """Network has backbone, actor head, and critic head."""
        network = ActorCriticNetwork(
            input_size=106,
            action_size=100,
            hidden_sizes=[128]
        )

        # Should have backbone (shared layers)
        assert hasattr(network, 'backbone')

        # Should have actor head (policy)
        assert hasattr(network, 'actor_head')

        # Should have critic head (value)
        assert hasattr(network, 'critic_head')

    def test_network_with_different_hidden_sizes(self):
        """Network supports different hidden layer configurations."""
        # Single hidden layer
        net1 = ActorCriticNetwork(input_size=106, action_size=100, hidden_sizes=[128])
        assert net1 is not None

        # Two hidden layers
        net2 = ActorCriticNetwork(input_size=106, action_size=100, hidden_sizes=[256, 128])
        assert net2 is not None

        # Three hidden layers
        net3 = ActorCriticNetwork(input_size=106, action_size=100, hidden_sizes=[256, 128, 64])
        assert net3 is not None

    def test_network_default_hidden_sizes(self):
        """Network has reasonable default hidden sizes."""
        network = ActorCriticNetwork(input_size=106, action_size=100)
        # Should have some hidden layers by default
        assert hasattr(network, 'backbone')


class TestActorCriticForward:
    """Test forward pass through the network."""

    def test_forward_pass_returns_two_outputs(self):
        """Forward pass returns (action_logits, state_value)."""
        network = ActorCriticNetwork(input_size=106, action_size=100, hidden_sizes=[128])

        # Create batch of features
        batch = torch.randn(4, 106)

        logits, value = network(batch)

        assert logits is not None
        assert value is not None

    def test_action_logits_have_correct_shape(self):
        """Action logits have shape [batch_size, action_size]."""
        network = ActorCriticNetwork(input_size=106, action_size=100, hidden_sizes=[128])

        batch_size = 8
        batch = torch.randn(batch_size, 106)

        logits, _ = network(batch)

        assert logits.shape == (batch_size, 100)

    def test_state_value_has_correct_shape(self):
        """State value has shape [batch_size, 1]."""
        network = ActorCriticNetwork(input_size=106, action_size=100, hidden_sizes=[128])

        batch_size = 8
        batch = torch.randn(batch_size, 106)

        _, value = network(batch)

        assert value.shape == (batch_size, 1)

    def test_forward_with_single_sample(self):
        """Forward pass works with single sample (no batch)."""
        network = ActorCriticNetwork(input_size=106, action_size=100, hidden_sizes=[128])

        # Single sample - shape [106]
        sample = torch.randn(106)

        logits, value = network(sample.unsqueeze(0))

        assert logits.shape == (1, 100)
        assert value.shape == (1, 1)

    def test_forward_with_numpy_input(self):
        """Network can handle numpy array input (converted to tensor)."""
        network = ActorCriticNetwork(input_size=106, action_size=100, hidden_sizes=[128])

        # NumPy input
        np_input = np.random.randn(4, 106).astype(np.float32)
        tensor_input = torch.from_numpy(np_input)

        logits, value = network(tensor_input)

        assert logits.shape == (4, 100)
        assert value.shape == (4, 1)


class TestActionMasking:
    """Test action masking for invalid actions."""

    def test_apply_action_mask_zeros_invalid_actions(self):
        """Applying action mask sets invalid actions to -inf."""
        network = ActorCriticNetwork(input_size=106, action_size=100, hidden_sizes=[128])

        batch = torch.randn(4, 106)
        logits, _ = network(batch)

        # Create action mask: first 10 actions valid, rest invalid
        action_mask = torch.zeros(4, 100, dtype=torch.bool)
        action_mask[:, :10] = True

        # Apply mask
        masked_logits = network.apply_action_mask(logits, action_mask)

        # Invalid actions should be -inf
        assert torch.all(masked_logits[:, 10:] == float('-inf'))

        # Valid actions should be unchanged
        assert torch.allclose(masked_logits[:, :10], logits[:, :10])

    def test_softmax_with_masked_actions_is_zero(self):
        """Softmax of masked actions produces zero probability."""
        network = ActorCriticNetwork(input_size=106, action_size=100, hidden_sizes=[128])

        batch = torch.randn(4, 106)
        logits, _ = network(batch)

        # Mask all but first action
        action_mask = torch.zeros(4, 100, dtype=torch.bool)
        action_mask[:, 0] = True

        masked_logits = network.apply_action_mask(logits, action_mask)
        probs = torch.softmax(masked_logits, dim=-1)

        # Only first action should have non-zero probability
        assert torch.allclose(probs[:, 1:], torch.zeros_like(probs[:, 1:]))
        assert torch.allclose(probs[:, 0], torch.ones(4))


class TestNetworkOutputs:
    """Test network output properties."""

    def test_action_logits_are_floats(self):
        """Action logits are floating point values."""
        network = ActorCriticNetwork(input_size=106, action_size=100, hidden_sizes=[128])

        batch = torch.randn(4, 106)
        logits, _ = network(batch)

        assert logits.dtype in [torch.float32, torch.float64]

    def test_state_value_is_float(self):
        """State value is a floating point scalar."""
        network = ActorCriticNetwork(input_size=106, action_size=100, hidden_sizes=[128])

        batch = torch.randn(4, 106)
        _, value = network(batch)

        assert value.dtype in [torch.float32, torch.float64]

    def test_logits_range_is_reasonable(self):
        """Action logits are in reasonable range (not NaN or inf)."""
        network = ActorCriticNetwork(input_size=106, action_size=100, hidden_sizes=[128])

        batch = torch.randn(4, 106)
        logits, _ = network(batch)

        assert not torch.any(torch.isnan(logits))
        assert not torch.any(torch.isinf(logits))

    def test_value_range_is_reasonable(self):
        """State value is in reasonable range (not NaN or inf)."""
        network = ActorCriticNetwork(input_size=106, action_size=100, hidden_sizes=[128])

        batch = torch.randn(4, 106)
        _, value = network(batch)

        assert not torch.any(torch.isnan(value))
        assert not torch.any(torch.isinf(value))


class TestNetworkInitialization:
    """Test network parameter initialization."""

    def test_network_parameters_are_initialized(self):
        """Network parameters are initialized (not all zeros)."""
        network = ActorCriticNetwork(input_size=106, action_size=100, hidden_sizes=[128])

        # Check that at least some parameters are non-zero
        total_params = sum(p.numel() for p in network.parameters())
        zero_params = sum((p == 0).sum().item() for p in network.parameters())

        # Should have mostly non-zero parameters
        assert zero_params < total_params * 0.5

    def test_network_has_trainable_parameters(self):
        """Network has trainable parameters."""
        network = ActorCriticNetwork(input_size=106, action_size=100, hidden_sizes=[128])

        trainable_params = [p for p in network.parameters() if p.requires_grad]
        assert len(trainable_params) > 0

    def test_network_parameter_count(self):
        """Network has reasonable number of parameters."""
        network = ActorCriticNetwork(input_size=106, action_size=100, hidden_sizes=[128])

        total_params = sum(p.numel() for p in network.parameters())

        # Should have at least input_size * hidden_size parameters
        min_params = 106 * 128
        assert total_params > min_params


class TestGradientFlow:
    """Test gradient backpropagation."""

    def test_gradients_flow_to_all_parameters(self):
        """Gradients flow to all network parameters during backprop."""
        network = ActorCriticNetwork(input_size=106, action_size=100, hidden_sizes=[128])

        batch = torch.randn(4, 106)
        logits, value = network(batch)

        # Create dummy loss
        loss = logits.sum() + value.sum()
        loss.backward()

        # Check all parameters have gradients
        for name, param in network.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"

    def test_actor_head_gradients(self):
        """Actor head receives gradients from policy loss."""
        network = ActorCriticNetwork(input_size=106, action_size=100, hidden_sizes=[128])

        batch = torch.randn(4, 106)
        logits, _ = network(batch)

        # Policy loss (only affects actor head)
        loss = logits.mean()
        loss.backward()

        # Actor head should have gradients
        for param in network.actor_head.parameters():
            if param.requires_grad:
                assert param.grad is not None

    def test_critic_head_gradients(self):
        """Critic head receives gradients from value loss."""
        network = ActorCriticNetwork(input_size=106, action_size=100, hidden_sizes=[128])

        batch = torch.randn(4, 106)
        _, value = network(batch)

        # Value loss (only affects critic head)
        loss = value.mean()
        loss.backward()

        # Critic head should have gradients
        for param in network.critic_head.parameters():
            if param.requires_grad:
                assert param.grad is not None


class TestNetworkDevice:
    """Test network device placement (CPU/CUDA)."""

    def test_network_defaults_to_cpu(self):
        """Network is created on CPU by default."""
        network = ActorCriticNetwork(input_size=106, action_size=100, hidden_sizes=[128])

        # Check first parameter's device
        first_param = next(network.parameters())
        assert first_param.device.type == 'cpu'

    def test_network_can_move_to_device(self):
        """Network can be moved to different device."""
        network = ActorCriticNetwork(input_size=106, action_size=100, hidden_sizes=[128])

        # Try to move to CPU (should always work)
        network = network.to('cpu')
        first_param = next(network.parameters())
        assert first_param.device.type == 'cpu'

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_network_can_move_to_cuda(self):
        """Network can be moved to CUDA if available."""
        network = ActorCriticNetwork(input_size=106, action_size=100, hidden_sizes=[128])

        network = network.to('cuda')
        first_param = next(network.parameters())
        assert first_param.device.type == 'cuda'


class TestCreateActorCriticNetwork:
    """Test convenience factory function."""

    def test_create_network_from_feature_extractor(self):
        """Can create network from FeatureExtractor."""
        extractor = FeatureExtractor()
        network = create_actor_critic_network(
            feature_size=extractor.feature_size,
            action_size=100
        )

        assert network is not None

        # Test forward pass
        batch = torch.randn(4, extractor.feature_size)
        logits, value = network(batch)
        assert logits.shape == (4, 100)
        assert value.shape == (4, 1)

    def test_create_network_with_custom_hidden_sizes(self):
        """Factory function accepts custom hidden sizes."""
        network = create_actor_critic_network(
            feature_size=106,
            action_size=100,
            hidden_sizes=[256, 128, 64]
        )

        assert network is not None


class TestNetworkDeterminism:
    """Test network determinism with fixed seed."""

    def test_same_seed_produces_same_initialization(self):
        """Same random seed produces same initial weights."""
        torch.manual_seed(42)
        net1 = ActorCriticNetwork(input_size=106, action_size=100, hidden_sizes=[128])

        torch.manual_seed(42)
        net2 = ActorCriticNetwork(input_size=106, action_size=100, hidden_sizes=[128])

        # Compare first layer weights
        for p1, p2 in zip(net1.parameters(), net2.parameters()):
            assert torch.allclose(p1, p2)

    def test_same_input_produces_same_output(self):
        """Same input produces same output (deterministic forward pass)."""
        network = ActorCriticNetwork(input_size=106, action_size=100, hidden_sizes=[128])
        network.eval()  # Disable dropout if any

        input_tensor = torch.randn(4, 106)

        logits1, value1 = network(input_tensor)
        logits2, value2 = network(input_tensor)

        assert torch.allclose(logits1, logits2)
        assert torch.allclose(value1, value2)
