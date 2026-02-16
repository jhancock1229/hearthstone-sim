# Phase 5 — RL Environment & Training Baseline

This document covers everything built in Phase 5: the agent hierarchy, RL infrastructure, and how to use it all.

## What Was Built

Phase 5 added **4 AI agents**, a **Gymnasium RL environment**, a **PPO training pipeline**, and supporting infrastructure. Starting from 483 tests (end of Phase 4), we ended at **653 tests** — 170 new tests covering every component.

### Component Overview

```
agents/
  base.py               # Abstract Agent interface (Phase 3)
  random_agent.py       # Random baseline (Phase 3)
  greedy_agent.py       # Heuristic agent — NEW
  mcts_agent.py         # Monte Carlo Tree Search — NEW
  rl/
    features.py         # Observation → 106-dim feature vector — NEW
    networks.py         # Actor-Critic neural network — NEW
    env.py              # Gymnasium environment wrapper — NEW
    ppo_agent.py        # PPO agent with training — NEW
training/
  config.py             # TrainingConfig dataclass — NEW
  trainer.py            # Training loop orchestrator — NEW
analysis/
  metrics.py            # EpisodeMetrics, MetricsTracker — NEW
scripts/
  train.py              # CLI: launch training — NEW
  evaluate.py           # CLI: evaluate agents — NEW
  simulate.py           # CLI: run simulations (updated)
```

---

## Agents

### 1. RandomAgent (baseline)

Picks a uniformly random legal action each turn. Used as the floor for measuring agent quality.

```python
from agents.random_agent import RandomAgent
agent = RandomAgent(seed=42)
```

### 2. GreedyAgent (heuristic)

Uses hand-crafted heuristics to evaluate actions:
- **Card play priority (100)**: Prefers high stats-per-mana, large minions
- **Attack priority (50)**: Prefers favorable trades (kill without dying)
- **Hero power (20)**: Used when nothing better is available
- **End turn (0)**: Last resort

Win rate: ~70-76% vs Random.

```python
from agents.greedy_agent import GreedyAgent
agent = GreedyAgent()
```

### 3. MCTSAgent (search)

Monte Carlo Tree Search with UCB1 selection. Builds a game tree by:
1. **Selection**: Walk tree using UCB1 (exploration vs exploitation)
2. **Expansion**: Add new node for unexplored action
3. **Simulation**: Random rollout to estimate value
4. **Backpropagation**: Update visit counts and rewards up the tree

```python
from agents.mcts_agent import MCTSAgent
agent = MCTSAgent(num_simulations=100, exploration_constant=1.41)
```

### 4. PPOAgent (learned)

Proximal Policy Optimization agent with:
- **Actor-Critic network**: Shared backbone → policy logits + value estimate
- **Action masking**: Only legal actions can be selected
- **GAE**: Generalized Advantage Estimation for stable training
- **Clipped objective**: Prevents destructive policy updates

```python
from agents.rl.ppo_agent import PPOAgent

# For evaluation (inference only)
agent = PPOAgent()
agent.load("checkpoints/my_model.pt")
action = agent.choose_action(observation)

# For training
action_idx, value, log_prob = agent.get_action_and_value(obs_features, action_mask)
agent.rollout_buffer.add(obs, action_idx, reward, value, log_prob, done, mask)
losses = agent.update()
```

---

## RL Infrastructure

### Feature Extraction (`agents/rl/features.py`)

Converts rich `Observation` objects into a fixed-size 106-dimensional numpy vector:

| Feature Group | Count | Description |
|---------------|-------|-------------|
| Scalar | 10 | Health, mana, deck/hand sizes, fatigue (both players) |
| Self Board | 42 | 7 slots x 6 features (present, attack, health, taunt, divine_shield, poisonous) |
| Opponent Board | 42 | Same encoding as self board |
| Hand Histogram | 10 | Card count by mana cost (0-9) |
| Misc | 2 | Turn number, player index |

All features normalized to [0, 1].

```python
from agents.rl.features import FeatureExtractor, extract_features

extractor = FeatureExtractor()
features = extractor.extract(observation)  # shape: (106,), dtype: float32
```

### Neural Network (`agents/rl/networks.py`)

Actor-Critic architecture with shared backbone:

```
Input (106) → [Linear(128) → ReLU → Linear(64) → ReLU] → Actor Head (100 logits)
                                                         → Critic Head (1 value)
```

- **Actor head**: Outputs logits for 100 discrete actions
- **Critic head**: Outputs scalar state value estimate
- **Action masking**: Sets illegal action logits to -inf before softmax

```python
from agents.rl.networks import ActorCriticNetwork

network = ActorCriticNetwork(input_size=106, action_size=100, hidden_sizes=[128, 64])
logits, value = network(feature_tensor)
masked = network.apply_action_mask(logits, mask_tensor)
```

### Gymnasium Environment (`agents/rl/env.py`)

Standard Gymnasium interface wrapping the Hearthstone engine:

- **Observation space**: `Box(106,)` — feature vector
- **Action space**: `Discrete(100)` with action masking via `info['action_mask']`
- **Rewards**: Sparse (+1 win, -1 loss, 0 mid-game)
- **Agent plays as Player 1**, opponent is configurable (default: random)

**Action encoding (100 slots)**:

| Range | Count | Action |
|-------|-------|--------|
| 0-69 | 70 | PLAY_CARD: `hand_index * 7 + board_position` |
| 70-97 | 28 | ATTACK: `70 + attacker * 4 + target` (0=face, 1-3=minion) |
| 98 | 1 | HERO_POWER |
| 99 | 1 | END_TURN |

```python
from agents.rl.env import HearthstoneEnv

env = HearthstoneEnv(seed=42, max_turns=200)
obs, info = env.reset()
mask = info['action_mask']  # bool array, shape (100,)

obs, reward, terminated, truncated, info = env.step(action_index)
```

### Rollout Buffer & GAE (`agents/rl/ppo_agent.py`)

The `RolloutBuffer` collects (observation, action, reward, value, log_prob, done, mask) tuples during gameplay, then converts them to batched tensors with computed advantages:

```python
buffer = RolloutBuffer()
buffer.add(obs, action, reward, value, log_prob, done, mask)
batch = buffer.get_batch(gamma=0.99, gae_lambda=0.95)
# batch['observations'], batch['actions'], batch['advantages'], ...
```

GAE computes advantages by combining TD errors with exponential decay:

```
A_t = delta_t + (gamma * lambda) * delta_{t+1} + (gamma * lambda)^2 * delta_{t+2} + ...
delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
```

---

## Training

### Quick Start

```bash
# Install dependencies
pip install -e .[ml,dev]

# Train PPO vs random opponent (1000 episodes)
python scripts/train.py --episodes 1000 --opponent random

# Train with custom hyperparameters
python scripts/train.py --episodes 5000 --lr 1e-4 --entropy-coef 0.02 --opponent greedy

# Save checkpoints every 500 episodes
python scripts/train.py --episodes 5000 --save-interval 500 --checkpoint-dir my_checkpoints
```

### Programmatic Training

```python
from training.config import TrainingConfig
from training.trainer import Trainer

config = TrainingConfig(
    total_episodes=2000,
    learning_rate=3e-4,
    gamma=0.99,
    clip_epsilon=0.2,
    opponent="random",
    eval_interval=100,
    eval_games=50,
)

trainer = Trainer(config)
result = trainer.train()

print(f"Final win rate: {result['final_win_rate']:.1%}")
```

### Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `learning_rate` | 3e-4 | Adam optimizer learning rate |
| `gamma` | 0.99 | Discount factor |
| `gae_lambda` | 0.95 | GAE lambda (1.0 = Monte Carlo, 0.0 = TD) |
| `clip_epsilon` | 0.2 | PPO clipping range |
| `value_loss_coef` | 0.5 | Weight for critic loss |
| `entropy_coef` | 0.01 | Entropy bonus (exploration) |
| `ppo_epochs` | 4 | Passes over rollout data per update |
| `max_grad_norm` | 0.5 | Gradient clipping threshold |
| `hidden_sizes` | [128, 64] | Network hidden layer sizes |

---

## Evaluation

### CLI

```bash
# Compare any two agents
python scripts/evaluate.py --agent1 greedy --agent2 random --games 200

# Evaluate trained PPO model
python scripts/evaluate.py --agent1 ppo --model1 checkpoints/my_model.pt --agent2 greedy --games 100

# MCTS vs Greedy
python scripts/evaluate.py --agent1 mcts --agent2 greedy --games 50
```

### Programmatic

```python
from simulation.simulator import Simulator
from agents.greedy_agent import GreedyAgent
from agents.random_agent import RandomAgent

sim = Simulator(seed=42, max_turns=200)
result = sim.run_games(GreedyAgent(), RandomAgent(seed=1), num_games=100)

print(f"Greedy win rate: {result.player1_win_rate:.1%}")
print(f"Avg game length: {result.average_game_length:.1f} turns")
```

### Metrics Tracking

```python
from analysis.metrics import MetricsTracker

tracker = MetricsTracker()
tracker.record_episode(reward=1.0, length=45, losses={'policy_loss': 0.01, 'value_loss': 0.5})
tracker.record_episode(reward=-1.0, length=30, losses={'policy_loss': 0.02, 'value_loss': 0.3})

print(tracker.format_summary())
# Episodes: 2 | Win Rate: 50.0% | Reward: 0.000 | Length: 37.5 | ...
```

---

## Agent Comparison

Approximate win rates from simulation testing:

| Matchup | Player 1 Win Rate |
|---------|-------------------|
| Greedy vs Random | ~70-76% |
| MCTS vs Random | ~50% (simplified rollout) |
| Random vs Random | ~50% |
| PPO (untrained) vs Random | ~40-50% |

The Greedy agent is the strongest baseline out of the box. MCTS performs around random level because it uses a simplified health-heuristic rollout rather than full game simulation. PPO starts weak but can improve with training.

---

## Architecture Decisions

1. **Fixed 100-action space**: Rather than variable-size action lists, we use a fixed Discrete(100) space with action masking. This is standard for RL and avoids the complexity of variable-size output layers.

2. **Sparse rewards**: Only +1/-1 at game end. Dense rewards (e.g., damage dealt, cards played) were considered but risk reward hacking. Sparse rewards force the agent to learn which intermediate actions lead to wins.

3. **Shared backbone**: The Actor-Critic network shares feature extraction layers. This improves sample efficiency since both policy and value function benefit from the same learned representations.

4. **Feature normalization to [0,1]**: All features are clipped/normalized so the network sees consistent input scales. This prevents features with large raw values (like health=30) from dominating smaller ones (like boolean mechanics).

5. **Agent as base class**: All agents (Random, Greedy, MCTS, PPO) share the same `Agent` interface with `choose_action(observation) -> Action`. This means any agent can be swapped into the Simulator or used as an opponent.

---

## File Test Coverage

| Component | Tests | File |
|-----------|-------|------|
| GreedyAgent | 16 | `tests/agents/test_greedy_agent.py` |
| MCTSAgent | 29 | `tests/agents/test_mcts_agent.py` |
| FeatureExtractor | 21 | `tests/agents/rl/test_features.py` |
| ActorCriticNetwork | 28 | `tests/agents/rl/test_networks.py` |
| HearthstoneEnv | 29 | `tests/agents/rl/test_env.py` |
| PPOAgent + RolloutBuffer + GAE | 38 | `tests/agents/rl/test_ppo_agent.py` |
| **Phase 5 Total** | **161** | |
| **Project Total** | **653** | |
