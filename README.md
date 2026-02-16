# Hearthstone Simulator

A test-driven Hearthstone game simulator built in Python for AI research, deck optimization, and reinforcement learning experimentation.

**1014 tests** | Python 3.12+ | PyTorch + Gymnasium

---

## Installation

```bash
# Clone and enter the project
cd hearthstone-sim

# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate

# Install all dependencies
pip install -e ".[ml,dev,analysis]"
```

**Dependency groups:**
- `ml` — PyTorch, Gymnasium, NumPy (required for RL training and simulation)
- `dev` — pytest, pytest-cov, ruff (required for tests and linting)
- `analysis` — matplotlib, pandas (optional, for visualization and export)

Verify everything works:

```bash
make test
```

---

## Quick Start

### Run a simulation between two agents

```bash
# 100 games: greedy vs random
python3 scripts/simulate.py --p1 greedy --p2 random --games 100

# 1000 games: MCTS vs greedy, save results
python3 scripts/simulate.py --p1 mcts --p2 greedy --games 1000 --output results.json
```

### Train an RL agent

```bash
# Train PPO agent for 5000 episodes against greedy opponent
python3 scripts/train.py --episodes 5000 --opponent greedy --lr 1e-4

# Train against random, evaluate every 50 episodes
python3 scripts/train.py --episodes 2000 --opponent random --eval-interval 50
```

### Evaluate agents head-to-head

```bash
# Compare a trained PPO model against MCTS
python3 scripts/evaluate.py --agent1 ppo --model1 checkpoints/model.pt --agent2 mcts --games 500
```

### Play interactively

```bash
python3 scripts/play.py --opponent random
```

---

## User Guide

### 1. Understanding the Card Pool

The simulator ships with **7,662 real Hearthstone cards** loaded from JSON. The game engine fully supports minion combat with these mechanics: TAUNT, DIVINE_SHIELD, LIFESTEAL, CHARGE, RUSH, POISONOUS, and WINDFURY. Spells, weapons, and hero cards are parsed but their effects do not execute in the current engine.

```python
from pathlib import Path
from hearthstone.cards.registry import CardRegistry
from hearthstone.enums import CardType, CardClass

registry = CardRegistry.from_json(Path("hearthstone/data/cards_collectible.json"))

# Look up a specific card
card = registry.get_by_id("CS2_124")
print(card.name, card.attack, card.health)

# Filter cards
mage_minions = registry.filter(card_class=CardClass.MAGE, card_type=CardType.MINION)
taunt_cards = registry.filter(mechanic="TAUNT")
```

### 2. Running Simulations Programmatically

```python
from simulation.simulator import Simulator
from agents.random_agent import RandomAgent
from agents.greedy_agent import GreedyAgent
from agents.mcts_agent import MCTSAgent

sim = Simulator(seed=42, max_turns=200)

# Random vs Greedy
result = sim.run_games(RandomAgent(seed=1), GreedyAgent(), num_games=100)
print(f"Greedy win rate: {result.player2_win_rate:.1%}")
print(f"Avg game length: {result.average_game_length:.1f} turns")

# With custom decks (list of card pool keys)
from deckbuilding.deck import CARD_POOL
deck_a = ["Brute"] * 15 + ["Wasp"] * 15
deck_b = ["Giant"] * 10 + ["Shield"] * 20
result = sim.run_games(
    GreedyAgent(), GreedyAgent(), num_games=50,
    deck1_genotype=deck_a, deck2_genotype=deck_b,
)
```

**Available agents:**

| Agent | Description | Speed |
|-------|-------------|-------|
| `RandomAgent` | Picks random legal actions | Fastest |
| `GreedyAgent` | Evaluates each action with a heuristic, picks the best | Fast |
| `MCTSAgent` | Monte Carlo Tree Search with UCB1 selection | Slow (but strongest) |
| `PPOAgent` | Trained RL policy network | Fast (after training) |

### 3. Building and Evaluating Decks

#### Heuristic evaluation

```python
from deckbuilding.deck import Deck
from deckbuilding.evaluator import HeuristicEvaluator, DeckEvaluator
from hearthstone.cards.base import MinionCard

# Build a deck of minion cards
cards = [MinionCard(name=f"M{i}", mana_cost=i % 7 + 1, attack=i % 5 + 1, health=i % 5 + 1)
         for i in range(30)]
deck = Deck(cards=cards, name="Test Deck")

evaluator = HeuristicEvaluator()
result = evaluator.evaluate(deck)
print(f"Score: {result.score:.3f}")
print(f"Mana curve: {result.metrics['mana_curve_score']:.2f}")
print(f"Card quality: {result.metrics['card_quality_score']:.2f}")
```

#### Simulation-based evaluation (real games)

```python
from deckbuilding.evaluator import SimulationEvaluator

evaluator = SimulationEvaluator(num_games=50, seed=42)
genotype = ["Brute"] * 15 + ["Wasp"] * 15
result = evaluator.evaluate_genotype(genotype)
print(f"Win rate: {result.score:.1%}")
print(f"Avg game length: {result.metrics['avg_game_length']:.1f}")
```

### 4. Genetic Algorithm Deck Optimization

The basic GA evolves decks using a simplified match function:

```python
from deckbuilding.genetic import run_ga

# Evolve a population of 30 decks over 10 generations
results = run_ga(pop_size=30, generations=10, games_per_eval=10)
best_genotype, best_score = results[0]
print(f"Best deck score: {best_score:.3f}")
print(f"Best deck: {best_genotype}")
```

### 5. Co-Evolutionary Deck Optimization (Recommended)

The co-evolutionary system evolves decks **against each other** using real game simulations with GreedyAgent. This produces decks that are strong against a diverse field, not just one opponent.

```python
from pathlib import Path
from hearthstone.cards.registry import CardRegistry
from deckbuilding.deck import build_pool_from_registry
from deckbuilding.coevolution import CoevolutionConfig, CoevolutionEngine

# Load real card pool (minions with supported mechanics only)
registry = CardRegistry.from_json(Path("hearthstone/data/cards_collectible.json"))
pool = build_pool_from_registry(registry)
print(f"Card pool: {len(pool)} minions")

# Configure and run
config = CoevolutionConfig(
    pool=pool,
    population_size=50,     # 50 decks in the population
    generations=20,         # 20 evolutionary generations
    opponents_per_eval=7,   # each deck plays 7 random opponents per generation
    games_per_matchup=5,    # 5 games per deck pair
    mutation_rate=0.05,     # 5% per-card mutation rate
    elite_fraction=0.1,     # top 10% survive unchanged
    seed=42,
)

engine = CoevolutionEngine(config)
results = engine.run()

# Best deck
best_genotype, best_fitness = results[0]
print(f"\nBest fitness: {best_fitness:.3f}")

# Analyze meta-game
for entry in engine.meta_tracker.get_meta_summary():
    print(f"  {entry['name']}: {entry['win_rate']:.1%} win rate, "
          f"{entry['popularity']:.1%} popularity")
```

**How it works:**
1. A population of random decks is generated from the card pool
2. Each generation, every deck plays against K randomly sampled opponents
3. Games are played using GreedyAgent (both sides play intelligently)
4. Fitness = average win rate across opponents
5. Top performers are kept as elites; the rest undergo crossover and mutation
6. Archetype labels (Aggro/Midrange/Control/Tempo) are assigned by mana curve
7. All results feed into MetaTracker (archetype stats) and MatchupMatrix (per-deck stats)

**Performance estimate:** pop=50, K=7, 5 games/matchup = ~1,750 games/generation. With GreedyAgent at ~0.1s/game, expect ~3 min/generation, ~60 min for 20 generations.

You can use the smaller synthetic pool for faster iteration:

```python
from deckbuilding.deck import CARD_POOL
config = CoevolutionConfig(pool=CARD_POOL, population_size=20, generations=10, seed=42)
engine = CoevolutionEngine(config)
results = engine.run()  # Runs in seconds
```

### 6. Meta-Game Analysis

After running co-evolution (or any set of matches), you can analyze the meta:

```python
# Archetype win rates and matchups
tracker = engine.meta_tracker
print(tracker.get_meta_summary())
print(tracker.get_counters("Aggro"))        # What beats Aggro?
print(tracker.get_win_rate("Control"))       # Control's overall win rate

# Per-deck matchup matrix
matrix = engine.matchup_matrix
print(matrix.get_win_rate("g5_d0", "g5_d3"))  # Specific matchup
print(matrix.get_best_matchup("g5_d0"))        # Best matchup for deck 0
```

### 7. Training a PPO Agent

```python
from training.config import TrainingConfig
from training.trainer import Trainer

config = TrainingConfig(
    total_episodes=5000,
    opponent="greedy",
    learning_rate=3e-4,
    eval_interval=100,
    eval_games=50,
    save_interval=500,
    seed=42,
)

trainer = Trainer(config, progress_callback=lambda info: (
    print(f"Ep {info['episode']}: reward={info['reward']:.2f}")
    if info['episode'] % 100 == 0 else None
))
result = trainer.train()

print(f"Final win rate: {result['final_win_rate']:.1%}")
print(f"Mean reward: {result['mean_reward']:.3f}")
```

#### Curriculum training (staged difficulty)

```python
from training.curriculum import CurriculumManager

curriculum = CurriculumManager.create_default()
# Stage 1: Random opponents (easy)
# Stage 2: Greedy opponents (medium)
# Stage 3: Self-play (hard)
# Stage 4: Mixed opponents (final)

print(curriculum.get_status())
# Advance when win rate thresholds are met
```

### 8. Exporting Results

```python
from analysis.metrics import MetricsTracker
from analysis.export import (
    export_training_csv,
    export_training_json,
    export_matchups_csv,
    export_meta_json,
)

# After training
export_training_csv(tracker, "training_results.csv")
export_training_json(tracker, "training_results.json")

# After co-evolution
export_matchups_csv(engine.matchup_matrix, "matchups.csv")
export_meta_json(engine.meta_tracker, "meta_summary.json")
```

### 9. Visualization Data

The visualize module generates chart-ready data structures (dicts of lists). Rendering with matplotlib is optional — you can also use plotly, export to CSV, or feed into any charting tool.

#### Training curves

```python
from analysis.visualize import training_curve_data

data = training_curve_data(tracker, window=50)
# data keys: episodes, rewards, lengths, policy_losses, value_losses,
#            entropies, rolling_win_rates, win_rates, eval_win_rates
```

Plot with matplotlib:

```python
import matplotlib.pyplot as plt

data = training_curve_data(tracker, window=50)

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

axes[0, 0].plot(data["episodes"], data["rolling_win_rates"])
axes[0, 0].set_title("Rolling Win Rate")
axes[0, 0].set_ylabel("Win Rate")

axes[0, 1].plot(data["episodes"], data["rewards"], alpha=0.3)
axes[0, 1].set_title("Episode Rewards")

axes[1, 0].plot(data["episodes"], data["policy_losses"])
axes[1, 0].set_title("Policy Loss")
axes[1, 0].set_xlabel("Episode")

axes[1, 1].plot(data["episodes"], data["lengths"], alpha=0.3)
axes[1, 1].set_title("Episode Length")
axes[1, 1].set_xlabel("Episode")

plt.tight_layout()
plt.savefig("training_curves.png")
plt.show()
```

#### Matchup heatmap

```python
from analysis.visualize import matchup_heatmap_data
import matplotlib.pyplot as plt
import numpy as np

heatmap = matchup_heatmap_data(engine.matchup_matrix)
# heatmap keys: labels, values (2D list, diagonal=0.5, None for missing)

labels = heatmap["labels"]
grid = np.array([[v if v is not None else 0.5 for v in row]
                 for row in heatmap["values"]])

fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(grid, cmap="RdYlGn", vmin=0, vmax=1)
ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right", fontsize=7)
ax.set_yticks(range(len(labels)), labels, fontsize=7)
ax.set_title("Matchup Win Rates (row vs column)")
plt.colorbar(im)
plt.tight_layout()
plt.savefig("matchup_heatmap.png")
plt.show()
```

#### Elo progression

```python
from analysis.visualize import elo_progression_data
import matplotlib.pyplot as plt

data = elo_progression_data(elo_tracker)
# data keys: matches (list of dicts), agents (dict of agent_id -> rating history)

for agent_id, ratings in data["agents"].items():
    plt.plot(ratings, label=agent_id)
plt.xlabel("Match")
plt.ylabel("Elo Rating")
plt.title("Elo Rating Progression")
plt.legend()
plt.savefig("elo_progression.png")
plt.show()
```

#### Action distribution

```python
from analysis.visualize import action_distribution_data
import matplotlib.pyplot as plt

# Pass a list of action type strings collected during gameplay
actions = ["PLAY_CARD", "ATTACK", "ATTACK", "END_TURN", "PLAY_CARD"]
data = action_distribution_data(actions, proportions=True)
# data keys: each action type -> proportion, plus "total"

action_types = [k for k in data if k != "total"]
values = [data[k] for k in action_types]

plt.bar(action_types, values)
plt.ylabel("Proportion")
plt.title(f"Action Distribution ({data['total']} actions)")
plt.savefig("action_distribution.png")
plt.show()
```

#### Quick visualization from CLI

Train and visualize in one shot:

```bash
# Train, then export metrics to CSV for external plotting
python3 -c "
from training.config import TrainingConfig
from training.trainer import Trainer
from analysis.metrics import MetricsTracker
from analysis.export import export_training_csv

config = TrainingConfig(total_episodes=500, opponent='greedy',
                        eval_interval=50, eval_games=10, save_interval=0, seed=42)
trainer = Trainer(config)
result = trainer.train()

tracker = MetricsTracker()
for r, l in zip(trainer.episode_rewards, trainer.episode_lengths):
    tracker.record_episode(reward=r, length=l)

export_training_csv(tracker, 'training_results.csv')
print(tracker.format_summary())
print('Exported to training_results.csv')
"
```

Run co-evolution and export meta-game data:

```bash
python3 -c "
from deckbuilding.deck import CARD_POOL
from deckbuilding.coevolution import CoevolutionConfig, CoevolutionEngine
from analysis.export import export_meta_json, export_matchups_csv

config = CoevolutionConfig(pool=CARD_POOL, population_size=20, generations=5, seed=42)
engine = CoevolutionEngine(config)
results = engine.run()

best, score = results[0]
print(f'Best deck fitness: {score:.3f}')
print(f'Best deck: {best}')
for entry in engine.meta_tracker.get_meta_summary():
    print(f'  {entry[\"name\"]}: {entry[\"win_rate\"]:.1%} win rate')

export_meta_json(engine.meta_tracker, 'meta_summary.json')
export_matchups_csv(engine.matchup_matrix, 'matchups.csv')
print('Exported meta_summary.json and matchups.csv')
"
```

Plot training curves from an exported CSV:

```bash
python3 -c "
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('training_results.csv')
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Rolling win rate (window=50)
df['win'] = (df['reward'] > 0).astype(float)
df['rolling_wr'] = df['win'].rolling(50, min_periods=1).mean()
axes[0].plot(df.index, df['rolling_wr'])
axes[0].set_title('Rolling Win Rate (window=50)')
axes[0].set_xlabel('Episode')
axes[0].set_ylabel('Win Rate')

# Reward over time
axes[1].plot(df.index, df['reward'], alpha=0.3)
axes[1].set_title('Episode Reward')
axes[1].set_xlabel('Episode')

plt.tight_layout()
plt.savefig('training_plots.png', dpi=150)
plt.show()
print('Saved training_plots.png')
"
```

---

## CLI Reference

### `scripts/simulate.py` — Run agent matchups

```bash
python3 scripts/simulate.py --p1 greedy --p2 mcts --games 1000 --seed 42 --verbose
```

| Flag | Default | Description |
|------|---------|-------------|
| `--p1` | `random` | Player 1 agent: `random`, `greedy`, `mcts` |
| `--p2` | `random` | Player 2 agent: `random`, `greedy`, `mcts` |
| `--games` | `100` | Number of games |
| `--seed` | — | Random seed |
| `--max-turns` | `200` | Max turns per game |
| `--verbose` | — | Print progress |
| `--output` | — | Save results to JSON file |

### `scripts/train.py` — Train PPO agent

```bash
python3 scripts/train.py --episodes 5000 --opponent greedy --lr 1e-4 --name my_experiment
```

| Flag | Default | Description |
|------|---------|-------------|
| `--episodes` | `1000` | Total training episodes |
| `--opponent` | `random` | Opponent type: `random`, `greedy`, `self` |
| `--lr` | `3e-4` | Learning rate |
| `--gamma` | `0.99` | Discount factor |
| `--clip-epsilon` | `0.2` | PPO clipping range |
| `--ppo-epochs` | `4` | PPO epochs per update |
| `--entropy-coef` | `0.01` | Entropy bonus weight |
| `--eval-interval` | `100` | Evaluate every N episodes |
| `--eval-games` | `50` | Games per evaluation |
| `--save-interval` | `500` | Checkpoint save interval |
| `--max-turns` | `200` | Max turns per game |
| `--seed` | `42` | Random seed |
| `--name` | `ppo_run` | Experiment name |
| `--checkpoint-dir` | `checkpoints` | Checkpoint directory |

### `scripts/evaluate.py` — Evaluate agents

```bash
python3 scripts/evaluate.py --agent1 ppo --model1 checkpoints/model.pt --agent2 greedy --games 500
```

| Flag | Default | Description |
|------|---------|-------------|
| `--agent1` | `random` | Agent 1: `random`, `greedy`, `mcts`, `ppo` |
| `--agent2` | `random` | Agent 2: `random`, `greedy`, `mcts`, `ppo` |
| `--model1` | — | Model checkpoint path (PPO only) |
| `--model2` | — | Model checkpoint path (PPO only) |
| `--games` | `100` | Number of games |
| `--seed` | `42` | Random seed |
| `--max-turns` | `200` | Max turns per game |
| `--verbose` | — | Print progress |

### `scripts/play.py` — Interactive play

```bash
python3 scripts/play.py --opponent random --seed 42
```

### `scripts/import_cards.py` — Import card data

```bash
python3 scripts/import_cards.py --collectible-only --heroes
```

---

## Makefile Targets

```bash
make test          # Run full test suite
make test-cov      # Run tests with coverage
make lint          # Run ruff linter
make clean         # Remove __pycache__ and .pytest_cache
make sim           # 100 games, seed 42
make sim-1k        # 1000 games
make sim-10k       # 10000 games with verbose output
make sim-save      # 1000 games, save results to JSON
```

---

## Project Structure

```
hearthstone-sim/
├── hearthstone/              # Core game engine
│   ├── engine/               # Game loop, player, combat, actions, events
│   ├── cards/                # Card dataclasses, effects, registry
│   └── data/                 # 7662 real card definitions (JSON)
├── simulation/               # Game simulation layer
│   ├── simulator.py          # N-game rollout engine (custom deck support)
│   ├── observation.py        # Agent-facing game view
│   ├── action_space.py       # Legal action enumeration
│   ├── game_state.py         # Immutable game snapshots
│   └── replay.py             # Deterministic replay system
├── agents/                   # AI agents
│   ├── random_agent.py       # Random baseline
│   ├── greedy_agent.py       # Heuristic greedy agent
│   ├── mcts_agent.py         # Monte Carlo Tree Search
│   └── rl/                   # Reinforcement learning
│       ├── env.py            # Gymnasium environment wrapper
│       ├── features.py       # 106-dim feature extraction
│       ├── networks.py       # Actor-Critic neural network
│       ├── ppo_agent.py      # PPO-Clip agent with GAE
│       └── self_play.py      # Elo tracking + opponent pool
├── deckbuilding/             # Deck construction and optimization
│   ├── deck.py               # Deck class, card pool, registry bridge
│   ├── constraints.py        # Deck validation rules
│   ├── evaluator.py          # Heuristic + simulation evaluation
│   ├── genetic.py            # Basic genetic algorithm
│   ├── coevolution.py        # Co-evolutionary optimization engine
│   └── meta.py               # Archetype + meta-game tracking
├── training/                 # Training infrastructure
│   ├── config.py             # TrainingConfig dataclass
│   ├── trainer.py            # PPO training loop
│   ├── curriculum.py         # Staged difficulty progression
│   └── distributed.py        # Multi-worker training
├── analysis/                 # Results analysis
│   ├── metrics.py            # Episode metrics + win rate tracking
│   ├── matchups.py           # NxN matchup matrix
│   ├── visualize.py          # Chart-ready data generation
│   └── export.py             # CSV/JSON export
├── scripts/                  # CLI entry points
│   ├── simulate.py           # Run agent matchups
│   ├── train.py              # Launch PPO training
│   ├── evaluate.py           # Agent comparison
│   ├── play.py               # Interactive human vs agent
│   └── import_cards.py       # Card data import
└── tests/                    # 1014 tests across all modules
```

---

## Development

See [MANIFEST.md](MANIFEST.md) for detailed per-file status tracking and development progress across all 9 phases.
