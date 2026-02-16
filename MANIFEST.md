# File Manifest

Tracks every file in the project, its current status, and development phase.

**Project Status:** 1102 passing tests | Phases 1-9 ✅ Complete | Roadmap Items 1-4 ✅ Complete

**Status legend:**
- `ACTIVE` — Contains real, tested code
- `PLACEHOLDER` — Empty/stub file awaiting implementation
- `CONFIG` — Configuration file
- `FIXTURE` — Test infrastructure

---

## Phase 1 — Core Engine (mana, health, draw) ✅ COMPLETE

| File | Status | Notes |
|------|--------|-------|
| `hearthstone/engine/game.py` | ACTIVE | Game init with two players, hero class params |
| `hearthstone/engine/player.py` | ACTIVE | Player state with combat methods, hero_class, armor, take_damage() |
| `hearthstone/cards/base.py` | ACTIVE | Card and MinionCard dataclasses |
| `hearthstone/enums.py` | ACTIVE | CardType, Rarity, CardClass enums implemented |
| `hearthstone/exceptions.py` | ACTIVE | IllegalActionError, GameOverError |
| `tests/engine/test_core_mechanics.py` | ACTIVE | 10 tests: init, mana, draw/fatigue |
| `tests/conftest.py` | ACTIVE | Shared fixtures: player, wisp |

## Phase 2 — Combat, Board, Keywords, Events ✅ COMPLETE

| File | Status | Notes |
|------|--------|-------|
| `hearthstone/engine/board.py` | ACTIVE | Board state, 7-slot limit |
| `hearthstone/engine/combat.py` | ACTIVE | Attack resolution, damage, death, keywords |
| `hearthstone/engine/events.py` | ACTIVE | Event system for triggers |
| `hearthstone/cards/effects.py` | ACTIVE | Battlecry, Deathrattle, Aura |
| `hearthstone/cards/keywords.py` | PLACEHOLDER | Keywords implemented in combat.py/base.py instead |
| `hearthstone/zones.py` | ACTIVE | Hand, Board, Deck, Graveyard, Secrets zones |
| `tests/cards/test_base.py` | ACTIVE | 11 tests: card creation, Divine Shield property |
| `tests/cards/test_effects.py` | ACTIVE | 5 tests: Battlecry, Deathrattle registries |
| `tests/cards/test_keywords.py` | ACTIVE | 30 tests: all keyword mechanics (169 total tests in Phase 1+2) |
| `hearthstone/cards/battlecries.py` | ACTIVE | Battlecry text parser: 6 patterns (deal damage, draw, heal, armor, summon, buff) |
| `tests/cards/test_battlecries.py` | ACTIVE | 29 tests: text parser, effect resolver, play_card integration |

## Phase 3 — Action Space, Simulation Interface, Basic Agents ✅ COMPLETE

| File | Status | Notes |
|------|--------|-------|
| `hearthstone/engine/actions.py` | ACTIVE | Action execution fully implemented |
| `tests/engine/test_actions.py` | ACTIVE | 31 tests for PlayCard, Attack, HeroPower, EndTurn |
| `tests/engine/test_hero_powers.py` | ACTIVE | 35 tests: all 11 hero powers, player attributes, action space, simulator |
| `simulation/game_state.py` | ACTIVE | Immutable game snapshot (416 total tests in Phases 1-3) |
| `simulation/observation.py` | ACTIVE | Agent-facing view with information hiding |
| `simulation/action_space.py` | ACTIVE | Legal action enumeration with TAUNT enforcement |
| `simulation/simulator.py` | ACTIVE | Fast N-game rollout with statistics, custom deck/pool support |
| `simulation/replay.py` | ACTIVE | Seed + action log replay system |
| `agents/base.py` | ACTIVE | Abstract Agent interface with choose_action |
| `agents/random_agent.py` | ACTIVE | Random baseline agent (seeded, deterministic) |
| `scripts/play.py` | ACTIVE | Human vs agent CLI interface |
| `scripts/simulate.py` | ACTIVE | Run N games, output stats CLI (supports greedy, mcts agents) |
| `tests/simulation/test_action_space.py` | ACTIVE | 26 legal action enumeration tests |
| `tests/simulation/test_taunt_mechanics.py` | ACTIVE | 10 TAUNT enforcement tests |
| `tests/simulation/test_observation.py` | ACTIVE | 34 info hiding and perspective tests |
| `tests/simulation/test_game_state.py` | ACTIVE | 33 immutable snapshot tests |
| `tests/simulation/test_simulator.py` | ACTIVE | 32 tests: game execution, statistics, custom decks/pools |
| `tests/simulation/test_replay.py` | ACTIVE | 26 deterministic replay tests |
| `tests/agents/test_base_agent.py` | ACTIVE | 23 agent interface tests |
| `tests/agents/test_random_agent.py` | ACTIVE | 24 random agent tests |

## Phase 4 — Data-Driven Card Pool ✅ COMPLETE

| File | Status | Notes |
|------|--------|-------|
| `hearthstone/cards/registry.py` | ACTIVE | Card lookup by ID, DBF ID, filtering by class/type/mechanic |
| `hearthstone/data/cards_collectible.json` | ACTIVE | 7662 real Hearthstone cards (collectible only) |
| `hearthstone/data/cards.json` | ACTIVE | 7662 cards (currently same as collectible, can be expanded with tokens/powers) |
| `hearthstone/data/heroes.json` | ACTIVE | 11 basic hero cards (HERO_01 through HERO_11) |
| `tests/cards/test_registry.py` | ACTIVE | 16 tests: loading, lookup, filtering, type parsing |
| `deckbuilding/deck.py` | ACTIVE | Deck class, CardSpec with rarity, build_pool_from_registry, deck codes |
| `tests/deckbuilding/test_deck.py` | ACTIVE | 53 tests: creation, validation, stats, serialization, registry pool, card_class filtering, card_set filtering |
| `deckbuilding/constraints.py` | ACTIVE | Deck validation: size, copy limits, class restrictions, formats, bans |
| `tests/deckbuilding/test_constraints.py` | ACTIVE | 23 tests: basic rules, class restrictions, formats, STANDARD_SETS integrity |
| `deckbuilding/evaluator.py` | ACTIVE | Deck evaluation: heuristic, simulation, hybrid; evaluate_genotype with real Simulator |
| `tests/deckbuilding/test_evaluator.py` | ACTIVE | 21 tests: heuristic, simulation, ranking, evaluate_genotype |
| `scripts/import_cards.py` | ACTIVE | Import card data from HearthstoneJSON API |

## Phase 5 — RL Environment & Training Baseline ✅ COMPLETE

| File | Status | Notes |
|------|--------|-------|
| `agents/greedy_agent.py` | ACTIVE | Heuristic agent with greedy value maximization |
| `tests/agents/test_greedy_agent.py` | ACTIVE | 16 tests: action selection, heuristics, edge cases |
| `agents/mcts_agent.py` | ACTIVE | Monte Carlo Tree Search agent with UCB1 selection |
| `tests/agents/test_mcts_agent.py` | ACTIVE | 29 tests: node structure, UCB1, MCTS phases, edge cases |
| `agents/rl/features.py` | ACTIVE | Observation → 106-dim feature vector (normalized to [0,1]) |
| `tests/agents/rl/test_features.py` | ACTIVE | 21 tests: scalar, board, hand, misc features, normalization |
| `agents/rl/networks.py` | ACTIVE | Actor-Critic network with shared backbone, action masking |
| `tests/agents/rl/test_networks.py` | ACTIVE | 28 tests: architecture, forward pass, masking, gradients, determinism |
| `agents/rl/env.py` | ACTIVE | Gymnasium wrapper with Discrete(100) action space, sparse rewards |
| `tests/agents/rl/test_env.py` | ACTIVE | 29 tests: creation, reset, step, gameplay, action mapping, truncation |
| `agents/rl/ppo_agent.py` | ACTIVE | PPO-Clip agent with GAE, action masking, rollout buffer, save/load |
| `tests/agents/rl/test_ppo_agent.py` | ACTIVE | 38 tests: creation, action selection, buffer, GAE, update, env integration, save/load |
| `training/config.py` | ACTIVE | TrainingConfig dataclass with all hyperparameters, serialization |
| `tests/training/test_config.py` | ACTIVE | 26 tests: defaults, custom, to_dict, from_dict, round-trip |
| `training/trainer.py` | ACTIVE | Training loop with episode collection, PPO updates, eval, checkpoints |
| `tests/training/test_trainer.py` | ACTIVE | 37 tests: creation, opponents, train loop, eval, checkpoints, callbacks |
| `analysis/metrics.py` | ACTIVE | EpisodeMetrics, MetricsTracker, compute_win_rate, rolling summaries |
| `tests/analysis/test_metrics.py` | ACTIVE | 33 tests: EpisodeMetrics, compute_win_rate, recording, summary, format |
| `scripts/train.py` | ACTIVE | CLI to launch PPO training with configurable params |
| `scripts/evaluate.py` | ACTIVE | CLI to evaluate any agent pair over N games |

## Phase 6 — Deck Optimization ✅ COMPLETE

| File | Status | Notes |
|------|--------|-------|
| `deckbuilding/genetic.py` | ACTIVE | GA with crossover, mutation, tournament selection, play_match evaluation |
| `tests/deckbuilding/test_genetic.py` | ACTIVE | 43 tests: play_match, evaluate_deck, crossover, mutate, tournament, run_ga |

## Phase 7 — Self-Play & Curriculum ✅ COMPLETE

| File | Status | Notes |
|------|--------|-------|
| `agents/rl/self_play.py` | ACTIVE | EloTracker + SelfPlayManager with opponent pool, snapshot scheduling |
| `tests/agents/rl/test_self_play.py` | ACTIVE | 34 tests: Elo updates, snapshots, opponent selection, pool management |
| `training/curriculum.py` | ACTIVE | Stage dataclass + CurriculumManager with staged progression, default curriculum |
| `tests/training/test_curriculum.py` | ACTIVE | 26 tests: stages, advancement, env config, status, default curriculum |
| `training/distributed.py` | ACTIVE | ExperienceBuffer + GameWorker + DistributedTrainer with centralized PPO |
| `tests/training/test_distributed.py` | ACTIVE | 39 tests: buffer ops, worker episodes, trainer collection, training loop |

## Phase 8 — Analysis & Meta-Game ✅ COMPLETE

| File | Status | Notes |
|------|--------|-------|
| `deckbuilding/meta.py` | ACTIVE | Archetype dataclass + MetaTracker with win rates, popularity, counters, RPS detection |
| `tests/deckbuilding/test_meta.py` | ACTIVE | 25 tests: archetypes, match recording, win rates, counters, meta summary |
| `analysis/matchups.py` | ACTIVE | MatchupMatrix with symmetric win/loss tracking, best/worst matchup queries |
| `tests/analysis/test_matchups.py` | ACTIVE | 18 tests: recording, win rates, symmetric, table export, best/worst |
| `analysis/visualize.py` | ACTIVE | Data generators: training curves, action distributions, Elo progression, heatmaps |
| `tests/analysis/test_visualize.py` | ACTIVE | 15 tests: training curves, action counts, Elo data, heatmap format |
| `analysis/export.py` | ACTIVE | CSV/JSON export for training metrics, matchup matrices, meta summaries |
| `tests/analysis/test_export.py` | ACTIVE | 18 tests: CSV format, JSON validity, empty trackers, loss columns |

## Phase 9 — Co-Evolutionary Deck Optimization ✅ COMPLETE

| File | Status | Notes |
|------|--------|-------|
| `deckbuilding/coevolution.py` | ACTIVE | CoevolutionConfig, CoevolutionEngine, classify_archetype, constrained_mutate |
| `tests/deckbuilding/test_coevolution.py` | ACTIVE | 36 tests: config, archetype classification, constrained mutation, engine run, meta integration, class-restricted evolution, set-restricted evolution |

## CI/CD

| File | Status | Notes |
|------|--------|-------|
| `.github/workflows/tests.yml` | ACTIVE | Matrix: Ubuntu/macOS/Windows × Python 3.12/3.13/3.14, unit tests + smoke test |
| `scripts/smoke_test.py` | ACTIVE | CI smoke test: 10-game Greedy vs Random, asserts Greedy wins majority |

---

## Roadmap — Remaining Work for "Best Deck for Current Season"

### 1. Class-Restricted Evolution — ✅ COMPLETE
**Priority: Critical | Effort: Small**
CoevolutionEngine generates decks mixing all classes together. The validation rules exist in `constraints.py` but aren't enforced during generation. Need: per-class card pools, class-specific evolution runs.

### 2. Standard Format Filtering — ✅ COMPLETE
**Priority: Critical | Effort: Small**
`build_pool_from_registry` grabs cards from all sets ever printed. Format definitions exist in `constraints.py` (STANDARD_SETS) but aren't applied during pool building. Need: filter pool to current Standard rotation sets.

### 3. Hero Powers — ✅ COMPLETE
**Priority: High | Effort: Medium**
All 11 hero powers implemented (Mage, Warlock, Priest, Paladin, Hunter, Warrior, Shaman, Rogue, Druid, Demon Hunter, Death Knight). Player has `hero_class`, `armor`, `take_damage()`. Game accepts hero class params. Action space enumerates hero powers with targeting for Mage/Priest. 35 new tests.

### 4. Battlecry Effects — ✅ COMPLETE
**Priority: High | Effort: Large**
Text-based battlecry parser resolves 6 common patterns: deal damage, draw cards, restore health, gain armor, summon token, buff all friendly. BATTLECRY added to SUPPORTED_MECHANICS, unlocking ~700+ cards for GA pool. `play_card()` now accepts `opponent` param and triggers battlecry resolution. 29 new tests.

### 5. Spell System — NOT STARTED
**Priority: High | Effort: Large**
2,170 spell cards exist in the registry but spell execution is a placeholder in `actions.py`. Need: targeting system, effect resolution for damage spells, buffs, removal, and AoE.

### 6. Weapon System — NOT STARTED
**Priority: Medium | Effort: Medium**
229 weapon cards parsed but no equip/attack/durability logic. Need: weapon slot on player, hero attack action, durability tracking, weapon deathrattles.

### 7. Smarter Evaluation Agent — NOT STARTED
**Priority: Medium | Effort: Medium**
GreedyAgent makes reasonable tactical plays but can't recognize card synergies, deck archetypes, combos, or adapt strategy to matchups. Need: improved heuristics or trained RL agent that understands deck-level strategy.

---

## Infrastructure (always active)

| File | Status | Notes |
|------|--------|-------|
| `pyproject.toml` | CONFIG | Build system, deps (ml/dev/analysis groups), pytest config, ruff |
| `Makefile` | CONFIG | test, lint, clean commands |
| `PROJECT_STRUCTURE.md` | CONFIG | Architecture documentation |
| `MANIFEST.md` | CONFIG | This file |
| `hearthstone/__init__.py` | ACTIVE | Package init |
| `hearthstone/engine/__init__.py` | ACTIVE | Package init |
| `hearthstone/cards/__init__.py` | ACTIVE | Package init |
| `simulation/__init__.py` | ACTIVE | Package init |
| `agents/__init__.py` | ACTIVE | Package init |
| `agents/rl/__init__.py` | ACTIVE | Package init |
| `deckbuilding/__init__.py` | ACTIVE | Package init |
| `training/__init__.py` | ACTIVE | Package init |
| `analysis/__init__.py` | ACTIVE | Package init |
