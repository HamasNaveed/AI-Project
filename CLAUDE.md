# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Run the game (interactive)
python dungeon_rpg_ai_enhanced.py

# Run all tests
python -m unittest test_dungeon_rpg -v
# or
python test_dungeon_rpg.py

# Run a single test class
python -m unittest test_dungeon_rpg.TestMinimaxSearch -v

# Run a single test method
python -m unittest test_dungeon_rpg.TestMinimaxSearch.test_minimax_returns_action -v
```

No external dependencies — stdlib only (`copy`, `random`, `collections`, `typing`, `unittest`, `time`).

## Architecture

All logic lives in `dungeon_rpg_ai_enhanced.py`. The design separates immutable simulation from mutable game state.

**State layer**
- `CombatantState` — all mutable per-combatant data (hp, stamina, defending flag, action history). `clone()` is O(n) and is the only way to create search-tree copies. `simulate_turn()` always clones before mutating, so originals are never touched.

**AI decision pipeline**
1. `ai_choose_action()` — entry point; calls `minimax()` with depth=3 and fallback to `"Rest"` if no valid action is returned.
2. `minimax()` — depth-limited minimax with alpha-beta pruning. `maximising=True` is the AI/enemy turn; `maximising=False` simulates optimal player response. Accepts optional `OptimizationMetrics` and `PatternAnalyzer` to enable enhanced tracking.
3. `heuristic()` — evaluates a leaf node from the enemy's perspective: HP ratio (×100), stamina ratio (×20), reputation modifier, pattern bonuses, low-stamina penalty (−30 if stamina < 2).

**Behavioural systems**
- `ReputationSystem` — records player actions each turn; `aggression_ratio()` and `evil_ratio()` shift AI heuristic weights at runtime.
- `PatternAnalyzer` — pure static methods; detects repeating windows, cyclic sequences (ABAB), computes Shannon entropy and a `predictability_score` [0,1]. Used inside `heuristic()` to add bonuses when the player is predictable.
- `OptimizationMetrics` — optional counter for nodes evaluated vs. pruned, passed through `minimax()` to measure alpha-beta efficiency.

**Simulation**
- `simulate_turn(attacker, defender, action)` — pure function; clones both states, applies stamina cost/recovery/damage, returns `(new_attacker, new_defender)`. Boss flag adds 1.2× damage; `defending=True` halves incoming damage via integer division.
- `available_actions(actor)` / `can_perform(actor, action)` — stamina gate used by both player input and minimax branching.

**Game loop**
- `run_combat()` — drives turns until a combatant dies; supports human input (`player_controlled=True`) or weighted-random auto-simulation. Instantiates `PatternAnalyzer` and optionally `OptimizationMetrics` per combat session.

**Constants** (`ACTIONS`, `ACTION_STAMINA_COST`, `ACTION_DAMAGE`, `REST_RECOVERY`) are module-level and referenced throughout — change them in one place.
