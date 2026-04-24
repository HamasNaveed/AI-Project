"""
AI2002 Assignment - Part B: Search-Based Algorithm Implementation
2D Turn-Based Dungeon RPG with Minimax AI
=========================================
Features:
  - Turn-based combat: Player vs AI Enemy/Boss
  - Stamina system (Attack:2, Heavy:4, Defend:1, Rest recovers 3)
  - Reputation system influencing AI heuristics
  - Depth-limited Minimax (depth 3) with Alpha-Beta Pruning & Memoization
  - Adaptive heuristics: HP, stamina, reputation, action patterns
"""

import copy
import random
from collections import Counter

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
ACTIONS = ["Attack", "Heavy", "Defend", "Rest"]

ACTION_STAMINA_COST = {
    "Attack": 2,
    "Heavy":  4,
    "Defend": 1,
    "Rest":   0,   # Rest recovers stamina instead
}
REST_RECOVERY = 3

# Base damage / effects
ACTION_DAMAGE = {
    "Attack": 15,
    "Heavy":  30,
    "Defend": 0,
    "Rest":   0,
}

# ─────────────────────────────────────────────
# CombatantState – lightweight, copyable
# ─────────────────────────────────────────────
class CombatantState:
    """Holds all mutable state for one combatant (player or enemy)."""

    def __init__(self, name: str, hp: int, max_hp: int, stamina: int,
                 max_stamina: int, is_boss: bool = False):
        self.name        = name
        self.hp          = hp
        self.max_hp      = max_hp
        self.stamina     = stamina
        self.max_stamina = max_stamina
        self.is_boss     = is_boss
        self.defending   = False          # active this turn only
        # Action history (used for pattern recognition)
        self.action_history: list[str] = []

    def is_alive(self) -> bool:
        return self.hp > 0

    def clone(self) -> "CombatantState":
        c = CombatantState(self.name, self.hp, self.max_hp,
                           self.stamina, self.max_stamina, self.is_boss)
        c.defending      = self.defending
        c.action_history = self.action_history.copy()
        return c

    def status_line(self) -> str:
        bar_len = 20
        hp_fill  = int(bar_len * self.hp  / self.max_hp)
        st_fill  = int(bar_len * self.stamina / self.max_stamina)
        hp_bar   = "█" * hp_fill  + "░" * (bar_len - hp_fill)
        st_bar   = "█" * st_fill  + "░" * (bar_len - st_fill)
        return (f"  {self.name:<12} HP [{hp_bar}] {self.hp:>3}/{self.max_hp}"
                f"  ST [{st_bar}] {self.stamina:>2}/{self.max_stamina}")


# ─────────────────────────────────────────────
# ReputationSystem
# ─────────────────────────────────────────────
class ReputationSystem:
    """
    Tracks combat reputation stats for the AI enemy.
    Dimensions: Good/Evil, Aggressive/Defensive.
    These shift based on observed player behaviour and influence AI heuristics.
    """

    def __init__(self):
        self.stats = {"Good": 0, "Evil": 0, "Aggressive": 0, "Defensive": 0}

    def record_action(self, action: str, actor: str = "player"):
        """Update reputation based on what the player does."""
        if actor == "player":
            if action in ("Attack", "Heavy"):
                self.stats["Aggressive"] += 1
                self.stats["Evil"]       += 1
            elif action == "Defend":
                self.stats["Defensive"] += 1
                self.stats["Good"]      += 1
            elif action == "Rest":
                self.stats["Defensive"] += 1

    def aggression_ratio(self) -> float:
        """0.0 = fully defensive, 1.0 = fully aggressive."""
        total = self.stats["Aggressive"] + self.stats["Defensive"]
        if total == 0:
            return 0.5
        return self.stats["Aggressive"] / total

    def evil_ratio(self) -> float:
        total = self.stats["Good"] + self.stats["Evil"]
        if total == 0:
            return 0.5
        return self.stats["Evil"] / total

    def summary(self) -> str:
        return (f"Reputation → Aggressive:{self.stats['Aggressive']} "
                f"Defensive:{self.stats['Defensive']} "
                f"Evil:{self.stats['Evil']} Good:{self.stats['Good']}")


# ─────────────────────────────────────────────
# Simulation helpers
# ─────────────────────────────────────────────
def can_perform(actor: CombatantState, action: str) -> bool:
    """Check if actor has enough stamina for action."""
    if action == "Rest":
        return True
    return actor.stamina >= ACTION_STAMINA_COST[action]


def available_actions(actor: CombatantState) -> list[str]:
    return [a for a in ACTIONS if can_perform(actor, a)]


def simulate_turn(attacker: CombatantState, defender: CombatantState,
                  action: str) -> tuple[CombatantState, CombatantState]:
    """
    Apply one action; returns (new_attacker, new_defender) clones.
    Does NOT modify the originals.
    """
    a = attacker.clone()
    d = defender.clone()

    # Reset defend flag from previous turn
    a.defending = False

    if action == "Rest":
        a.stamina = min(a.max_stamina, a.stamina + REST_RECOVERY)
        a.action_history.append("Rest")
        return a, d

    # Deduct stamina
    a.stamina = max(0, a.stamina - ACTION_STAMINA_COST[action])
    a.action_history.append(action)

    if action == "Defend":
        a.defending = True
        return a, d

    # Compute damage
    dmg = ACTION_DAMAGE[action]

    # Boss bonus: bosses deal 20 % more damage
    if a.is_boss:
        dmg = int(dmg * 1.2)

    # Defender halves damage if defending
    if d.defending:
        dmg = dmg // 2

    d.hp = max(0, d.hp - dmg)
    return a, d


# ─────────────────────────────────────────────
# Heuristic
# ─────────────────────────────────────────────
def heuristic(enemy: CombatantState, player: CombatantState,
              reputation: ReputationSystem) -> float:
    """
    Evaluate the game state from the AI (enemy) perspective.
    Higher score = better for AI.

    Factors:
      1. HP difference (weighted heavily)
      2. Stamina advantage
      3. Reputation-driven aggression modifier
      4. Player action pattern (punish predictable play)
    """
    # 1. HP factor
    hp_score = (enemy.hp / enemy.max_hp) - (player.hp / player.max_hp)
    hp_score *= 100

    # 2. Stamina factor
    st_score = (enemy.stamina / enemy.max_stamina) - (player.stamina / player.max_stamina)
    st_score *= 20

    # 3. Reputation modifier
    #    If player is aggressive, bias towards Defend; else bias towards Heavy.
    aggr = reputation.aggression_ratio()
    rep_modifier = 0.0
    if aggr > 0.6:
        # Player is aggressive → reward AI for defending
        rep_modifier = 10 if enemy.defending else -5
    else:
        # Player is passive → punish AI for being passive
        rep_modifier = -10 if enemy.defending else 5

    # 4. Pattern recognition: if player repeats same action ≥ 3 times, AI bonus
    pattern_bonus = 0.0
    if len(player.action_history) >= 3:
        last_three = player.action_history[-3:]
        if len(set(last_three)) == 1:          # all same
            pattern_bonus = 15.0               # AI "figured out" the pattern

    # 5. Low stamina penalty (avoid being unable to act)
    stamina_penalty = -30 if enemy.stamina < 2 else 0

    return hp_score + st_score + rep_modifier + pattern_bonus + stamina_penalty


# ─────────────────────────────────────────────
# Minimax with Alpha-Beta Pruning & Memoization
# ─────────────────────────────────────────────
def minimax(enemy: CombatantState, player: CombatantState,
            reputation: ReputationSystem,
            depth: int, alpha: float, beta: float,
            maximising: bool, memo: dict = None) -> tuple[float, str | None]:
    """
    Depth-limited Minimax with Alpha-Beta pruning and Memoization.

    Returns (score, best_action).
    maximising=True  → AI (enemy) turn
    maximising=False → Player turn
    """
    if memo is None:
        memo = {}

    # State Hashing for Memoization (ignoring full history list for fast caching)
    state_key = (
        enemy.hp, enemy.stamina, enemy.defending,
        player.hp, player.stamina, player.defending,
        depth, maximising
    )

    if state_key in memo:
        return memo[state_key]

    # Terminal conditions
    if depth == 0 or not enemy.is_alive() or not player.is_alive():
        return heuristic(enemy, player, reputation), None

    best_action = None

    if maximising:
        # AI's turn – maximise score
        max_val = float("-inf")
        actions = available_actions(enemy)
        if not actions:
            actions = ["Rest"]

        for action in actions:
            new_enemy, new_player = simulate_turn(enemy, player, action)
            score, _ = minimax(new_enemy, new_player, reputation,
                               depth - 1, alpha, beta, False, memo)
            if score > max_val:
                max_val    = score
                best_action = action
            alpha = max(alpha, score)
            if beta <= alpha:
                break   # Alpha-Beta pruning
                
        memo[state_key] = (max_val, best_action)
        return max_val, best_action

    else:
        # Player's turn – minimise score (assume best player play)
        min_val = float("inf")
        actions = available_actions(player)
        if not actions:
            actions = ["Rest"]

        for action in actions:
            new_player, new_enemy = simulate_turn(player, enemy, action)
            score, _ = minimax(new_enemy, new_player, reputation,
                               depth - 1, alpha, beta, True, memo)
            if score < min_val:
                min_val    = score
                best_action = action
            beta = min(beta, score)
            if beta <= alpha:
                break
                
        memo[state_key] = (min_val, best_action)
        return min_val, best_action


def ai_choose_action(enemy: CombatantState, player: CombatantState,
                     reputation: ReputationSystem,
                     depth: int = 3) -> str:
    """Entry point for the AI to pick its action via Minimax."""
    memoization_table = {}
    _, action = minimax(enemy, player, reputation,
                        depth, float("-inf"), float("inf"), True, memoization_table)
    
    # Fallback safety
    if action is None or not can_perform(enemy, action):
        return "Rest"
    return action


# ─────────────────────────────────────────────
# Player input
# ─────────────────────────────────────────────
def get_player_action(player: CombatantState) -> str:
    """Prompt the human player for an action."""
    avail = available_actions(player)
    print("\n  Your available actions:")
    for i, a in enumerate(ACTIONS, 1):
        cost = ACTION_STAMINA_COST[a]
        status = "" if a in avail else "  [NOT ENOUGH STAMINA]"
        recovery = "  (+3 stamina)" if a == "Rest" else f"  (cost: {cost} ST)"
        print(f"    [{i}] {a:<12}{recovery}{status}")

    while True:
        try:
            choice = int(input("  Enter choice (1-4): ").strip())
            action = ACTIONS[choice - 1]
            if action not in avail:
                print("  ✗ Not enough stamina. Choose another.")
            else:
                return action
        except (ValueError, IndexError):
            print("  ✗ Invalid input. Enter 1, 2, 3, or 4.")


# ─────────────────────────────────────────────
# Combat Log helpers
# ─────────────────────────────────────────────
def divider(char="─", width=60):
    print(char * width)

def print_header(text: str):
    divider("═")
    print(f"  {text}")
    divider("═")

def describe_action(actor_name: str, action: str, target_name: str,
                    dmg_dealt: int, blocked: bool) -> str:
    verbs = {
        "Attack": f"attacks {target_name}",
        "Heavy":  f"unleashes a heavy blow on {target_name}",
        "Defend": "takes a defensive stance",
        "Rest":   "rests to recover stamina",
    }
    desc = f"  ⚔  {actor_name} {verbs[action]}"
    if dmg_dealt > 0:
        desc += f"  →  {dmg_dealt} damage"
        if blocked:
            desc += " (partially blocked!)"
    return desc


# ─────────────────────────────────────────────
# Main Combat Loop
# ─────────────────────────────────────────────
def run_combat(player: CombatantState, enemy: CombatantState,
               reputation: ReputationSystem,
               player_controlled: bool = True,
               minimax_depth: int = 3):
    """
    Full combat loop.
    player_controlled=True  → Human input for player actions.
    player_controlled=False → Simulate player actions randomly (demo mode).
    """
    turn = 1
    print_header(f"⚔  COMBAT START: {player.name} vs {enemy.name}  ⚔")

    while player.is_alive() and enemy.is_alive():
        divider()
        print(f"\n  ── Turn {turn} ──")
        print(player.status_line())
        print(enemy.status_line())
        print(f"  {reputation.summary()}")

        # ── Player turn ──────────────────────────────
        print(f"\n  [{player.name}'s turn]")
        if player_controlled:
            p_action = get_player_action(player)
        else:
            # Auto-simulate: weight towards attack when healthy, defend when low
            weights = []
            for a in ACTIONS:
                if a not in available_actions(player):
                    weights.append(0)
                elif a == "Heavy" and player.hp > 40:
                    weights.append(3)
                elif a == "Defend" and player.hp < 30:
                    weights.append(4)
                elif a == "Rest" and player.stamina < 3:
                    weights.append(5)
                else:
                    weights.append(2)
            p_action = random.choices(ACTIONS, weights=weights, k=1)[0]
            print(f"  (Auto) {player.name} chooses: {p_action}")

        # Record player action in reputation before simulating
        reputation.record_action(p_action, actor="player")

        # Calculate damage for logging
        was_defending = enemy.defending
        old_enemy_hp  = enemy.hp
        player, enemy = simulate_turn(player, enemy, p_action)
        p_dmg = old_enemy_hp - enemy.hp
        print(describe_action(player.name, p_action, enemy.name, p_dmg, was_defending))

        if not enemy.is_alive():
            break

        # ── Enemy (AI) turn ──────────────────────────
        print(f"\n  [{enemy.name}'s turn  –  Minimax depth {minimax_depth}]")
        e_action = ai_choose_action(enemy, player, reputation, minimax_depth)

        was_defending_p = player.defending
        old_player_hp   = player.hp
        enemy, player   = simulate_turn(enemy, player, e_action)
        e_dmg = old_player_hp - player.hp
        print(describe_action(enemy.name, e_action, player.name, e_dmg, was_defending_p))
        print(f"  (AI reasoning: chose '{e_action}' via Minimax α-β pruning)")

        turn += 1

    # ── Result ────────────────────────────────────────
    divider("═")
    if player.is_alive():
        print(f"\n  🏆  VICTORY!  {player.name} defeated {enemy.name}!")
    else:
        print(f"\n  💀  DEFEAT!   {enemy.name} has won the battle.")
    print(f"\n  Final HP  →  {player.name}: {player.hp}  |  {enemy.name}: {enemy.hp}")
    print(f"  Turns elapsed: {turn - 1}")
    divider("═")


# ─────────────────────────────────────────────
# Demo / Entry Point
# ─────────────────────────────────────────────
def main():
    print_header("🏰  DUNGEON RPG  –  AI2002 Part B Demo  🏰")
    print("""
  Stamina costs:  Attack=2  |  Heavy Attack=4  |  Defend=1  |  Rest recovers 3
  AI uses depth-3 Minimax with Alpha-Beta pruning and adaptive heuristics.
    """)

    # ── Choose mode ───────────────────────────────────
    print("  Select mode:")
    print("    [1] Play yourself (vs AI Enemy)")
    print("    [2] Play yourself (vs AI Boss – tougher)")
    print("    [3] Auto-simulation (watch AI vs AI demo)")
    while True:
        try:
            mode = int(input("  Choice: ").strip())
            if mode in (1, 2, 3):
                break
        except ValueError:
            pass
        print("  Enter 1, 2, or 3.")

    is_boss = (mode == 2)
    player_controlled = (mode != 3)

    # ── Build combatants ──────────────────────────────
    player = CombatantState(
        name="Hero",
        hp=100, max_hp=100,
        stamina=10, max_stamina=10,
        is_boss=False,
    )

    if is_boss:
        enemy = CombatantState(
            name="Dragon Boss",
            hp=140, max_hp=140,
            stamina=12, max_stamina=12,
            is_boss=True,
        )
        depth = 3
    else:
        enemy = CombatantState(
            name="Dark Knight",
            hp=100, max_hp=100,
            stamina=10, max_stamina=10,
            is_boss=False,
        )
        depth = 3

    reputation = ReputationSystem()

    # ── Run ───────────────────────────────────────────
    run_combat(player, enemy, reputation,
               player_controlled=player_controlled,
               minimax_depth=depth)


if __name__ == "__main__":
    main()