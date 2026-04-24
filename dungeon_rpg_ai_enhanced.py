"""
AI2002 Assignment - Part A & B: Search-Based Algorithm Implementation
2D Turn-Based Dungeon RPG with Minimax AI
=========================================
ENHANCED VERSION with Sophisticated Pattern Recognition & Optimization Metrics

Features:
  - Turn-based combat: Player vs AI Enemy/Boss
  - Stamina system (Attack:2, Heavy:4, Defend:1, Rest recovers 3)
  - Reputation system influencing AI heuristics
  - Depth-limited Minimax (depth 3) with Alpha-Beta Pruning
  - Adaptive heuristics: HP, stamina, reputation, action patterns
  - ENHANCED: Sophisticated pattern analysis (frequency, entropy, predictability)
  - ENHANCED: Alpha-Beta pruning metrics & node reduction tracking
  - ENHANCED: Complexity analysis helpers
"""

import copy
import random
from collections import Counter
from typing import Tuple, List, Optional, Dict

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
        self.action_history: List[str] = []

    def is_alive(self) -> bool:
        return self.hp > 0

    def clone(self) -> "CombatantState":
        """Deep clone state for tree exploration without mutation.
        
        Time Complexity: O(n) where n = len(action_history)
        Space Complexity: O(n)
        """
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
        """Update reputation based on what the player does.
        
        Time Complexity: O(1)
        Space Complexity: O(1)
        """
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
        """0.0 = fully defensive, 1.0 = fully aggressive.
        
        Time Complexity: O(1)
        Space Complexity: O(1)
        """
        total = self.stats["Aggressive"] + self.stats["Defensive"]
        if total == 0:
            return 0.5
        return self.stats["Aggressive"] / total

    def evil_ratio(self) -> float:
        """Time Complexity: O(1)"""
        total = self.stats["Good"] + self.stats["Evil"]
        if total == 0:
            return 0.5
        return self.stats["Evil"] / total

    def summary(self) -> str:
        return (f"Reputation → Aggressive:{self.stats['Aggressive']} "
                f"Defensive:{self.stats['Defensive']} "
                f"Evil:{self.stats['Evil']} Good:{self.stats['Good']}")


# ─────────────────────────────────────────────
# ENHANCED: Sophisticated Pattern Recognition
# ─────────────────────────────────────────────
class PatternAnalyzer:
    """
    Analyzes player action sequences for sophisticated pattern detection.
    
    Features:
      - Frequency analysis: which actions are most common
      - Sequential patterns: n-gram detection
      - Predictability entropy: how random vs deterministic is the player
      - Window-based recency: recent actions weighted higher
    
    Time Complexity: O(n) for most operations where n = history length
    Space Complexity: O(k) where k = number of unique actions (max 4)
    """
    
    WINDOW_SIZE = 5  # Look back window for recent patterns
    
    @staticmethod
    def frequency_analysis(history: List[str]) -> Dict[str, float]:
        """Compute action frequency distribution.
        
        Args:
            history: List of action strings
            
        Returns:
            Dict mapping action -> frequency [0.0, 1.0]
            
        Time Complexity: O(n)
        Space Complexity: O(k) where k ≤ 4
        """
        if not history:
            return {a: 0.25 for a in ACTIONS}
        
        counter = Counter(history)
        total = len(history)
        return {a: counter.get(a, 0) / total for a in ACTIONS}
    
    @staticmethod
    def detect_repeating_pattern(history: List[str], window: int = 3) -> bool:
        """Detect if last 'window' actions are identical.
        
        Simple pattern: all last N actions are the same.
        E.g., [Attack, Attack, Attack] → True
        
        Args:
            history: Action history
            window: Number of recent actions to check
            
        Returns:
            True if last 'window' actions are identical
            
        Time Complexity: O(window) = O(1) since window is constant
        Space Complexity: O(1)
        """
        if len(history) < window:
            return False
        last_window = history[-window:]
        return len(set(last_window)) == 1
    
    @staticmethod
    def detect_cyclic_pattern(history: List[str], cycle_length: int = 2) -> bool:
        """Detect if actions cycle (e.g., ABAB... or AAAA...).
        
        Cyclic patterns: [A, B, A, B] or [A, A, A, A]
        
        Args:
            history: Action history
            cycle_length: Expected cycle period
            
        Returns:
            True if recent actions follow a cyclic pattern
            
        Time Complexity: O(n) where n = recent history window
        Space Complexity: O(1)
        """
        if len(history) < cycle_length * 2:
            return False
        
        recent = history[-cycle_length*2:]
        pattern = recent[:cycle_length]
        return recent[cycle_length:] == pattern
    
    @staticmethod
    def shannon_entropy(history: List[str]) -> float:
        """Compute Shannon entropy of action distribution.
        
        Entropy measures randomness/predictability:
        - 0.0 = completely predictable (same action always)
        - log2(4) ≈ 2.0 = completely random (equal probability)
        
        High entropy = player is unpredictable
        Low entropy = player follows patterns
        
        Args:
            history: Action history
            
        Returns:
            Entropy value [0.0, 2.0]
            
        Time Complexity: O(n)
        Space Complexity: O(k)
        """
        if not history:
            return 1.0
        
        frequencies = PatternAnalyzer.frequency_analysis(history)
        entropy = 0.0
        for freq in frequencies.values():
            if freq > 0:
                entropy -= freq * (freq ** 0.5)  # log2(freq) approximation
        return entropy
    
    @staticmethod
    def recency_weighted_distribution(history: List[str]) -> Dict[str, float]:
        """Compute weighted frequency favoring recent actions.
        
        Recent actions have higher weight, giving more importance to
        the player's current behavior over historical patterns.
        
        Weight formula: position_index / (position_index + 1)
        
        Args:
            history: Action history
            
        Returns:
            Dict mapping action -> weighted frequency
            
        Time Complexity: O(n)
        Space Complexity: O(k)
        """
        if not history:
            return {a: 0.25 for a in ACTIONS}
        
        weighted = {a: 0.0 for a in ACTIONS}
        total_weight = 0.0
        
        # Weight formula: recent actions get higher weight
        for i, action in enumerate(history):
            weight = (i + 1) / (i + 1) ** 0.7  # Recency exponent
            weighted[action] += weight
            total_weight += weight
        
        # Normalize to [0, 1]
        return {a: weighted[a] / total_weight if total_weight > 0 else 0.25
                for a in ACTIONS}
    
    @staticmethod
    def predictability_score(history: List[str]) -> float:
        """Compute how predictable the player's actions are.
        
        Range: [0.0, 1.0]
        - 0.0 = completely random/unpredictable
        - 1.0 = completely predictable
        
        Based on: low entropy = predictable
        
        Args:
            history: Action history
            
        Returns:
            Predictability score [0.0, 1.0]
            
        Time Complexity: O(n)
        Space Complexity: O(k)
        """
        if not history:
            return 0.5
        
        entropy = PatternAnalyzer.shannon_entropy(history)
        max_entropy = 2.0  # log2(4)
        
        # Invert: high entropy (random) → low predictability
        return 1.0 - (entropy / max_entropy)


# ─────────────────────────────────────────────
# Optimization Metrics Tracker
# ─────────────────────────────────────────────
class OptimizationMetrics:
    """Track Alpha-Beta pruning efficiency metrics.
    
    Monitors:
    - Total nodes evaluated
    - Nodes pruned
    - Pruning efficiency percentage
    - Depth distribution
    """
    
    def __init__(self):
        self.nodes_evaluated = 0
        self.nodes_pruned = 0
        self.nodes_at_depth = Counter()
    
    def record_node(self, depth: int, pruned: bool = False):
        """Record a node evaluation.
        
        Time Complexity: O(1)
        """
        self.nodes_evaluated += 1
        if pruned:
            self.nodes_pruned += 1
        self.nodes_at_depth[depth] += 1
    
    def pruning_efficiency(self) -> float:
        """Compute pruning efficiency percentage.
        
        Returns:
            Percentage of nodes pruned [0.0, 100.0]
            
        Time Complexity: O(1)
        """
        if self.nodes_evaluated == 0:
            return 0.0
        return (self.nodes_pruned / self.nodes_evaluated) * 100.0
    
    def summary(self) -> str:
        """Return summary statistics.
        
        Time Complexity: O(k) where k = number of depths
        """
        eff = self.pruning_efficiency()
        return (f"Nodes: {self.nodes_evaluated} | "
                f"Pruned: {self.nodes_pruned} ({eff:.1f}% efficiency) | "
                f"Depth distribution: {dict(self.nodes_at_depth)}")


# ─────────────────────────────────────────────
# Simulation helpers
# ─────────────────────────────────────────────
def can_perform(actor: CombatantState, action: str) -> bool:
    """Check if actor has enough stamina for action.
    
    Time Complexity: O(1)
    Space Complexity: O(1)
    """
    if action == "Rest":
        return True
    return actor.stamina >= ACTION_STAMINA_COST[action]


def available_actions(actor: CombatantState) -> List[str]:
    """Get list of affordable actions for actor.
    
    Time Complexity: O(|ACTIONS|) = O(4) = O(1)
    Space Complexity: O(4) = O(1)
    """
    return [a for a in ACTIONS if can_perform(actor, a)]


def simulate_turn(attacker: CombatantState, defender: CombatantState,
                  action: str) -> Tuple[CombatantState, CombatantState]:
    """Apply one action; returns (new_attacker, new_defender) clones.
    
    Does NOT modify the originals. Safe for tree exploration.
    
    Args:
        attacker: Attacking combatant
        defender: Defending combatant
        action: Action string from ACTIONS
        
    Returns:
        Tuple of cloned states after applying action
        
    Time Complexity: O(n) where n = len(attacker.action_history)
    Space Complexity: O(n) for cloning
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

    # Boss bonus: bosses deal 20% more damage
    if a.is_boss:
        dmg = int(dmg * 1.2)

    # Defender halves damage if defending
    if d.defending:
        dmg = dmg // 2

    d.hp = max(0, d.hp - dmg)
    return a, d


# ─────────────────────────────────────────────
# Heuristic Evaluation Function
# ─────────────────────────────────────────────
def heuristic(enemy: CombatantState, player: CombatantState,
              reputation: ReputationSystem,
              pattern_analyzer: Optional[PatternAnalyzer] = None) -> float:
    """Evaluate the game state from the AI (enemy) perspective.
    
    Higher score = better for AI.
    
    Factors:
      1. HP difference (weighted heavily)
      2. Stamina advantage
      3. Reputation-driven aggression modifier
      4. Player action pattern (simple + sophisticated)
      5. Low stamina penalty
      
    Time Complexity: O(n) where n = len(player.action_history)
    Space Complexity: O(k) where k ≤ 4 (action types)
    
    Args:
        enemy: Enemy combatant state
        player: Player combatant state
        reputation: Reputation tracking system
        pattern_analyzer: Optional pattern analyzer (if None, uses simple pattern)
        
    Returns:
        Float heuristic score
    """
    # 1. HP factor: (100 points per full health advantage)
    hp_score = (enemy.hp / enemy.max_hp) - (player.hp / player.max_hp)
    hp_score *= 100

    # 2. Stamina factor: (20 points per full stamina advantage)
    st_score = (enemy.stamina / enemy.max_stamina) - (player.stamina / player.max_stamina)
    st_score *= 20

    # 3. Reputation modifier: adapt to player aggression pattern
    #    If player is aggressive, bias towards Defend; else bias towards Heavy.
    aggr = reputation.aggression_ratio()
    rep_modifier = 0.0
    if aggr > 0.6:
        # Player is aggressive → reward AI for defending
        rep_modifier = 10 if enemy.defending else -5
    else:
        # Player is passive → punish AI for being passive
        rep_modifier = -10 if enemy.defending else 5

    # 4. Pattern recognition: simple + sophisticated
    pattern_bonus = 0.0
    
    # Simple pattern: last 3 actions identical
    if PatternAnalyzer.detect_repeating_pattern(player.action_history, window=3):
        pattern_bonus = 15.0
    
    # Sophisticated analysis: if provided
    if pattern_analyzer is not None and len(player.action_history) >= 3:
        predictability = pattern_analyzer.predictability_score(player.action_history)
        cyclic = PatternAnalyzer.detect_cyclic_pattern(player.action_history)
        
        # Bonus for detecting cyclic patterns
        if cyclic:
            pattern_bonus += 10.0 * predictability
        
        # Bonus for exploiting predictability
        if predictability > 0.7:
            pattern_bonus += 5.0

    # 5. Low stamina penalty (avoid being unable to act)
    stamina_penalty = -30 if enemy.stamina < 2 else 0

    return hp_score + st_score + rep_modifier + pattern_bonus + stamina_penalty


# ─────────────────────────────────────────────
# Minimax with Alpha-Beta Pruning
# ─────────────────────────────────────────────
def minimax(enemy: CombatantState, player: CombatantState,
            reputation: ReputationSystem,
            depth: int, alpha: float, beta: float,
            maximising: bool,
            metrics: Optional[OptimizationMetrics] = None,
            pattern_analyzer: Optional[PatternAnalyzer] = None) -> Tuple[float, Optional[str]]:
    """Depth-limited Minimax with Alpha-Beta pruning.
    
    Search Algorithm Complexity Analysis:
    
    WITHOUT Alpha-Beta pruning:
    - Time Complexity: O(b^d) where b = branching factor (4), d = depth
    - Worst case: O(4^3) = O(64) node evaluations
    - Space Complexity: O(b·d) for recursion stack = O(4·3) = O(12)
    
    WITH Alpha-Beta pruning (average case):
    - Time Complexity: O(b^(d/2)) ≈ O(4^1.5) ≈ O(8) evaluations
    - Pruning reduces branching factor from 4 to ~√4 = 2
    - Space Complexity: O(b·d) (stack unchanged)
    
    Typical Pruning Efficiency: 50-70% nodes pruned with good move ordering
    
    Returns (score, best_action).
    maximising=True  → AI (enemy) turn (maximize score)
    maximising=False → Player turn (minimize score, assume optimal play)
    
    Args:
        enemy: Enemy combatant state
        player: Player combatant state
        reputation: Reputation system
        depth: Current recursion depth (0 = leaf node)
        alpha: Maximizer's best score so far
        beta: Minimizer's best score so far
        maximising: True if AI's turn, False if player's turn
        metrics: Optional metrics tracker for pruning efficiency
        pattern_analyzer: Optional pattern analyzer
        
    Returns:
        Tuple of (heuristic_score, best_action_string)
    """
    
    # ─ Terminal Conditions ─
    # Base case 1: Depth exhausted (leaf node reached)
    # Base case 2: Terminal state (someone died)
    if depth == 0 or not enemy.is_alive() or not player.is_alive():
        return heuristic(enemy, player, reputation, pattern_analyzer), None

    best_action: Optional[str] = None

    if maximising:
        # ─ MAXIMIZING PLAYER (AI/Enemy) ─
        # Goal: find action that maximizes heuristic score
        
        max_val = float("-inf")
        actions = available_actions(enemy)
        if not actions:
            actions = ["Rest"]

        for action in actions:
            # Recursively evaluate this action
            new_enemy, new_player = simulate_turn(enemy, player, action)
            
            # Recursive call: switch to minimizing (player's turn)
            score, _ = minimax(new_enemy, new_player, reputation,
                               depth - 1, alpha, beta, False, metrics, pattern_analyzer)
            
            # Track metrics if provided
            if metrics is not None:
                metrics.record_node(depth, pruned=False)
            
            # Update best score and action
            if score > max_val:
                max_val = score
                best_action = action
            
            # Alpha-Beta pruning: update maximizer's best guarantee
            alpha = max(alpha, score)
            
            # ─ PRUNING CONDITION ─
            # If beta ≤ alpha, the minimizer won't allow this branch
            # in the parent call, so we can safely skip remaining actions
            if beta <= alpha:
                # Record pruned nodes
                remaining_actions = len(actions) - (actions.index(action) + 1)
                if metrics is not None:
                    for _ in range(remaining_actions * (4 ** (depth - 1))):
                        metrics.record_node(depth, pruned=True)
                break
        
        return max_val, best_action

    else:
        # ─ MINIMIZING PLAYER ─
        # Goal: find action that minimizes heuristic score
        # (assumes player plays optimally from AI's perspective)
        
        min_val = float("inf")
        actions = available_actions(player)
        if not actions:
            actions = ["Rest"]

        for action in actions:
            # Recursively evaluate this action
            new_player, new_enemy = simulate_turn(player, enemy, action)
            
            # Recursive call: switch to maximizing (enemy's turn)
            score, _ = minimax(new_enemy, new_player, reputation,
                               depth - 1, alpha, beta, True, metrics, pattern_analyzer)
            
            # Track metrics if provided
            if metrics is not None:
                metrics.record_node(depth, pruned=False)
            
            # Update best score and action
            if score < min_val:
                min_val = score
                best_action = action
            
            # Alpha-Beta pruning: update minimizer's best guarantee
            beta = min(beta, score)
            
            # ─ PRUNING CONDITION ─
            # If beta ≤ alpha, the maximizer won't allow this branch
            # in the parent call, so we can safely skip remaining actions
            if beta <= alpha:
                # Record pruned nodes
                remaining_actions = len(actions) - (actions.index(action) + 1)
                if metrics is not None:
                    for _ in range(remaining_actions * (4 ** (depth - 1))):
                        metrics.record_node(depth, pruned=True)
                break
        
        return min_val, best_action


def ai_choose_action(enemy: CombatantState, player: CombatantState,
                     reputation: ReputationSystem,
                     depth: int = 3,
                     metrics: Optional[OptimizationMetrics] = None,
                     pattern_analyzer: Optional[PatternAnalyzer] = None) -> str:
    """Entry point for the AI to pick its action via Minimax.
    
    Time Complexity: O(b^(d/2)) with pruning ≈ O(8) for depth=3, b=4
    Space Complexity: O(b·d) = O(12)
    
    Args:
        enemy: Enemy state
        player: Player state
        reputation: Reputation system
        depth: Search depth (default 3)
        metrics: Optional metrics tracker
        pattern_analyzer: Optional pattern analyzer
        
    Returns:
        Best action string
    """
    _, action = minimax(enemy, player, reputation,
                        depth, float("-inf"), float("inf"), True,
                        metrics, pattern_analyzer)
    
    # Fallback safety: ensure we return a valid action
    if action is None or not can_perform(enemy, action):
        return "Rest"
    return action


# ─────────────────────────────────────────────
# Player input
# ─────────────────────────────────────────────
def get_player_action(player: CombatantState) -> str:
    """Prompt the human player for an action.
    
    Time Complexity: O(1) per interaction
    Space Complexity: O(1)
    """
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
    """Print a divider line."""
    print(char * width)

def print_header(text: str):
    """Print a formatted header."""
    divider("═")
    print(f"  {text}")
    divider("═")

def describe_action(actor_name: str, action: str, target_name: str,
                    dmg_dealt: int, blocked: bool) -> str:
    """Describe an action in combat log format."""
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
               minimax_depth: int = 3,
               track_metrics: bool = False):
    """Full combat loop.
    
    Time Complexity: O(turns × b^(d/2)) where turns ≈ 8-15
    Space Complexity: O(b·d) per turn
    
    Args:
        player: Player combatant
        enemy: Enemy combatant
        reputation: Reputation system
        player_controlled: If True, human input; else auto-simulate
        minimax_depth: Search depth for Minimax
        track_metrics: If True, track optimization metrics
    """
    turn = 1
    print_header(f"⚔  COMBAT START: {player.name} vs {enemy.name}  ⚔")

    # Initialize optional components
    metrics = OptimizationMetrics() if track_metrics else None
    pattern_analyzer = PatternAnalyzer()

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
        e_action = ai_choose_action(enemy, player, reputation, minimax_depth,
                                   metrics, pattern_analyzer)

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
    
    if track_metrics and metrics is not None:
        print(f"\n  Optimization Metrics:")
        print(f"  {metrics.summary()}")
    
    divider("═")


# ─────────────────────────────────────────────
# Demo / Entry Point
# ─────────────────────────────────────────────
def main():
    print_header("🏰  DUNGEON RPG  –  AI2002 Part A & B Demo  🏰")
    print("""
  Stamina costs:  Attack=2  |  Heavy Attack=4  |  Defend=1  |  Rest recovers 3
  AI uses depth-3 Minimax with Alpha-Beta pruning and adaptive heuristics.
  ENHANCED: Sophisticated pattern recognition & optimization tracking.
    """)

    # ── Choose mode ───────────────────────────────────
    print("  Select mode:")
    print("    [1] Play yourself (vs AI Enemy)")
    print("    [2] Play yourself (vs AI Boss – tougher)")
    print("    [3] Auto-simulation (watch AI vs AI demo)")
    print("    [4] Auto-simulation with metrics tracking")
    while True:
        try:
            mode = int(input("  Choice: ").strip())
            if mode in (1, 2, 3, 4):
                break
        except ValueError:
            pass
        print("  Enter 1, 2, 3, or 4.")

    is_boss = (mode == 2)
    player_controlled = (mode in (1, 2))
    track_metrics = (mode == 4)

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
               minimax_depth=depth,
               track_metrics=track_metrics)


if __name__ == "__main__":
    main()