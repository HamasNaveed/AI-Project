# 🏰 Dungeon RPG with Minimax AI

A 2D turn-based dungeon RPG implementing **Minimax with Alpha-Beta pruning**, enhanced with pattern recognition and optimization metrics tracking.

---

## 🚀 Features

- **Turn-based combat** (Player vs AI Enemy/Boss)
- **Stamina-based action system:**
  - Attack (cost: 2)
  - Heavy Attack (cost: 4)
  - Defend (cost: 1)
  - Rest (recovers 3)
- **AI powered by:**
  - Minimax (depth-limited)
  - Alpha-Beta pruning
- **Advanced AI enhancements:**
  - Pattern recognition (frequency, cyclic, entropy)
  - Predictability scoring
  - Reputation-based decision making
- **Boss Rush mode** with branching paths
- **Performance tracking:**
  - Nodes evaluated
  - Nodes pruned
  - Pruning efficiency

---

## 📂 Project Structure

```
.
├── dungeon_rpg_ai_enhanced.py   # Main game + AI logic
├── test_dungeon_rpg.py          # Unit & integration tests
├── requirements.txt             # Dependencies
└── README.md                    # Documentation
```

---

## ⚙️ Installation

### 1. Clone or Download Project

```bash
git clone <your-repo-url>
cd dungeon-rpg-ai
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ How to Run

### Run the Game

```bash
python dungeon_rpg_ai_enhanced.py
```

### Game Modes

| # | Mode |
|---|------|
| 1 | Play vs AI |
| 2 | Play vs Bosses |
| 3 | AI vs AI Simulation |
| 4 | AI Simulation with Metrics |
| 5 | AI Hard Mode (Boss Rush) |

---

## 🧪 Run Tests

```bash
python test_dungeon_rpg.py
```

This will run:
- Unit tests
- Integration tests
- Performance benchmarks

---

## 🧠 AI & Algorithm Details

### Minimax Algorithm

- Depth-limited search (default depth = 3)
- Evaluates best move assuming optimal opponent

### Alpha-Beta Pruning

- Eliminates unnecessary branches
- Improves performance from:
  - `O(b^d)` → `O(b^(d/2))`

### Heuristic Function Factors

- HP difference
- Stamina advantage
- Reputation system
- Pattern recognition
- Predictability score

---

## 📊 Pattern Recognition

The AI analyzes player behavior using:

- Frequency distribution
- Repeating patterns
- Cyclic patterns
- Shannon entropy (predictability)
- Recency-weighted actions

---

## 📈 Optimization Metrics

Tracks:

- Nodes evaluated
- Nodes pruned
- Pruning efficiency (%)
- Depth distribution

---

## 🛠 Requirements

- **Python 3.8+**
- No external dependencies (uses Python standard library only)

---

## 👨‍💻 Author

**Hamas Naveed**
**Usman Shaukat**
**Muhammad Umer**
