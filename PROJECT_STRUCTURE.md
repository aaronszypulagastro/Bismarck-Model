# 📁 Projektstruktur - Emotion-augmented-nn

## 🗂️ Übersicht

Dieses Repository enthält **zwei separate Projekte**:

1. **Bismarck Trading System** (Trading mit LLM + Anti-Mistake Engine)
2. **Emotion-Augmented DQN** (RL mit Emotion Engine für LunarLander)

---

## 🤖 **BISMARCK PROJEKT** (Trading System)

### Core Files
```
bismarck_optimized_trading.py          # Main Trading Script
api_keys_config.py                     # API Keys
BISMARCK_README.md                     # Bismarck Dokumentation
README_v2.md                           # Bismarck v2 Architektur
DEV_SETUP.md                           # Dev Setup Guide
```

### Bismarc-spezifische Core Module
```
core/
├── anti_mistake_trading_engine.py     # Anti-Mistake Engine
├── llm_integration.py                 # LLM Integration
└── bismarck_real_apis.py              # Real API Integration
```

### Factor Factory (Bismarck v2)
```
bismarck_factor_factory/
├── __init__.py
├── rate_limit.py                      # TokenBucket + Rate Limiter
├── cache.py                           # SQLite Cache
├── scorer.py                          # [TODO] 3-Stage Scorer
└── dedupe.py                          # [TODO] Deduplication
```

### Tests (Bismarck)
```
tests/
├── test_cache.py                      # Cache Tests
├── test_rate_limit.py                 # Rate Limit Tests
└── test_scorer.py                     # [TODO] Scorer Tests
```

### Config (Bismarck)
```
configs/
└── lunarlander_enhanced_config.py     # LunarLander Config (misplaced?)
```

### Results (Bismarck Trading)
```
results/
├── *trading*.png                      # Trading Plots
├── regional/                          # Regional Trading Results
└── analysis/                          # Trading Analysis
```

---

## 🎮 **EMOTION-AUGMENTED DQN PROJEKT** (RL System)

### Core RL Components
```
core/
├── dueling_network.py                 # Dueling DQN Network
├── prioritized_replay_buffer.py       # Prioritized Experience Replay
├── enhanced_emotion_engine.py         # Enhanced Emotion Engine
├── competitive_emotion_engine.py      # Competitive Emotion Engine
├── rainbow_dqn_agent.py               # Rainbow DQN Agent
└── infrastructure_profile.py          # Infrastructure Profiling
```

### Training Scripts (DQN)
```
scripts/
├── colab_lunarlander_enhanced_emotion_fixed.py
├── colab_lunarlander_enhanced_emotion_v2.py
├── colab_lunarlander_enhanced_emotion.py
├── colab_lunarlander_hyperparameter_optimized.py
├── lunarlander_cpu_optimized.py
└── colab_meta_learning_complete.py
```

### Analysis & Visualization (DQN)
```
analysis/
├── compare_all_systems.py
├── compare_multi_environment.py
├── create_final_report.py
├── plot_utils.py
├── statistical_analysis.py
├── visualize_competitive.py
├── visualize_rainbow_comparison.py
└── visualize_regional_comparison.py
```

### Training Configs
```
training/
├── train_acrobot_regional.py
├── train_rainbow_universal.py
└── train_regional_infrastructure.py
```

### Models & Results (DQN)
```
models/                                # Saved RL Models
results/
├── results/                           # DQN Training Results
├── results_new/                       # New DQN Results
└── analysis/                          # DQN Analysis Plots
```

---

## 📄 Shared Files

```
requirements.txt                       # Shared Dependencies
.gitignore                            # Git Ignore Rules
CONTRIBUTING.md                       # Contributing Guidelines
README.md                             # Main README (DQN-focused)
__init__.py                           # Package Init
```

---

## 🔄 Empfohlene Projektstruktur

### Option 1: Separate Repositories
```
emotion-augmented-nn/                  # DQN Repository
├── core/ (emotion, rainbow, dueling)
├── scripts/ (lunarlander scripts)
├── analysis/ (DQN analysis)
└── training/ (RL training)

bismarck-trading/                      # Bismarck Repository  
├── core/ (anti_mistake, llm_integration, APIs)
├── bismarck_factor_factory/
├── tests/
├── configs/
└── results/ (trading results)
```

### Option 2: Subdirectories im gleichen Repo
```
emotion-augmented-nn/
├── bismarck/                          # Bismarck Projekt
│   ├── core/
│   ├── factor_factory/
│   ├── tests/
│   └── configs/
│
└── dqn/                               # DQN Projekt
    ├── core/
    ├── scripts/
    ├── analysis/
    └── training/
```

---

## 🎯 Aktuelle Mischungen (zu bereinigen)

❌ **Problem**: Bismarck und DQN teilen sich:
- `core/` Directory (gemischt)
- `results/` Directory (gemischt)  
- `configs/` Directory (gemischt)
- Root-Level Dateien

✅ **Lösung**: Siehe "Empfohlene Projektstruktur" oben
