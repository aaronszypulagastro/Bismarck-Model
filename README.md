# 🤖 Bismarck Trading System

> **Intelligentes Trading-System mit LLM & Anti-Mistake Engine**

## 📋 Überblick

Bismarck ist ein produktionsreifes intelligentes Trading-System, das:
- ✅ LLM-Integration für Datenanalyse
- ✅ Anti-Mistake Learning Engine
- ✅ Rate Limiting & Caching
- ✅ Echtzeit-Trading mit Alpaca API

## 🚀 Quick Start

### Installation

```bash
# Virtuelle Umgebung
python -m venv .venv
.venv\Scripts\activate

# Dependencies
pip install -r requirements.txt
```

### Setup

1. **API Keys** in `api_keys_config.py` eintragen
2. **Konfiguration** anpassen
3. **Trading starten**:

```bash
python bismarck_optimized_trading.py
```

## 📁 Struktur

```
bismarck-trading/
├── core/
│   ├── anti_mistake_trading_engine.py
│   ├── llm_integration.py
│   └── bismarck_real_apis.py
├── bismarck_factor_factory/
│   ├── rate_limit.py
│   └── cache.py
├── tests/
├── bismarck_optimized_trading.py
└── README_v2.md
```

## 🎯 Features

- **LLM Integration**: OpenAI GPT für Sentiment Analysis
- **Anti-Mistake Learning**: Lerne aus historischen Fehlern
- **Rate Limiting**: TokenBucket + Async Queue
- **Caching**: SQLite-basierter Cache mit TTL
- **Real Trading**: Alpaca API Integration

## 📚 Dokumentation

- **Architektur**: Siehe `README_v2.md`
- **Dev Setup**: Siehe `DEV_SETUP.md`
- **Struktur**: Siehe `PROJECT_STRUCTURE.md`

## 📄 License

MIT License
