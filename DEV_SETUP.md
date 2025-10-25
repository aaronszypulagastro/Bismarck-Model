# 🔧 Bismarck v2 - Development Setup

## Virtuelle Umgebung

```bash
# Windows PowerShell
python -m venv .venv_v2
.venv_v2\Scripts\python.exe -m pip install -U pip setuptools wheel

# Dependencies installieren
.venv_v2\Scripts\python.exe -m pip install pytest pytest-asyncio

# Optional: Für spätere Features
.venv_v2\Scripts\python.exe -m pip install duckdb xgboost
```

## Tests ausführen

```bash
# Alle Tests
.venv_v2\Scripts\python.exe -m pytest tests/ -v

# Spezifische Tests
.venv_v2\Scripts\python.exe -m pytest tests/test_rate_limit.py -v
.venv_v2\Scripts\python.exe -m pytest tests/test_cache.py -v

# Mit Coverage
.venv_v2\Scripts\python.exe -m pytest tests/ --cov=bismarck_factor_factory
```

## Aktueller Status

✅ **Implementiert:**
- Rate Limiting (TokenBucket + AsyncRateLimiter)
- Cache Manager (SQLite mit TTL)
- Unit Tests (7/9 passing)

⚠️ **In Entwicklung:**
- Scorer (3-Stage Pipeline)
- Config System
- Strategies
- Backtest Runner

## VS Code Interpreter

1. `Ctrl+Shift+P` → "Python: Select Interpreter"
2. Wähle: `.venv_v2\Scripts\python.exe`

## Troubleshooting

**ModuleNotFoundError**: Stell sicher dass `.venv_v2` aktiv ist
**PermissionError**: PowerShell als Admin starten
**ImportError**: `PYTHONPATH` auf Repository-Root setzen
