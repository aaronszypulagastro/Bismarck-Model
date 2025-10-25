# 🤖 Bismarck-Modell - Intelligent Trading System

> *"Aus Fehlern lernen kann jeder, aber aus den Fehlern anderer zu lernen erfordert Intelligenz."* - Otto von Bismarck

## 📋 Übersicht

Das **Bismarck-Modell** ist ein intelligentes automatisiertes Trading-System, das auf dem Prinzip des **Fehlerlernens** basiert. Anstatt Emotionen zu simulieren, analysiert es historische Fehlermuster und vermeidet diese proaktiv.

### 🎯 Kernprinzipien

- **Anti-Mistake Learning**: Lernt aus historischen Trading-Fehlern
- **Collective Intelligence**: Nutzt LLM-basierte Marktanalyse
- **Risk Management**: Konservativer Ansatz mit 1.5% Risiko pro Trade
- **Real-Time Data**: Live-Marktdaten von Alpha Vantage, News API, Alpaca

## 🚀 Features

### ✅ Implementiert

- **Live Trading**: Automatisches Trading mit echtem Alpaca Account
- **Real-Time Analysis**: Live Marktdaten & News Sentiment
- **LLM Integration**: OpenAI für intelligente Finanzanalyse
- **Technical Indicators**: RSI, MACD, Bollinger Bands
- **Risk Management**: Position Sizing, Confidence Threshold
- **Performance Tracking**: Account Monitoring, Trade History

### 🔄 In Entwicklung

- Rate Limiting Optimization
- Advanced Pattern Recognition
- Multi-Strategy Support
- Backtesting Framework

## 📦 Installation

### Voraussetzungen

```bash
Python 3.8+
pip
```

### Installations-Schritte

1. **Repository klonen:**
```bash
cd src/DQN
```

2. **Dependencies installieren:**
```bash
pip install -r requirements.txt
```

3. **API Keys konfigurieren:**

Öffne `api_keys_config.py` und füge deine API Keys ein:

```python
# Alpaca Trading
ALPACA_API_KEY = "dein_api_key"
ALPACA_SECRET_KEY = "dein_secret_key"

# Alpha Vantage (kostenlos erhältlich)
ALPHA_VANTAGE_API_KEY = "dein_key"

# News API (kostenlos erhältlich)
NEWS_API_KEY = "dein_key"

# OpenAI (optional, für LLM Analysis)
OPENAI_API_KEY = "dein_key"
```

4. **Paper Trading Account erstellen:**

Besuche [alpaca.markets](https://alpaca.markets) und erstelle einen kostenlosen Paper Trading Account.

## 🎮 Verwendung

### Schnellstart

```bash
python bismarck_optimized_trading.py
```

### Konfiguration

Das System ist standardmäßig konfiguriert für:

- **Trading Interval**: 180 Sekunden (3 Minuten)
- **Risk per Trade**: 1.5% des Buy Power
- **Confidence Threshold**: 0.65 (65%)
- **Max Positions**: 4 gleichzeitige Positionen
- **Symbole**: AAPL, GOOGL, MSFT, TSLA

### Parameter anpassen

In `bismarck_optimized_trading.py`:

```python
class BismarckOptimizedTrading:
    def __init__(self):
        # Trading Parameter
        self.mistake_threshold = 0.65      # Confidence Threshold
        self.risk_per_trade = 0.015        # 1.5% Risk
        self.max_positions = 4             # Max Positions
        self.trading_interval = 180        # 3 Minuten
        
        # Rate Limiting
        self.alpha_vantage_calls_per_minute = 5
        self.news_api_calls_per_minute = 100
```

### Symbole ändern

```python
symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN', 'NVDA']
```

## 📊 Performance Monitoring

### Live Dashboard

Das System zeigt in Echtzeit:

```
💰 Account: $187,167.67 Buying Power, $99,998.71 Portfolio
📊 Positionen: 3

📈 Analysiere AAPL...
   - Empfehlung: HOLD
   - Confidence: 0.30
   - Preis: $262.82
   - Signal: 2.0

✅ KAUF-ORDER ERFOLGREICH! Trade #9
```

### Alpaca Dashboard

Besuche [app.alpaca.markets](https://app.alpaca.markets) für:
- Account Übersicht
- Positionen
- Trade History
- Performance Charts

## 🔧 Architektur

### Core Components

```
core/
├── anti_mistake_trading_engine.py   # Fehlererkennung & Vermeidung
├── bismarck_real_apis.py            # API Integration
├── llm_integration.py               # LLM-basierte Analyse
└── prioritized_replay_buffer.py     # Q-Learning Buffer
```

### Trading Flow

```mermaid
Marktdaten → Technische Analyse → News Sentiment → LLM Analysis
     ↓
Fehler-Check (Mistake Patterns) → Confidence Score
     ↓
Risk Check → Position Sizing → Trade Execution
```

## 📈 Beispiel-Output

### Erfolgreicher Trade

```
🔄 OPTIMIERTE ANALYSE - 10:30:08
----------------------------------------
💰 Account: $171,961.62 Buying Power, $99,931.57 Portfolio
📊 Positionen: 3

📈 Analysiere GOOGL...
   - Empfehlung: BUY
   - Confidence: 0.95
   - Preis: $259.92
   - Signal: 5.0

🟢 OPTIMIERTER KAUF: 9 Shares GOOGL @ $259.92
✅ KAUF-ORDER ERFOLGREICH! Trade #9
```

### Rate Limiting

```
⏳ Rate Limit erreicht, warte auf AAPL...
⏳ Warte 60 Sekunden...
```

## 🛡️ Risk Management

### Risikokontrolle

- **Position Sizing**: Max 1.5% des Buy Power pro Trade
- **Confidence Threshold**: Nur Trades mit >65% Confidence
- **Max Positions**: Maximal 4 gleichzeitige Positionen
- **Stop Loss**: Implizit durch Confidence-Based Exits

### Fehler-Muster Erkennung

Das System erkennt und vermeidet:

- ❌ **Overtrading**: Zu viele Trades in kurzer Zeit
- ❌ **FOMO Entry**: Einstieg aus Angst etwas zu verpassen
- ❌ **Panic Sell**: Verkauf aus Panik
- ❌ **Greed Hold**: Zu lange halten aus Gier
- ❌ **Correlation Bias**: Zu ähnliche Positionen
- ❌ **News FOMO**: Blind News-Kauf

## 🔍 API Limits

### Alpha Vantage (Free Tier)

- **Limit**: 5 Calls/Minute
- **Lösung**: Cache-System implementiert

### News API

- **Limit**: 100 Calls/Tag (Free)
- **Lösung**: Aggressives Caching

### OpenAI

- **Limit**: Variabel
- **Lösung**: Rate Limiting + Retry Logic

## 📝 Trading Logs

Alle Trades werden automatisch geloggt:

```
Trade #1: BUY 11 Shares GOOGL @ $253.08 (Confidence: 0.90)
Trade #2: BUY 6 Shares TSLA @ $448.98 (Confidence: 0.90)
Trade #3: BUY 10 Shares GOOGL @ $259.92 (Confidence: 0.95)
...
```

## 🐛 Troubleshooting

### "Keine Analyse für [Symbol]"

**Problem**: Rate Limit erreicht

**Lösung**: 
- Warte 60 Sekunden
- Oder reduziere Anzahl Symbole
- Oder erhöhe Trading Interval

### "API Key ungültig"

**Problem**: Falsche API Keys

**Lösung**: Überprüfe `api_keys_config.py`

### "KAUF-ORDER FEHLGESCHLAGEN"

**Problem**: Insufficient Buying Power

**Lösung**: 
- Reduziere `risk_per_trade`
- Oder reduziere `max_positions`

## 📚 Weiterführende Informationen

### Dokumentation

- [Alpaca API Docs](https://alpaca.markets/docs/)
- [Alpha Vantage Docs](https://www.alphavantage.co/documentation/)
- [News API Docs](https://newsapi.org/docs)

### Trading Grundlagen

- [Position Sizing](https://www.investopedia.com/articles/trading/04/092904.asp)
- [Risk Management](https://www.investopedia.com/articles/trading/09/risk-management.asp)
- [Technical Indicators](https://www.investopedia.com/technical-analysis-4689657)

## ⚠️ Disclaimer

**WICHTIG**: Dieses System ist für **Paper Trading** (Demo Account) konzipiert. 

- **Keine Finanzberatung**: Alle Trades auf eigene Gefahr
- **Backtesting**: Immer erst mit historischen Daten testen
- **Real Trading**: Nur mit Beträgen, die man verlieren kann
- **Regulierung**: Beachte lokale Trading-Regulierungen

## 🤝 Contributing

Beiträge sind willkommen! Bitte:

1. Fork das Repository
2. Erstelle einen Feature Branch
3. Commit deine Änderungen
4. Push zum Branch
5. Erstelle einen Pull Request

## 📄 Lizenz

MIT License - Siehe LICENSE Datei

## 👤 Autor

**Aaron Szypul** - Emotion-Augmented NN Project

## 🙏 Danksagungen

- Alpaca für Paper Trading API
- Alpha Vantage für Market Data
- News API für Financial News
- OpenAI für LLM Capabilities

---

## 📞 Support

Bei Fragen oder Problemen:

1. Öffne ein GitHub Issue
2. Schaue in die bestehenden Issues
3. Kontaktiere den Maintainer

**Viel Erfolg beim Trading! 🚀**
