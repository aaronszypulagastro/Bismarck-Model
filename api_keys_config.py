"""
API-Keys Konfiguration für Bismarck-Modell
==========================================

Hier kannst du deine API-Keys eintragen.
"""

# =============================================================================
# KOSTENLOSE APIs (Empfohlen für Start)
# =============================================================================

# Alpha Vantage API (kostenlos)
# Registrierung: https://www.alphavantage.co/support/#api-key
# 5 Requests pro Minute kostenlos
ALPHA_VANTAGE_API_KEY = "your_alpha_vantage_key_here"

# News API (kostenlos)
# Registrierung: https://newsapi.org/register
# 1000 Requests pro Monat kostenlos
NEWS_API_KEY = "your_news_api_key_here"

# Alpaca Paper Trading (kostenlos)
# Registrierung: https://alpaca.markets/
# Virtuelles Trading kostenlos
ALPACA_API_KEY = "your_alpaca_key_here"
ALPACA_SECRET_KEY = "your_alpaca_secret_here"

# =============================================================================
# KOSTENPFLICHTIGE APIs (Für bessere Performance)
# =============================================================================

# OpenAI API (kostenpflichtig)
# Registrierung: https://platform.openai.com/api-keys
# ~$0.03 pro 1K Tokens
OPENAI_API_KEY = "your_openai_key_here"

# Anthropic API (kostenpflichtig)
# Registrierung: https://console.anthropic.com/
# ~$0.015 pro 1K Tokens
ANTHROPIC_API_KEY = "your_anthropic_key_here"

# =============================================================================
# SETUP ANWEISUNGEN
# =============================================================================

"""
SCHRITT-FÜR-SCHRITT ANWEISUNGEN:

1. 📊 ALPHA VANTAGE (Kostenlos - EMPFOHLEN):
   - Gehe zu: https://www.alphavantage.co/support/#api-key
   - Klicke auf "Get Free API Key"
   - Registriere dich mit deiner E-Mail
   - Erhalte deinen API-Key
   - Ersetze "your_alpha_vantage_key_here" mit deinem Key

2. 📰 NEWS API (Kostenlos - EMPFOHLEN):
   - Gehe zu: https://newsapi.org/register
   - Registriere dich kostenlos
   - Erhalte deinen API-Key
   - Ersetze "your_news_api_key_here" mit deinem Key

3. 💰 ALPACA (Paper Trading - Kostenlos - EMPFOHLEN):
   - Gehe zu: https://alpaca.markets/
   - Klicke auf "Get Started"
   - Registriere dich für Paper Trading
   - Erhalte API Key und Secret Key
   - Ersetze "your_alpaca_key_here" und "your_alpaca_secret_here"

4. 🧠 OPENAI (Kostenpflichtig - Optional):
   - Gehe zu: https://platform.openai.com/api-keys
   - Erstelle einen Account
   - Füge Kreditkarte hinzu
   - Erstelle API-Key
   - Ersetze "your_openai_key_here" mit deinem Key

5. 🧠 ANTHROPIC (Kostenpflichtig - Optional):
   - Gehe zu: https://console.anthropic.com/
   - Erstelle einen Account
   - Füge Kreditkarte hinzu
   - Erstelle API-Key
   - Ersetze "your_anthropic_key_here" mit deinem Key

NACH DEM SETUP:
- Teste die Keys: python test_api_keys.py
- Starte Demo: python scripts/bismarck_real_apis_demo.py
"""
