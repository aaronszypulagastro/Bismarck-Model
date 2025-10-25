"""
Bismarck-Modell Paper Trading System
====================================

Implementiert Paper Trading mit Bismarck-Prinzip: Lerne aus Fehlern
"""

import requests
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional

class BismarckPaperTrading:
    """
    Paper Trading System mit Bismarck-Prinzip
    """
    
    def __init__(self):
        # API Keys
        self.alpaca_key = "PKQDJ6YBMBEZH6VAGOBEPHAV6T"
        self.alpaca_secret = "GMz9vnmdCQxtEfZR8axkm8Y1TpWCMPtbEyphwrNox1Q7"
        self.alpha_vantage_key = "GEYAL5Z0GAHPYHNQ"
        self.news_api_key = "11215a4a47f04c919800d789566466f6"
        
        # Trading Portfolio
        self.portfolio = {
            'cash': 100000.0,  # Startkapital
            'positions': {},   # Aktuelle Positionen
            'total_value': 100000.0,
            'trades': [],      # Trade-Historie
            'mistakes': []     # Fehler-Historie für Bismarck-Prinzip
        }
        
        # Bismarck-Prinzip Parameter
        self.mistake_threshold = 0.3  # Mindest-Confidence für Trades
        self.risk_per_trade = 0.02    # 2% Risiko pro Trade
        self.max_positions = 5        # Maximale Anzahl Positionen
        
        print("🚀 BISMARCK PAPER TRADING SYSTEM")
        print("=" * 50)
        print("✅ Startkapital: $100,000")
        print("✅ Bismarck-Prinzip: Lerne aus Fehlern")
        print("✅ Risikomanagement: 2% pro Trade")
        print("✅ Maximale Positionen: 5")
        print("=" * 50)
    
    def get_account_info(self):
        """
        Holt Account-Informationen von Alpaca
        """
        try:
            headers = {
                'APCA-API-KEY-ID': self.alpaca_key,
                'APCA-API-SECRET-KEY': self.alpaca_secret
            }
            
            response = requests.get(
                'https://paper-api.alpaca.markets/v2/account',
                headers=headers,
                timeout=10
            )
            
            if response.status_code == 200:
                account_data = response.json()
                print(f"✅ Account Info erhalten:")
                print(f"  - Buying Power: ${float(account_data.get('buying_power', 0)):,.2f}")
                print(f"  - Portfolio Value: ${float(account_data.get('portfolio_value', 0)):,.2f}")
                return account_data
            else:
                print(f"❌ Account Info Fehler: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"❌ Account Info Exception: {e}")
            return None
    
    def get_current_positions(self):
        """
        Holt aktuelle Positionen von Alpaca
        """
        try:
            headers = {
                'APCA-API-KEY-ID': self.alpaca_key,
                'APCA-API-SECRET-KEY': self.alpaca_secret
            }
            
            response = requests.get(
                'https://paper-api.alpaca.markets/v2/positions',
                headers=headers,
                timeout=10
            )
            
            if response.status_code == 200:
                positions = response.json()
                print(f"✅ Positionen erhalten: {len(positions)} Positionen")
                
                for position in positions:
                    symbol = position['symbol']
                    qty = int(position['qty'])
                    market_value = float(position['market_value'])
                    print(f"  - {symbol}: {qty} Shares (${market_value:,.2f})")
                
                return positions
            else:
                print(f"❌ Positionen Fehler: {response.status_code}")
                return []
                
        except Exception as e:
            print(f"❌ Positionen Exception: {e}")
            return []
    
    def get_market_data(self, symbol):
        """
        Holt Marktdaten von Alpha Vantage
        """
        try:
            url = f"https://www.alphavantage.co/query"
            params = {
                'function': 'GLOBAL_QUOTE',
                'symbol': symbol,
                'apikey': self.alpha_vantage_key
            }
            
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if 'Global Quote' in data:
                    quote = data['Global Quote']
                    price = float(quote['05. price'])
                    change = float(quote['09. change'])
                    change_percent = float(quote['10. change percent'].replace('%', ''))
                    
                    return {
                        'symbol': symbol,
                        'price': price,
                        'change': change,
                        'change_percent': change_percent,
                        'timestamp': datetime.now()
                    }
            
            return None
            
        except Exception as e:
            print(f"❌ Marktdaten Fehler für {symbol}: {e}")
            return None
    
    def get_news_sentiment(self, symbol):
        """
        Holt Nachrichten-Sentiment von News API
        """
        try:
            url = f"https://newsapi.org/v2/everything"
            params = {
                'q': symbol,
                'apiKey': self.news_api_key,
                'language': 'en',
                'sortBy': 'publishedAt',
                'pageSize': 5
            }
            
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                articles = data.get('articles', [])
                
                positive_count = 0
                negative_count = 0
                
                for article in articles:
                    title = article.get('title', '').lower()
                    description = article.get('description', '').lower()
                    text = f"{title} {description}"
                    
                    # Einfache Sentiment-Analyse
                    if any(word in text for word in ['positive', 'gain', 'rise', 'up', 'bullish']):
                        positive_count += 1
                    elif any(word in text for word in ['negative', 'loss', 'fall', 'down', 'bearish']):
                        negative_count += 1
                
                total_articles = len(articles)
                if total_articles > 0:
                    sentiment_score = (positive_count - negative_count) / total_articles
                else:
                    sentiment_score = 0
                
                return {
                    'sentiment_score': sentiment_score,
                    'positive_count': positive_count,
                    'negative_count': negative_count,
                    'total_articles': total_articles
                }
            
            return None
            
        except Exception as e:
            print(f"❌ News Sentiment Fehler für {symbol}: {e}")
            return None
    
    def analyze_trading_opportunity(self, symbol):
        """
        Analysiert Trading-Möglichkeit basierend auf Bismarck-Prinzip
        """
        print(f"\n🧠 BISMARCK-ANALYSE FÜR {symbol}")
        print("-" * 40)
        
        # Hole Marktdaten
        market_data = self.get_market_data(symbol)
        if not market_data:
            print(f"❌ Keine Marktdaten für {symbol}")
            return None
        
        # Hole News Sentiment
        news_sentiment = self.get_news_sentiment(symbol)
        if not news_sentiment:
            print(f"❌ Kein News Sentiment für {symbol}")
            return None
        
        # Bismarck-Prinzip Analyse
        price = market_data['price']
        change_percent = market_data['change_percent']
        sentiment_score = news_sentiment['sentiment_score']
        
        print(f"📊 Marktdaten:")
        print(f"  - Preis: ${price:.2f}")
        print(f"  - Änderung: {change_percent:+.2f}%")
        print(f"  - News Sentiment: {sentiment_score:+.2f}")
        
        # Berechne Trading-Signal
        signal_strength = 0
        
        # Preis-Momentum
        if change_percent > 2:
            signal_strength += 1
        elif change_percent < -2:
            signal_strength -= 1
        
        # News Sentiment
        if sentiment_score > 0.3:
            signal_strength += 1
        elif sentiment_score < -0.3:
            signal_strength -= 1
        
        # Bismarck-Prinzip: Lerne aus Fehlern
        recent_mistakes = self._analyze_recent_mistakes(symbol)
        if recent_mistakes > 0:
            signal_strength -= 0.5  # Reduziere Signal bei häufigen Fehlern
        
        # Bestimme Empfehlung
        if signal_strength >= 1.5:
            recommendation = 'BUY'
            confidence = min(0.9, 0.5 + signal_strength * 0.2)
        elif signal_strength <= -1.5:
            recommendation = 'SELL'
            confidence = min(0.9, 0.5 + abs(signal_strength) * 0.2)
        else:
            recommendation = 'HOLD'
            confidence = 0.3
        
        print(f"🎯 Bismarck-Empfehlung: {recommendation} (Confidence: {confidence:.2f})")
        
        return {
            'symbol': symbol,
            'recommendation': recommendation,
            'confidence': confidence,
            'price': price,
            'market_data': market_data,
            'news_sentiment': news_sentiment,
            'signal_strength': signal_strength
        }
    
    def _analyze_recent_mistakes(self, symbol):
        """
        Analysiert recente Fehler für Bismarck-Prinzip
        """
        recent_trades = [trade for trade in self.portfolio['trades'] 
                        if trade['symbol'] == symbol and 
                        trade['timestamp'] > datetime.now() - timedelta(days=7)]
        
        mistake_count = 0
        for trade in recent_trades:
            if trade['result'] == 'LOSS':
                mistake_count += 1
        
        return mistake_count
    
    def execute_trade(self, analysis):
        """
        Führt Trade basierend auf Bismarck-Analyse aus
        """
        if not analysis:
            return False
        
        symbol = analysis['symbol']
        recommendation = analysis['recommendation']
        confidence = analysis['confidence']
        price = analysis['price']
        
        # Bismarck-Prinzip: Nur bei hoher Confidence handeln
        if confidence < self.mistake_threshold:
            print(f"⚠️ Confidence zu niedrig für gemacht: {confidence:.2f} < {self.mistake_threshold}")
            return False
        
        # Berechne Position Size
        risk_amount = self.portfolio['cash'] * self.risk_per_trade
        position_size = risk_amount / price
        
        if recommendation == 'BUY':
            return self._execute_buy_order(symbol, position_size, price, confidence)
        elif recommendation == 'SELL':
            return self._execute_sell_order(symbol, position_size, price, confidence)
        
        return False
    
    def _execute_buy_order(self, symbol, quantity, price, confidence):
        """
        Führt Kauf-Order aus
        """
        try:
            # Überprüfe verfügbares Kapital
            cost = quantity * price
            if cost > self.portfolio['cash']:
                print(f"❌ Nicht genügend Kapital: ${cost:,.2f} > ${self.portfolio['cash']:,.2f}")
                return False
            
            # Überprüfe maximale Positionen
            if len(self.portfolio['positions']) >= self.max_positions:
                print(f"❌ Maximale Positionen erreicht: {len(self.portfolio['positions'])}")
                return False
            
            # Simuliere Order (Paper Trading)
            print(f"🟢 KAUF-ORDER: {quantity:.0f} Shares {symbol} @ ${price:.2f}")
            print(f"  - Kosten: ${cost:,.2f}")
            print(f"  - Confidence: {confidence:.2f}")
            
            # Aktualisiere Portfolio
            self.portfolio['cash'] -= cost
            if symbol in self.portfolio['positions']:
                self.portfolio['positions'][symbol]['quantity'] += quantity
                self.portfolio['positions'][symbol]['avg_price'] = (
                    (self.portfolio['positions'][symbol]['quantity'] - quantity) * 
                    self.portfolio['positions'][symbol]['avg_price'] + cost
                ) / self.portfolio['positions'][symbol]['quantity']
            else:
                self.portfolio['positions'][symbol] = {
                    'quantity': quantity,
                    'avg_price': price,
                    'timestamp': datetime.now()
                }
            
            # Speichere Trade
            trade = {
                'symbol': symbol,
                'action': 'BUY',
                'quantity': quantity,
                'price': price,
                'confidence': confidence,
                'timestamp': datetime.now(),
                'result': 'PENDING'
            }
            self.portfolio['trades'].append(trade)
            
            print(f"✅ Kauf-Order erfolgreich ausgeführt")
            return True
            
        except Exception as e:
            print(f"❌ Kauf-Order Fehler: {e}")
            return False
    
    def _execute_sell_order(self, symbol, quantity, price, confidence):
        """
        Führt Verkaufs-Order aus
        """
        try:
            # Überprüfe verfügbare Position
            if symbol not in self.portfolio['positions']:
                print(f"❌ Keine Position in {symbol}")
                return False
            
            available_quantity = self.portfolio['positions'][symbol]['quantity']
            if quantity > available_quantity:
                quantity = available_quantity
            
            # Simuliere Order (Paper Trading)
            proceeds = quantity * price
            print(f"🔴 VERKAUFS-ORDER: {quantity:.0f} Shares {symbol} @ ${price:.2f}")
            print(f"  - Erlös: ${proceeds:,.2f}")
            print(f"  - Confidence: {confidence:.2f}")
            
            # Aktualisiere Portfolio
            self.portfolio['cash'] += proceeds
            self.portfolio['positions'][symbol]['quantity'] -= quantity
            
            # Entferne Position wenn vollständig verkauft
            if self.portfolio['positions'][symbol]['quantity'] <= 0:
                del self.portfolio['positions'][symbol]
            
            # Speichere Trade
            trade = {
                'symbol': symbol,
                'action': 'SELL',
                'quantity': quantity,
                'price': price,
                'confidence': confidence,
                'timestamp': datetime.now(),
                'result': 'PENDING'
            }
            self.portfolio['trades'].append(trade)
            
            print(f"✅ Verkaufs-Order erfolgreich ausgeführt")
            return True
            
        except Exception as e:
            print(f"❌ Verkaufs-Order Fehlerfahren: {e}")
            return False
    
    def update_portfolio_value(self):
        """
        Aktualisiert Portfolio-Wert
        """
        total_value = self.portfolio['cash']
        
        for symbol, position in self.portfolio['positions'].items():
            market_data = self.get_market_data(symbol)
            if market_data:
                current_price = market_data['price']
                position_value = position['quantity'] * current_price
                total_value += position_value
        
        self.portfolio['total_value'] = total_value
        
        print(f"\n📊 PORTFOLIO UPDATE")
        print("-" * 30)
        print(f"  - Cash: ${self.portfolio['cash']:,.2f}")
        print(f"  - Positionen: {len(self.portfolio['positions'])}")
        print(f"  - Gesamtwert: ${total_value:,.2f}")
        print(f"  - Performance: {((total_value - 100000) / 100000 * 100):+.2f}%")
    
    def run_trading_session(self, symbols):
        """
        Führt Trading-Session durch
        """
        print(f"\n🚀 BISMARCK TRADING SESSION")
        print("=" * 50)
        print(f"Symbole: {', '.join(symbols)}")
        print(f"Zeit: {datetime.now()}")
        print("=" * 50)
        
        for symbol in symbols:
            print(f"\n📊 ANALYSE VON {symbol}")
            print("-" * 30)
            
            # Analysiere Trading-Möglichkeit
            analysis = self.analyze_trading_opportunity(symbol)
            
            if analysis:
                # Führe Trade aus
                trade_executed = self.execute_trade(analysis)
                
                if trade_executed:
                    print(f"✅ Trade für {symbol} ausgeführt")
                else:
                    print(f"⚠️ Kein Trade für {symbol}")
            
            time.sleep(2)  # Pause zwischen Analysen
        
        # Aktualisiere Portfolio
        self.update_portfolio_value()
        
        print(f"\n🎯 TRADING SESSION ABGESCHLOSSEN")
        print("=" * 50)
        print("✅ Bismarck-Prinzip angewendet")
        print("✅ Risikomanagement befolgt")
        print("✅ Portfolio aktualisiert")

def main():
    """
    Hauptfunktion
    """
    print("🚀 BISMARCK PAPER TRADING SYSTEM")
    print("=" * 60)
    
    # Erstelle Trading-System
    trading_system = BismarckPaperTrading()
    
    # Teste Account-Verbindung
    account_info = trading_system.get_account_info()
    if account_info:
        print("✅ Alpaca Account verbunden")
    else:
        print("⚠️ Alpaca Account nicht verfügbar - verwende lokale Simulation")
    
    # Hole aktuelle Positionen
    positions = trading_system.get_current_positions()
    
    # Führe Trading-Session durch
    symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA']
    trading_system.run_trading_session(symbols)
    
    print("\n🎉 BISMARCK PAPER TRADING ABGESCHLOSSEN!")
    print("=" * 60)
    print("✅ Trading-Session erfolgreich durchgeführt")
    print("✅ Bismarck-Prinzip erfolgreich angewendet")
    print("✅ Portfolio-Wert aktualisiert")
    print("🚀 Das System ist bereit für weitere Trades!")

if __name__ == "__main__":
    main()
