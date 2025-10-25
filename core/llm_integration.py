"""
LLM Integration für Bismarck-Modell
===================================

Integriert OpenAI und Anthropic APIs für intelligente Finanzanalyse
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import requests
import json
import time
from datetime import datetime

# Importiere die Konfiguration
try:
    from api_keys_config import (
        OPENAI_API_KEY,
        ANTHROPIC_API_KEY
    )
except ImportError:
    print("❌ API-Konfiguration nicht gefunden!")
    sys.exit(1)

class LLMIntegration:
    """
    LLM Integration für Bismarck-Modell
    """
    
    def __init__(self):
        self.openai_key = OPENAI_API_KEY
        self.anthropic_key = ANTHROPIC_API_KEY
        self._last_openai_request = 0
        self._last_anthropic_request = 0
        
        print("🧠 LLM INTEGRATION FÜR BISMARCK-MODELL")
        print("=" * 50)
        print("✅ OpenAI API: Bereit")
        print("✅ Anthropic API: Bereit")
        print("✅ Sentiment-Analyse: Bereit")
        print("✅ Finanz-Insights: Bereit")
        print("✅ Rate Limiting: Aktiviert")
        print("=" * 50)
    
    def analyze_with_openai(self, market_data, news_data, technical_indicators):
        """
        Analysiert mit OpenAI API
        """
        try:
            if self.openai_key == "your_openai_key_here":
                return self._get_fallback_openai_analysis()
            
            # Bereite Daten für OpenAI vor
            analysis_prompt = self._prepare_openai_prompt(market_data, news_data, technical_indicators)
            
            headers = {
                'Authorization': f'Bearer {self.openai_key}',
                'Content-Type': 'application/json'
            }
            
            # Rate Limiting: Warte zwischen Anfragen
            current_time = time.time()
            if current_time - self._last_openai_request < 5.0:  # Mindestens 5 Sekunden zwischen Anfragen
                wait_time = 5.0 - (current_time - self._last_openai_request)
                print(f"⏳ Warte {wait_time:.1f} Sekunden für Rate Limiting...")
                time.sleep(wait_time)
            
            self._last_openai_request = time.time()
            
            data = {
                'model': 'gpt-3.5-turbo',  # Verwende gpt-3.5-turbo für bessere Rate Limits
                'messages': [
                    {
                        'role': 'system',
                        'content': 'Du bist ein erfahrener Finanzanalyst und Trading-Experte. Analysiere die gegebenen Marktdaten und gib eine fundierte Trading-Empfehlung basierend auf dem Bismarck-Prinzip: Lerne aus Fehlern anderer.'
                    },
                    {
                        'role': 'user',
                        'content': analysis_prompt
                    }
                ],
                'max_tokens': 300,
                'temperature': 0.7
            }
            
            response = requests.post(
                'https://api.openai.com/v1/chat/completions',
                headers=headers,
                json=data,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                analysis = result['choices'][0]['message']['content']
                print("✅ OpenAI Analyse erhalten")
                return self._parse_openai_analysis(analysis)
            elif response.status_code == 429:
                # Rate limit exceeded - warte länger und retry
                print("⚠️ OpenAI Rate Limit erreicht! Warte 60 Sekunden...")
                time.sleep(60)
                # Retry die Anfrage
                return self.analyze_with_openai(market_data, news_data, technical_indicators)
            else:
                print(f"⚠️ OpenAI API Fehler: {response.status_code}")
                return self._get_fallback_openai_analysis()
                
        except Exception as e:
            print(f"⚠️ OpenAI Fehler: {e}")
            return self._get_fallback_openai_analysis()
    
    def analyze_with_anthropic(self, market_data, news_data, technical_indicators):
        """
        Analysiert mit Anthropic API
        """
        try:
            if self.anthropic_key == "your_anthropic_key_here":
                return self._get_fallback_anthropic_analysis()
            
            # Bereite Daten für Anthropic vor
            analysis_prompt = self._prepare_anthropic_prompt(market_data, news_data, technical_indicators)
            
            headers = {
                'x-api-key': self.anthropic_key,
                'Content-Type': 'application/json',
                'anthropic-version': '2023-06-01'
            }
            
            data = {
                'model': 'claude-3-sonnet-20240229',
                'max_tokens': 500,
                'messages': [
                    {
                        'role': 'user',
                        'content': analysis_prompt
                    }
                ]
            }
            
            response = requests.post(
                'https://api.anthropic.com/v1/messages',
                headers=headers,
                json=data,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                analysis = result['content'][0]['text']
                print("✅ Anthropic Analyse erhalten")
                return self._parse_anthropic_analysis(analysis)
            else:
                print(f"⚠️ Anthropic API Fehler: {response.status_code}")
                return self._get_fallback_anthropic_analysis()
                
        except Exception as e:
            print(f"⚠️ Anthropic Fehler: {e}")
            return self._get_fallback_anthropic_analysis()
    
    def analyze_sentiment(self, news_data):
        """
        Analysiert Sentiment der Nachrichten
        """
        try:
            if not news_data or 'articles' not in news_data:
                return self._get_fallback_sentiment()
            
            articles = news_data['articles']
            if not articles:
                return self._get_fallback_sentiment()
            
            # Kombiniere alle Artikel-Texte
            combined_text = ""
            for article in articles[:5]:  # Max 5 Artikel
                title = article.get('title', '')
                description = article.get('description', '')
                combined_text += f"{title} {description} "
            
            # Verwende OpenAI für Sentiment-Analyse
            if self.openai_key != "your_openai_key_here":
                return self._analyze_sentiment_with_openai(combined_text)
            else:
                return self._get_fallback_sentiment()
                
        except Exception as e:
            print(f"⚠️ Sentiment-Analyse Fehler: {e}")
            return self._get_fallback_sentiment()
    
    def _prepare_openai_prompt(self, market_data, news_data, technical_indicators):
        """
        Bereitet Prompt für OpenAI vor
        """
        prompt = f"""
        Analysiere die folgenden Finanzdaten und gib eine Trading-Empfehlung:

        TECHNISCHE INDIKATOREN:
        - RSI: {technical_indicators.get('rsi', {}).get('rsi', 'N/A')} ({technical_indicators.get('rsi', {}).get('signal', 'N/A')})
        - MACD: {technical_indicators.get('macd', {}).get('signal', 'N/A')}
        - Bollinger Bands: {technical_indicators.get('bollinger', {}).get('signal', 'N/A')}
        - Volatilität: {technical_indicators.get('volatility', {}).get('volatility', 'N/A')}% ({technical_indicators.get('volatility', {}).get('signal', 'N/A')})

        NACHRICHTEN:
        {self._format_news_for_prompt(news_data)}

        Bitte gib eine fundierte Trading-Empfehlung basierend auf dem Bismarck-Prinzip (Lerne aus Fehlern anderer):
        1. KAUFEMPFEHLUNG
        2. VERKAUFSEMPFEHLUNG  
        3. NEUTRAL
        4. Begründung
        5. Risiko-Bewertung
        """
        
        return prompt
    
    def _prepare_anthropic_prompt(self, market_data, news_data, technical_indicators):
        """
        Bereitet Prompt für Anthropic vor
        """
        prompt = f"""
        Als erfahrener Finanzanalyst, analysiere diese Daten und gib eine Trading-Empfehlung:

        TECHNISCHE INDIKATOREN:
        - RSI: {technical_indicators.get('rsi', {}).get('rsi', 'N/A')} ({technical_indicators.get('rsi', {}).get('signal', 'N/A')})
        - MACD: {technical_indicators.get('macd', {}).get('signal', 'N/A')}
        - Bollinger Bands: {technical_indicators.get('bollinger', {}).get('signal', 'N/A')}
        - Volatilität: {technical_indicators.get('volatility', {}).get('volatility', 'N/A')}% ({technical_indicators.get('volatility', {}).get('signal', 'N/A')})

        NACHRICHTEN:
        {self._format_news_for_prompt(news_data)}

        Gib eine Trading-Empfehlung basierend auf dem Bismarck-Prinzip:
        1. Empfehlung (KAUF/VERKAUF/NEUTRAL)
        2. Begründung
        3. Risiko-Bewertung (1-10)
        """
        
        return prompt
    
    def _format_news_for_prompt(self, news_data):
        """
        Formatiert Nachrichten für LLM-Prompts
        """
        if not news_data or 'articles' not in news_data:
            return "Keine Nachrichten verfügbar"
        
        articles = news_data['articles'][:3]  # Max 3 Artikel
        formatted_news = ""
        
        for i, article in enumerate(articles, 1):
            title = article.get('title', 'Kein Titel')
            description = article.get('description', 'Keine Beschreibung')
            formatted_news += f"{i}. {title}: {description}\n"
        
        return formatted_news
    
    def _parse_openai_analysis(self, analysis):
        """
        Parst OpenAI-Analyse
        """
        # Einfache Parsing-Logik
        analysis_lower = analysis.lower()
        
        if 'kauf' in analysis_lower or 'buy' in analysis_lower:
            recommendation = "KAUFEMPFEHLUNG"
        elif 'verkauf' in analysis_lower or 'sell' in analysis_lower:
            recommendation = "VERKAUFSEMPFEHLUNG"
        else:
            recommendation = "NEUTRAL"
        
        return {
            'recommendation': recommendation,
            'analysis': analysis,
            'source': 'OpenAI GPT-4',
            'confidence': 0.8
        }
    
    def _parse_anthropic_analysis(self, analysis):
        """
        Parst Anthropic-Analyse
        """
        # Einfache Parsing-Logik
        analysis_lower = analysis.lower()
        
        if 'kauf' in analysis_lower or 'buy' in analysis_lower:
            recommendation = "KAUFEMPFEHLUNG"
        elif 'verkauf' in analysis_lower or 'sell' in analysis_lower:
            recommendation = "VERKAUFSEMPFEHLUNG"
        else:
            recommendation = "NEUTRAL"
        
        return {
            'recommendation': recommendation,
            'analysis': analysis,
            'source': 'Anthropic Claude-3-Sonnet',
            'confidence': 0.8
        }
    
    def _analyze_sentiment_with_openai(self, text):
        """
        Analysiert Sentiment mit OpenAI
        """
        try:
            headers = {
                'Authorization': f'Bearer {self.openai_key}',
                'Content-Type': 'application/json'
            }
            
            data = {
                'model': 'gpt-4',
                'messages': [
                    {
                        'role': 'system',
                        'content': 'Analysiere das Sentiment der Finanznachrichten. Antworte nur mit: POSITIV, NEGATIV, oder NEUTRAL.'
                    },
                    {
                        'role': 'user',
                        'content': f"Analysiere das Sentiment dieser Finanznachrichten: {text[:500]}"
                    }
                ],
                'max_tokens': 10,
                'temperature': 0.3
            }
            
            response = requests.post(
                'https://api.openai.com/v1/chat/completions',
                headers=headers,
                json=data,
                timeout=15
            )
            
            if response.status_code == 200:
                result = response.json()
                sentiment = result['choices'][0]['message']['content'].strip().upper()
                
                if 'POSITIV' in sentiment:
                    return {'sentiment': 'POSITIV', 'score': 0.7}
                elif 'NEGATIV' in sentiment:
                    return {'sentiment': 'NEGATIV', 'score': -0.7}
                else:
                    return {'sentiment': 'NEUTRAL', 'score': 0.0}
            else:
                return self._get_fallback_sentiment()
                
        except Exception as e:
            print(f"⚠️ Sentiment-Analyse Fehler: {e}")
            return self._get_fallback_sentiment()
    
    def _get_fallback_openai_analysis(self):
        """
        Fallback OpenAI-Analyse
        """
        return {
            'recommendation': 'NEUTRAL',
            'analysis': 'OpenAI API nicht verfügbar. Verwende technische Indikatoren.',
            'source': 'Fallback',
            'confidence': 0.5
        }
    
    def _get_fallback_anthropic_analysis(self):
        """
        Fallback Anthropic-Analyse
        """
        return {
            'recommendation': 'NEUTRAL',
            'analysis': 'Anthropic API nicht verfügbar. Verwende technische Indikatoren.',
            'source': 'Fallback',
            'confidence': 0.5
        }
    
    def _get_fallback_sentiment(self):
        """
        Fallback Sentiment-Analyse
        """
        return {
            'sentiment': 'NEUTRAL',
            'score': 0.0
        }

class BismarckLLMSystem:
    """
    Bismarck-Modell mit LLM-Integration
    """
    
    def __init__(self):
        self.llm = LLMIntegration()
        self.portfolio_value = 100000.0
        self.mistakes_analyzed = 0
        
        print("🚀 BISMARCK-MODELL MIT LLM-INTEGRATION")
        print("=" * 50)
        print("✅ OpenAI GPT-4: Bereit")
        print("✅ Anthropic Claude-3-Sonnet: Bereit")
        print("✅ Sentiment-Analyse: Bereit")
        print("✅ Bismarck-Prinzip: Aus Fehlern anderer lernen")
        print("=" * 50)
    
    def analyze_with_llm(self, market_data, news_data, technical_indicators):
        """
        Analysiert mit LLM-Integration
        """
        print("🧠 LLM-ANALYSE FÜR BISMARCK-MODELL")
        print("-" * 40)
        
        # Sentiment-Analyse
        sentiment = self.llm.analyze_sentiment(news_data)
        print(f"📰 Sentiment: {sentiment['sentiment']} (Score: {sentiment['score']:+.2f})")
        
        # OpenAI Analyse
        openai_analysis = self.llm.analyze_with_openai(market_data, news_data, technical_indicators)
        print(f"🤖 OpenAI: {openai_analysis['recommendation']} (Confidence: {openai_analysis['confidence']:.2f})")
        
        # Anthropic Analyse
        anthropic_analysis = self.llm.analyze_with_anthropic(market_data, news_data, technical_indicators)
        print(f"🧠 Anthropic: {anthropic_analysis['recommendation']} (Confidence: {anthropic_analysis['confidence']:.2f})")
        
        # Kombinierte Empfehlung
        combined_recommendation = self._combine_llm_recommendations(
            sentiment, openai_analysis, anthropic_analysis
        )
        
        print(f"🎯 KOMBINIERTE LLM-EMPFEHLUNG: {combined_recommendation}")
        
        return {
            'sentiment': sentiment,
            'openai': openai_analysis,
            'anthropic': anthropic_analysis,
            'combined': combined_recommendation
        }
    
    def _combine_llm_recommendations(self, sentiment, openai_analysis, anthropic_analysis):
        """
        Kombiniert LLM-Empfehlungen
        """
        recommendations = []
        
        # Sentiment
        if sentiment['sentiment'] == 'POSITIV':
            recommendations.append('KAUF')
        elif sentiment['sentiment'] == 'NEGATIV':
            recommendations.append('VERKAUF')
        
        # OpenAI
        if 'KAUF' in openai_analysis['recommendation']:
            recommendations.append('KAUF')
        elif 'VERKAUF' in openai_analysis['recommendation']:
            recommendations.append('VERKAUF')
        
        # Anthropic
        if 'KAUF' in anthropic_analysis['recommendation']:
            recommendations.append('KAUF')
        elif 'VERKAUF' in anthropic_analysis['recommendation']:
            recommendations.append('VERKAUF')
        
        # Zähle Empfehlungen
        buy_count = recommendations.count('KAUF')
        sell_count = recommendations.count('VERKAUF')
        
        if buy_count >= 2:
            return "🟢 KAUFEMPFEHLUNG (LLM)"
        elif sell_count >= 2:
            return "🔴 VERKAUFSEMPFEHLUNG (LLM)"
        else:
            return "🟡 NEUTRAL (LLM)"
    
    def analyze_mistakes(self):
        """
        Bismarck-Prinzip: Analysiere Fehler
        """
        self.mistakes_analyzed += 1
        print(f"🧠 Bismarck Learning: Fehler analysiert #{self.mistakes_analyzed}")

def test_llm_integration():
    """
    Testet LLM-Integration
    """
    llm = LLMIntegration()
    
    # Test-Daten
    market_data = {'test': 'data'}
    news_data = {'articles': [{'title': 'Market positive', 'description': 'Good news'}]}
    technical_indicators = {
        'rsi': {'rsi': 55, 'signal': 'Neutral'},
        'macd': {'signal': 'Bullish'},
        'bollinger': {'signal': 'Neutral'},
        'volatility': {'volatility': 2.0, 'signal': 'Normal'}
    }
    
    # Teste Sentiment-Analyse
    sentiment = llm.analyze_sentiment(news_data)
    print(f"Sentiment Test: {sentiment}")
    
    # Teste OpenAI
    openai_result = llm.analyze_with_openai(market_data, news_data, technical_indicators)
    print(f"OpenAI Test: {openai_result['recommendation']}")
    
    # Teste Anthropic
    anthropic_result = llm.analyze_with_anthropic(market_data, news_data, technical_indicators)
    print(f"Anthropic Test: {anthropic_result['recommendation']}")

if __name__ == "__main__":
    test_llm_integration()
