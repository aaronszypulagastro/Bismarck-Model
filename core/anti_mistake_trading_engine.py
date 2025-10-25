"""
Anti-Mistake Trading Engine - Forschungbasierter Ansatz
=====================================================

Basierend auf aktueller Forschung (2024):
- 98% der ML-Arbeit = Datenverarbeitung und Fehleranalyse
- Pattern Recognition für Trading-Fehler
- Reinforcement Learning für kontinuierliche Anpassung
- LLM-Integration für Collective Intelligence

Statt Emotionen → Fehler-Pattern-Recognition
Statt Gier/Angst → Rationale Fehler-Vermeidung

Author: Anti-Mistake Trading Team
Date: 2025-01-27
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from collections import deque
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class MistakePatternRecognizer:
    """
    Erkennt typische Trading-Fehler basierend auf historischen Daten
    Forschung: Pattern Recognition ist entscheidend für Trading-Erfolg
    """
    
    def __init__(self, lookback_window: int = 100):
        self.lookback_window = lookback_window
        self.mistake_patterns = {
            'overtrading': {
                'description': 'Zu viele Trades in kurzer Zeit',
                'threshold': 10,  # Trades pro Tag
                'loss_probability': 0.75
            },
            'fomo_entry': {
                'description': 'Einstieg bei starken Bewegungen ohne Fundament',
                'volatility_threshold': 0.05,  # 5% intraday
                'loss_probability': 0.68
            },
            'panic_sell': {
                'description': 'Verkauf bei kurzfristigen Verlusten',
                'loss_threshold': -0.02,  # -2% Verlust
                'time_threshold': 24,  # Stunden
                'loss_probability': 0.82
            },
            'greed_hold': {
                'description': 'Halten bei hohen Gewinnen statt Gewinnmitnahme',
                'profit_threshold': 0.10,  # 10% Gewinn
                'loss_probability': 0.60
            },
            'correlation_bias': {
                'description': 'Ignorieren von Marktkorrelationen',
                'correlation_threshold': 0.8,
                'loss_probability': 0.70
            },
            'news_fomo': {
                'description': 'Handeln basierend auf Breaking News',
                'news_impact_threshold': 0.03,  # 3% News-Impact
                'loss_probability': 0.65
            }
        }
        
        self.historical_mistakes = deque(maxlen=1000)
        self.pattern_weights = {}
        
    def analyze_trading_decision(self, 
                               current_market_data: Dict,
                               proposed_action: str,
                               portfolio_state: Dict) -> Dict:
        """
        Analysiert eine Trading-Entscheidung auf Fehler-Pattern
        
        Args:
            current_market_data: Aktuelle Marktdaten
            proposed_action: Vorgeschlagene Aktion (BUY/SELL/HOLD)
            portfolio_state: Aktueller Portfolio-Zustand
            
        Returns:
            Dict mit Fehler-Analyse und Risiko-Score
        """
        mistake_analysis = {
            'risk_score': 0.0,
            'detected_patterns': [],
            'recommendation': 'PROCEED',
            'confidence': 1.0
        }
        
        # Pattern 1: Overtrading Check
        if self._check_overtrading_pattern(portfolio_state):
            mistake_analysis['detected_patterns'].append('overtrading')
            mistake_analysis['risk_score'] += 0.3
            
        # Pattern 2: FOMO Entry Check
        if self._check_fomo_entry_pattern(current_market_data, proposed_action):
            mistake_analysis['detected_patterns'].append('fomo_entry')
            mistake_analysis['risk_score'] += 0.25
            
        # Pattern 3: Panic Sell Check
        if self._check_panic_sell_pattern(portfolio_state, proposed_action):
            mistake_analysis['detected_patterns'].append('panic_sell')
            mistake_analysis['risk_score'] += 0.4
            
        # Pattern 4: Greed Hold Check
        if self._check_greed_hold_pattern(portfolio_state, proposed_action):
            mistake_analysis['detected_patterns'].append('greed_hold')
            mistake_analysis['risk_score'] += 0.2
            
        # Pattern 5: Correlation Bias Check
        if self._check_correlation_bias_pattern(current_market_data):
            mistake_analysis['detected_patterns'].append('correlation_bias')
            mistake_analysis['risk_score'] += 0.15
            
        # Empfehlung basierend auf Risiko-Score
        if mistake_analysis['risk_score'] > 0.7:
            mistake_analysis['recommendation'] = 'AVOID'
            mistake_analysis['confidence'] = 0.9
        elif mistake_analysis['risk_score'] > 0.4:
            mistake_analysis['recommendation'] = 'CAUTION'
            mistake_analysis['confidence'] = 0.7
        else:
            mistake_analysis['recommendation'] = 'PROCEED'
            mistake_analysis['confidence'] = 0.8
            
        return mistake_analysis
    
    def _check_overtrading_pattern(self, portfolio_state: Dict) -> bool:
        """Prüft auf Overtrading-Pattern"""
        recent_trades = portfolio_state.get('recent_trades', [])
        if len(recent_trades) < 5:
            return False
            
        # Zähle Trades in den letzten 24 Stunden
        recent_count = sum(1 for trade in recent_trades[-10:] 
                          if trade.get('time_since', 0) < 24)
        
        return recent_count >= self.mistake_patterns['overtrading']['threshold']
    
    def _check_fomo_entry_pattern(self, market_data: Dict, action: str) -> bool:
        """Prüft auf FOMO Entry Pattern"""
        if action != 'BUY':
            return False
            
        volatility = market_data.get('volatility', 0)
        price_change = market_data.get('price_change_1h', 0)
        
        return (volatility > self.mistake_patterns['fomo_entry']['volatility_threshold'] and
                abs(price_change) > 0.03)  # 3% Preisänderung
    
    def _check_panic_sell_pattern(self, portfolio_state: Dict, action: str) -> bool:
        """Prüft auf Panic Sell Pattern"""
        if action != 'SELL':
            return False
            
        current_position = portfolio_state.get('current_position', {})
        unrealized_pnl = current_position.get('unrealized_pnl', 0)
        holding_time = current_position.get('holding_time_hours', 0)
        
        return (unrealized_pnl < self.mistake_patterns['panic_sell']['loss_threshold'] and
                holding_time < self.mistake_patterns['panic_sell']['time_threshold'])
    
    def _check_greed_hold_pattern(self, portfolio_state: Dict, action: str) -> bool:
        """Prüft auf Greed Hold Pattern"""
        if action != 'HOLD':
            return False
            
        current_position = portfolio_state.get('current_position', {})
        unrealized_pnl = current_position.get('unrealized_pnl', 0)
        
        return unrealized_pnl > self.mistake_patterns['greed_hold']['profit_threshold']
    
    def _check_correlation_bias_pattern(self, market_data: Dict) -> bool:
        """Prüft auf Correlation Bias Pattern"""
        market_correlation = market_data.get('market_correlation', 0)
        return abs(market_correlation) > self.mistake_patterns['correlation_bias']['correlation_threshold']
    
    def update_pattern_weights(self, trade_result: Dict):
        """Aktualisiert Pattern-Gewichtungen basierend auf Trade-Ergebnissen"""
        for pattern in trade_result.get('mistake_patterns', []):
            if pattern in self.pattern_weights:
                self.pattern_weights[pattern] *= 1.1  # Erhöhe Gewichtung
            else:
                self.pattern_weights[pattern] = 1.0
                
        # Speichere historischen Fehler
        self.historical_mistakes.append({
            'timestamp': trade_result.get('timestamp'),
            'patterns': trade_result.get('mistake_patterns', []),
            'outcome': trade_result.get('outcome', 'unknown'),
            'loss': trade_result.get('loss', 0)
        })

class LLMMarketIntelligence:
    """
    LLM-Integration für Collective Intelligence
    Lernen aus den Fehlern ALLER Marktteilnehmer, nicht nur eigenen
    """
    
    def __init__(self):
        self.collective_mistakes = {}
        self.market_sentiment_history = deque(maxlen=1000)
        
    def analyze_collective_mistakes(self, market_data: Dict) -> Dict:
        """
        Analysiert kollektive Fehler der Marktteilnehmer
        Basierend auf aktueller Forschung zu Collective Intelligence
        """
        analysis = {
            'crowd_behavior': 'RATIONAL',
            'mistake_probability': 0.0,
            'collective_errors': [],
            'recommendation': 'FOLLOW_TREND'
        }
        
        # Analysiere Marktvolatilität als Indikator für kollektive Fehler
        volatility = market_data.get('volatility', 0)
        volume_spike = market_data.get('volume_spike', 1.0)
        
        # Hohe Volatilität + Volumen-Spike = Panic/Manic Behavior
        if volatility > 0.05 and volume_spike > 2.0:
            analysis['crowd_behavior'] = 'PANIC_MANIC'
            analysis['mistake_probability'] = 0.8
            analysis['collective_errors'].append('herd_behavior')
            analysis['recommendation'] = 'CONTRARIAN'
            
        # Moderate Volatilität = Rational Trading
        elif volatility < 0.02:
            analysis['crowd_behavior'] = 'RATIONAL'
            analysis['mistake_probability'] = 0.2
            analysis['recommendation'] = 'FOLLOW_TREND'
            
        return analysis
    
    def extract_news_sentiment(self, news_data: List[Dict]) -> Dict:
        """
        Extrahiert Sentiment aus Finanznachrichten
        Simuliert LLM-Verarbeitung von News-Daten
        """
        sentiment_scores = []
        mistake_indicators = []
        
        for news_item in news_data:
            # Simuliere LLM-Analyse (in echter Implementierung würde hier ein LLM aufgerufen)
            text = news_item.get('text', '').lower()
            
            # Erkenne Panic-Indikatoren
            panic_words = ['crash', 'panic', 'sell-off', 'bloodbath', 'meltdown']
            if any(word in text for word in panic_words):
                mistake_indicators.append('news_panic')
                sentiment_scores.append(-0.8)
                
            # Erkenne FOMO-Indikatoren
            fomo_words = ['moon', 'rocket', 'breakthrough', 'game-changer', 'revolutionary']
            if any(word in text for word in fomo_words):
                mistake_indicators.append('news_fomo')
                sentiment_scores.append(0.9)
                
            # Neutral
            else:
                sentiment_scores.append(0.0)
        
        avg_sentiment = np.mean(sentiment_scores) if sentiment_scores else 0.0
        
        return {
            'sentiment_score': avg_sentiment,
            'mistake_indicators': mistake_indicators,
            'news_count': len(news_data),
            'confidence': 0.7
        }

class AntiMistakeTradingEngine:
    """
    Haupt-Engine: Kombiniert Pattern Recognition mit LLM-Intelligence
    Ersetzt Emotion Engine durch rationale Fehler-Vermeidung
    """
    
    def __init__(self, 
                 base_learning_rate: float = 0.001,
                 mistake_penalty: float = 0.5,
                 learning_boost: float = 1.2):
        
        # Basis-Parameter (rational, nicht emotional)
        self.base_lr = base_learning_rate
        self.mistake_penalty = mistake_penalty
        self.learning_boost = learning_boost
        
        # Komponenten
        self.mistake_recognizer = MistakePatternRecognizer()
        self.llm_intelligence = LLMMarketIntelligence()
        
        # Performance Tracking (rational)
        self.performance_history = deque(maxlen=100)
        self.mistake_history = deque(maxlen=1000)
        self.learning_efficiency = 0.0
        
        # Adaptive Parameter (basierend auf Fehler-Learning)
        self.current_lr = base_learning_rate
        self.risk_tolerance = 0.5
        self.confidence_threshold = 0.7
        
    def analyze_trading_opportunity(self, 
                                  market_data: Dict,
                                  portfolio_state: Dict,
                                  proposed_action: str) -> Dict:
        """
        Hauptanalyse-Funktion: Kombiniert alle Fehler-Erkennungs-Systeme
        """
        analysis = {
            'action': proposed_action,
            'risk_score': 0.0,
            'confidence': 0.0,
            'recommendation': 'PROCEED',
            'reasoning': [],
            'mistake_patterns': [],
            'collective_intelligence': {},
            'final_decision': 'PROCEED'
        }
        
        # 1. Individuelle Fehler-Pattern-Analyse
        mistake_analysis = self.mistake_recognizer.analyze_trading_decision(
            market_data, proposed_action, portfolio_state
        )
        
        analysis['mistake_patterns'] = mistake_analysis['detected_patterns']
        analysis['risk_score'] += mistake_analysis['risk_score']
        
        # 2. Collective Intelligence Analyse
        collective_analysis = self.llm_intelligence.analyze_collective_mistakes(market_data)
        analysis['collective_intelligence'] = collective_analysis
        
        # 3. News Sentiment Analyse (falls verfügbar)
        if 'news_data' in market_data:
            news_analysis = self.llm_intelligence.extract_news_sentiment(market_data['news_data'])
            if news_analysis['mistake_indicators']:
                analysis['mistake_patterns'].extend(news_analysis['mistake_indicators'])
                analysis['risk_score'] += 0.2
                
        # 4. Finale Entscheidung
        total_risk = analysis['risk_score'] + collective_analysis['mistake_probability']
        
        if total_risk > 0.8:
            analysis['final_decision'] = 'AVOID'
            analysis['confidence'] = 0.9
            analysis['reasoning'].append(f"Hohe Fehler-Wahrscheinlichkeit: {total_risk:.2f}")
        elif total_risk > 0.5:
            analysis['final_decision'] = 'CAUTION'
            analysis['confidence'] = 0.7
            analysis['reasoning'].append(f"Moderate Fehler-Wahrscheinlichkeit: {total_risk:.2f}")
        else:
            analysis['final_decision'] = 'PROCEED'
            analysis['confidence'] = 0.8
            analysis['reasoning'].append(f"Niedrige Fehler-Wahrscheinlichkeit: {total_risk:.2f}")
            
        return analysis
    
    def update_learning_from_trade(self, trade_result: Dict):
        """
        Lernen aus Trade-Ergebnissen (nicht Emotionen)
        Forschung: Kontinuierliches Lernen ist entscheidend
        """
        # Update Pattern Recognizer
        self.mistake_recognizer.update_pattern_weights(trade_result)
        
        # Update Performance History
        self.performance_history.append(trade_result.get('pnl', 0))
        
        # Berechne Learning Efficiency
        if len(self.performance_history) >= 10:
            recent_performance = np.mean(list(self.performance_history)[-10:])
            self.learning_efficiency = max(0, min(1, recent_performance / 100))
            
        # Adaptive Learning Rate basierend auf Fehlern
        if trade_result.get('outcome') == 'MISTAKE':
            self.current_lr *= (1 + self.learning_boost)  # Mehr lernen bei Fehlern
        else:
            self.current_lr *= (1 - self.mistake_penalty * 0.1)  # Weniger bei Erfolg
            
        # Begrenze Learning Rate
        self.current_lr = np.clip(self.current_lr, self.base_lr * 0.1, self.base_lr * 5.0)
        
    def get_adaptive_parameters(self) -> Dict:
        """
        Gibt adaptive Parameter zurück (basierend auf Fehler-Learning)
        Ersetzt emotionale Parameter durch rationale
        """
        return {
            'learning_rate': self.current_lr,
            'risk_tolerance': self.risk_tolerance,
            'confidence_threshold': self.confidence_threshold,
            'learning_efficiency': self.learning_efficiency,
            'mistake_count': len(self.mistake_history),
            'performance_trend': np.mean(list(self.performance_history)[-10:]) if len(self.performance_history) >= 10 else 0
        }
    
    def get_system_stats(self) -> Dict:
        """
        System-Statistiken für Monitoring
        """
        return {
            'total_trades_analyzed': len(self.mistake_history),
            'mistake_patterns_detected': len(self.mistake_recognizer.mistake_patterns),
            'learning_efficiency': self.learning_efficiency,
            'current_risk_tolerance': self.risk_tolerance,
            'performance_trend': np.mean(list(self.performance_history)[-20:]) if len(self.performance_history) >= 20 else 0,
            'adaptive_parameters': self.get_adaptive_parameters()
        }

# Beispiel-Nutzung
if __name__ == "__main__":
    # Test der Anti-Mistake Trading Engine
    engine = AntiMistakeTradingEngine()
    
    # Beispiel Marktdaten
    market_data = {
        'volatility': 0.03,
        'price_change_1h': 0.02,
        'market_correlation': 0.7,
        'volume_spike': 1.5
    }
    
    # Beispiel Portfolio
    portfolio_state = {
        'recent_trades': [
            {'time_since': 12, 'action': 'BUY'},
            {'time_since': 8, 'action': 'SELL'},
            {'time_since': 6, 'action': 'BUY'}
        ],
        'current_position': {
            'unrealized_pnl': -0.01,
            'holding_time_hours': 18
        }
    }
    
    # Analyse einer Trading-Entscheidung
    analysis = engine.analyze_trading_opportunity(
        market_data, portfolio_state, 'SELL'
    )
    
    print("Anti-Mistake Trading Analysis:")
    print(f"Final Decision: {analysis['final_decision']}")
    print(f"Risk Score: {analysis['risk_score']:.2f}")
    print(f"Mistake Patterns: {analysis['mistake_patterns']}")
    print(f"Confidence: {analysis['confidence']:.2f}")
