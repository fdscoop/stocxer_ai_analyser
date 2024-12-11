
# main.py
import os
from flask import Flask, request, make_response
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import pandas as pd
import pandas_ta as ta
import numpy as np
from datetime import datetime
import redis
import logging
from typing import Dict, List, Tuple, Optional
from config import Config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app with rate limiting
app = Flask(__name__)
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=[Config.API_RATE_LIMIT]
)

# Initialize Redis
redis_client = redis.Redis(
    host=Config.REDIS_HOST,
    port=Config.REDIS_PORT,
    db=Config.REDIS_DB
)

class DataValidator:
    @staticmethod
    def validate_ohlcv(df: pd.DataFrame) -> Tuple[bool, str]:
        """Validate OHLCV data integrity."""
        try:
            if df.empty:
                return False, "Empty dataset"
            
            required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required_columns):
                return False, f"Missing required columns. Need: {required_columns}"
            
            # Check for minimum required data points
            if len(df) < 200:
                return False, "Insufficient data points (minimum 200 required)"
                
            # Validate price relationships
            invalid_prices = df[df['high'] < df['low']].index
            if not invalid_prices.empty:
                return False, f"Invalid price data: high < low at indexes {invalid_prices.tolist()}"
                
            # Validate volume
            if (df['volume'] <= 0).any():
                return False, "Invalid volume data: must be positive"
                
            # Check for gaps in time series
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            time_diff = df['timestamp'].diff()
            max_gap = time_diff.max()
            if max_gap > pd.Timedelta(minutes=5):  # Assuming 5-min data
                return False, f"Data gap detected: {max_gap}"
                
            return True, "Valid"
            
        except Exception as e:
            logger.error(f"Error in data validation: {str(e)}")
            return False, f"Validation error: {str(e)}"

class RiskManager:
    def __init__(self, max_position_size: float = Config.MAX_POSITION_SIZE):
        self.max_position_size = max_position_size
        
    def calculate_position_size(
        self,
        account_size: float,
        risk_per_trade: float,
        entry_price: float,
        stop_loss: float
    ) -> float:
        """Calculate safe position size based on risk parameters."""
        if entry_price <= 0 or stop_loss <= 0:
            raise ValueError("Invalid price values")
            
        risk_amount = account_size * risk_per_trade
        price_risk = abs(entry_price - stop_loss)
        
        if price_risk == 0:
            raise ValueError("Stop loss cannot be equal to entry price")
            
        position_size = risk_amount / price_risk
        max_allowed = account_size * self.max_position_size
        
        return min(position_size, max_allowed)
        
    def calculate_stop_loss(self, df: pd.DataFrame, direction: str) -> float:
        """Calculate stop loss based on ATR."""
        atr = ta.atr(df['high'], df['low'], df['close'], length=14).iloc[-1]
        current_price = df['close'].iloc[-1]
        
        multiplier = 2.0  # ATR multiplier for stop loss
        if direction == 'long':
            return current_price - (atr * multiplier)
        elif direction == 'short':
            return current_price + (atr * multiplier)
        else:
            raise ValueError("Invalid direction. Must be 'long' or 'short'")

class TechnicalAnalyzer:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        
    def calculate_indicators(self) -> pd.DataFrame:
        """Calculate comprehensive technical indicators."""
        try:
            # Moving Averages
            self.df['SMA_20'] = ta.sma(self.df['close'], length=20)
            self.df['SMA_50'] = ta.sma(self.df['close'], length=50)
            self.df['SMA_200'] = ta.sma(self.df['close'], length=200)
            self.df['EMA_20'] = ta.ema(self.df['close'], length=20)
            
            # Momentum
            self.df['RSI'] = ta.rsi(self.df['close'])
            macd = ta.macd(self.df['close'])
            self.df = pd.concat([self.df, macd], axis=1)
            
            # Volatility
            self.df['ATR'] = ta.atr(self.df['high'], self.df['low'], self.df['close'])
            bollinger = ta.bbands(self.df['close'])
            self.df = pd.concat([self.df, bollinger], axis=1)
            
            # Volume Analysis
            self.df['OBV'] = ta.obv(self.df['close'], self.df['volume'])
            self.df['Volume_SMA'] = ta.sma(self.df['volume'], length=20)
            
            return self.df
            
        except Exception as e:
            logger.error(f"Error calculating indicators: {str(e)}")
            raise
            
    def analyze_market_condition(self) -> Dict:
        """Analyze overall market condition."""
        try:
            current = self.df.iloc[-1]
            historical = self.df.iloc[-20:]
            
            # Volatility state
            atr = current['ATR']
            avg_atr = historical['ATR'].mean()
            volatility = 'high' if atr > avg_atr * 1.5 else 'normal'
            
            # Trend strength
            adx = ta.adx(self.df['high'], self.df['low'], self.df['close'])['ADX_14'].iloc[-1]
            trend_strength = 'strong' if adx > 25 else 'weak'
            
            # Volume analysis
            volume_sma = current['Volume_SMA']
            volume_trend = 'increasing' if current['volume'] > volume_sma * 1.2 else 'decreasing'
            
            return {
                'volatility': volatility,
                'trend_strength': trend_strength,
                'volume_trend': volume_trend,
                'adx_value': round(float(adx), 2),
                'atr_value': round(float(atr), 2)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing market condition: {str(e)}")
            raise

class SWOTAnalyzer:
    def __init__(self, df: pd.DataFrame, technical_analyzer: TechnicalAnalyzer):
        self.df = df
        self.ta = technical_analyzer
        
    def generate_swot(self) -> Dict:
        """Generate comprehensive SWOT analysis."""
        try:
            current = self.df.iloc[-1]
            market_condition = self.ta.analyze_market_condition()
            
            resistance_level = self._calculate_resistance_level()
            support_level = self._calculate_support_level()
            volume_analysis = self._analyze_volume_trend()
            price_action = self._analyze_price_action()
            
            return {
                "strengths": self._analyze_strengths(current, market_condition),
                "weaknesses": self._analyze_weaknesses(current, resistance_level),
                "opportunities": self._analyze_opportunities(
                    current, resistance_level, support_level
                ),
                "threats": self._analyze_threats(market_condition, volume_analysis)
            }
            
        except Exception as e:
            logger.error(f"Error generating SWOT analysis: {str(e)}")
            raise
            
    def _calculate_resistance_level(self) -> float:
        """Calculate nearest resistance level using recent highs."""
        return self.df['high'].rolling(window=20).max().iloc[-1]
        
    def _calculate_support_level(self) -> float:
        """Calculate nearest support level using recent lows."""
        return self.df['low'].rolling(window=20).min().iloc[-1]
        
    def _analyze_volume_trend(self) -> Dict:
        """Detailed volume trend analysis."""
        current_volume = self.df['volume'].iloc[-1]
        avg_volume = self.df['volume'].rolling(window=20).mean().iloc[-1]
        return {
            'trend': 'increasing' if current_volume > avg_volume else 'decreasing',
            'strength': current_volume / avg_volume
        }
        
    def _analyze_price_action(self) -> Dict:
        """Analyze recent price action patterns."""
        recent_highs = self.df['high'].rolling(window=5).max()
        recent_lows = self.df['low'].rolling(window=5).min()
        
        return {
            'higher_highs': recent_highs.is_monotonic_increasing,
            'higher_lows': recent_lows.is_monotonic_increasing
        }

class RecommendationEngine:
    def __init__(self, df: pd.DataFrame, risk_manager: RiskManager, 
                 technical_analyzer: TechnicalAnalyzer):
        self.df = df
        self.risk_manager = risk_manager
        self.ta = technical_analyzer
        
    def generate_recommendation(self) -> Dict:
        """Generate detailed trading recommendation."""
        try:
            current_price = self.df['close'].iloc[-1]
            market_condition = self.ta.analyze_market_condition()
            action = self._determine_action(market_condition)
            
            if action == "HOLD":
                return {"action": "HOLD", "reason": "Insufficient signal strength"}
                
            # Calculate risk parameters
            stop_loss = self.risk_manager.calculate_stop_loss(
                self.df,
                'long' if action == "BUY" else 'short'
            )
            
            risk_distance = abs(current_price - stop_loss)
            initial_target = (
                current_price + (risk_distance * 2) if action == "BUY"
                else current_price - (risk_distance * 2)
            )
            
            position_size = self.risk_manager.calculate_position_size(
                account_size=Config.DEFAULT_ACCOUNT_SIZE,
                risk_per_trade=Config.DEFAULT_RISK_PER_TRADE,
                entry_price=current_price,
                stop_loss=stop_loss
            )
            
            return {
                "action": action,
                "timeframe": self._determine_timeframe(market_condition),
                "entry_strategy": {
                    "entry_price": round(current_price, 2),
                    "position_size": round(position_size, 2),
                    "stop_loss": round(stop_loss, 2),
                    "initial_target": round(initial_target, 2),
                    "type": "Limit Order"
                },
                "risk_management": self._generate_risk_management(
                    current_price, stop_loss, position_size
                ),
                "execution_plan": self._generate_execution_plan(
                    action, current_price, stop_loss, initial_target
                ),
                "conditions_to_invalidate": self._generate_invalidation_conditions(
                    action, stop_loss, market_condition
                )
            }
            
        except Exception as e:
            logger.error(f"Error generating recommendation: {str(e)}")
            raise

@app.route('/analyze', methods=['POST'])
@limiter.limit(Config.API_RATE_LIMIT_MINUTE)
def analyze():
    """Main analysis endpoint."""
    try:
        # Validate request
        if not request.is_json:
            return make_response({
                "error": "Request must be JSON"
            }), 400
            
        data = request.json
        if 'data' not in data:
            return make_response({
                "error": "No 'data' key found in JSON"
            }), 400
            
        # Convert data to DataFrame
        df = pd.DataFrame(data['data'])
        
        # Validate data
        validator = DataValidator()
        is_valid, message = validator.validate_ohlcv(df)
        if not is_valid:
            return make_response({"error": message}), 400
            
        # Initialize analyzers
        technical_analyzer = TechnicalAnalyzer(df)
        df = technical_analyzer.calculate_indicators()
        risk_manager = RiskManager()
        swot_analyzer = SWOTAnalyzer(df, technical_analyzer)
        recommendation_engine = RecommendationEngine(
            df, risk_manager, technical_analyzer
        )
        
        # Generate analysis
        market_condition = technical_analyzer.analyze_market_condition()
        swot_analysis = swot_analyzer.generate_swot()
        recommendation = recommendation_engine.generate_recommendation()
        
        # Prepare response
        response = {
            "status": "success",
            "data": {
                "market_condition": market_condition,
                "swot_analysis": swot_analysis,
                "recommendation": recommendation,
                "timestamp": datetime.now().isoformat()
            }
        }
        
        return make_response(response), 200
        
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}", exc_info=True)
        return make_response({
            "status": "error",
            "message": "Internal server error",
            "details": str(e)
        }), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)