import os
from flask import Flask, request, make_response
import pandas as pd
import pandas_ta as ta
import traceback
import json

# Initialize Flask app
app = Flask(__name__)

class DataValidator:
    @staticmethod
    def validate_ohlcv(df: pd.DataFrame) -> tuple:
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
            
            return True, "Valid"
            
        except Exception as e:
            return False, f"Validation error: {str(e)}"

class TechnicalAnalyzer:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        
    def calculate_indicators(self):
        """Calculate comprehensive technical indicators."""
        # Moving Averages
        self.df['SMA_20'] = self.df['close'].rolling(window=20).mean()
        self.df['SMA_50'] = self.df['close'].rolling(window=50).mean()
        self.df['SMA_200'] = self.df['close'].rolling(window=200).mean()
        self.df['EMA_20'] = ta.ema(self.df['close'], length=20)
        
        # Momentum Indicators
        self.df['RSI'] = ta.rsi(self.df['close'])
        macd = ta.macd(self.df['close'])
        self.df['MACD'] = macd['MACD_12_26_9']
        self.df['Signal'] = macd['MACDs_12_26_9']
        
        # Trend Indicator
        self.df['ADX'] = ta.adx(self.df['high'], self.df['low'], self.df['close'])['ADX_14']
        
        return self.df
    
    def analyze_market_condition(self):
        """Analyze overall market condition."""
        current = self.df.iloc[-1]
        
        # Trend Analysis
        def determine_trend():
            if current['close'] > current['SMA_20'] and current['SMA_20'] > current['SMA_50']:
                return 'uptrend'
            elif current['close'] < current['SMA_20'] and current['SMA_20'] < current['SMA_50']:
                return 'downtrend'
            return 'sideways'
        
        trend = determine_trend()
        
        # Decision Logic
        def make_trading_decision():
            if trend == 'uptrend' and current['MACD'] > current['Signal'] and current['RSI'] < 70:
                return 'Buy'
            elif trend == 'downtrend' and current['MACD'] < current['Signal'] and current['RSI'] > 30:
                return 'Sell'
            return 'Hold'
        
        return {
            'trend': trend,
            'recommendation': make_trading_decision(),
            'indicators': {
                'RSI': round(current['RSI'], 2),
                'MACD': round(current['MACD'], 2),
                'Signal': round(current['Signal'], 2),
                'ADX': round(current['ADX'], 2)
            }
        }

@app.route('/run-script', methods=['POST'])
def run_script():
    try:
        # Validate incoming request
        if not request.is_json:
            return make_response({
                "response": {"status": "error", "message": "Request must be JSON"}
            }), 400
        
        data = request.json
        if 'data' not in data:
            return make_response({
                "response": {"status": "error", "message": "No 'data' key found in JSON"}
            }), 400
        
        # Convert string data to DataFrame
        data_string = data.get('data', '')
        rows = [r.strip() for r in data_string.split(',') if r.strip()]
        
        if len(rows) % 6 != 0:
            return make_response({
                "response": {
                    "status": "error", 
                    "message": f"Invalid data format. Expected multiple of 6 elements, got {len(rows)}"
                }
            }), 400
        
        # Reshape data
        formatted_data = [rows[i:i+6] for i in range(0, len(rows), 6)]
        df = pd.DataFrame(formatted_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
        # Validate data
        validator = DataValidator()
        is_valid, message = validator.validate_ohlcv(df)
        if not is_valid:
            return make_response({
                "response": {"status": "error", "message": message}
            }), 400
        
        # Convert columns to appropriate types
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        df = df.astype(float)
        
        # Perform technical analysis
        analyzer = TechnicalAnalyzer(df)
        df_with_indicators = analyzer.calculate_indicators()
        market_condition = analyzer.analyze_market_condition()
        
        # Prepare SWOT-like analysis
        def generate_swot(trend):
            if trend == 'uptrend':
                return {
                    "Strengths": "Strong upward momentum and positive market sentiment.",
                    "Weaknesses": "Potential overbought conditions.",
                    "Opportunities": "High probability of breakout above resistance levels.",
                    "Threats": "Market corrections and profit booking."
                }
            elif trend == 'downtrend':
                return {
                    "Strengths": "Clear downward momentum for short selling.",
                    "Weaknesses": "Potential oversold conditions.",
                    "Opportunities": "Good entry points for value investing.",
                    "Threats": "Reversal risk due to support levels."
                }
            else:
                return {
                    "Strengths": "Stable price movement.",
                    "Weaknesses": "Lack of clear direction.",
                    "Opportunities": "Potential for breakout or breakdown.",
                    "Threats": "Indecisive market conditions."
                }
        
        # Prepare final response
        response = {
            "response": {
                "status": "success",
                "market_condition": market_condition,
                "swot_analysis": generate_swot(market_condition['trend']),
                "total_records": len(df)
            }
        }
        
        return make_response(response), 200
    
    except Exception as e:
        print(f"Error processing request: {str(e)}")
        print(traceback.format_exc())
        return make_response({
            "response": {
                "status": "error",
                "message": "Internal server error",
                "details": str(e)
            }
        }), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)