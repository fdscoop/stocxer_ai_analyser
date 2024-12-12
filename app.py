import os
from flask import Flask, request, make_response
import pandas as pd
import pandas_ta as ta
import traceback
import json
from datetime import datetime
import numpy as np

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def index():
    return make_response({"response": "Welcome to the Financial Data Analysis API!"})

@app.route('/health')
def health():
    return make_response({"response": {"status": "healthy"}}), 200

def validate_request(request_data):
    """Validate incoming request data"""
    if not request_data.is_json:
        return False, "Request must be JSON"
    
    data = request_data.json
    if 'data' not in data:
        return False, "No 'data' key found in JSON"
    
    data_string = data.get('data', '')
    if not data_string:
        return False, "Empty data provided"
    
    return True, data_string

def prepare_dataframe(data_string):
    """Convert input data to DataFrame and validate"""
    # Split and clean rows
    rows = [r.strip() for r in data_string.split(',') if r.strip()]
    
    # Validate data format
    if len(rows) % 6 != 0:
        raise ValueError(f"Invalid data format. Expected multiple of 6 elements, got {len(rows)}")
    
    # Reshape data
    formatted_data = [rows[i:i+6] for i in range(0, len(rows), 6)]
    df = pd.DataFrame(formatted_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    
    # Convert and set timestamp as DatetimeIndex
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df.set_index('timestamp', inplace=True)
    
    # Validate timestamp
    if df.index.isnull().any():
        raise ValueError("Invalid or missing timestamps")
    
    # Convert columns to float
    try:
        df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
    except ValueError as e:
        raise ValueError(f"Non-numeric values found in data: {str(e)}")
    
    return df

def calculate_technical_indicators(df):
    """Calculate all technical indicators"""
    # Simple Moving Averages
    df['SMA_20'] = df['close'].rolling(window=20).mean()
    df['SMA_50'] = df['close'].rolling(window=50).mean()
    df['SMA_200'] = df['close'].rolling(window=200).mean()
    
    # Exponential Moving Averages
    df['EMA_20'] = ta.ema(df['close'], length=20)
    df['EMA_50'] = ta.ema(df['close'], length=50)
    df['EMA_200'] = ta.ema(df['close'], length=200)
    
    # MACD Indicator
    macd_df = ta.macd(df['close'])
    df['MACD'] = macd_df['MACD_12_26_9']
    df['Signal'] = macd_df['MACDs_12_26_9']
    df['Histogram'] = macd_df['MACDh_12_26_9']
    
    # Additional Indicators
    df['RSI'] = ta.rsi(df['close'])
    df['ADX'] = ta.adx(df['high'], df['low'], df['close'])['ADX_14']
    
    return df

def calculate_price_levels(df, last_record):
    """
    Calculate advanced price levels with multiple strategies
    
    Args:
        df (pandas.DataFrame): Full dataframe of price data
        last_record (dict): Last record in the dataframe
    
    Returns:
        dict: Calculated price levels with multiple strategies
    """
    # Get current price and key technical indicators
    current_close = last_record['close']
    trend = last_record['Trend']
    
    # Calculate volatility (Average True Range)
    atr_14 = ta.atr(df['high'], df['low'], df['close'], length=14)[-1]
    
    # Support and Resistance Calculation
    support_levels = [
        last_record['SMA_20'],    # 20-day Moving Average Support
        last_record['SMA_50'],    # 50-day Moving Average Support
        current_close * 0.95,     # 5% below current price
        current_close - atr_14    # One ATR below current price
    ]
    
    resistance_levels = [
        last_record['SMA_20'],    # 20-day Moving Average Resistance
        last_record['SMA_50'],    # 50-day Moving Average Resistance
        current_close * 1.05,     # 5% above current price
        current_close + atr_14    # One ATR above current price
    ]
    
    # Price Level Calculation Strategies
    price_strategies = {
        'uptrend': {
            'buy_price': max(support_levels),  # Most robust support level
            'sell_price': max(resistance_levels),  # Break-out resistance
            'stop_loss': current_close * 0.90,  # 10% below entry
            'first_target': current_close * 1.05,  # 5% gain
            'second_target': current_close * 1.10,  # 10% gain
            'risk_reward_ratio': 2.0  # 2:1 Risk-Reward
        },
        'downtrend': {
            'buy_price': min(support_levels),  # Potential bounce point
            'sell_price': min(resistance_levels),  # Selling at resistance
            'stop_loss': current_close * 1.10,  # 10% above entry
            'first_target': current_close * 0.95,  # 5% gain in downtrend
            'second_target': current_close * 0.90,  # 10% gain in downtrend
            'risk_reward_ratio': 1.5  # More conservative
        },
        'sideways': {
            'buy_price': current_close * 0.98,  # Slightly below current price
            'sell_price': current_close * 1.02,  # Slightly above current price
            'stop_loss': current_close * 0.95,  # 5% below current price
            'first_target': current_close * 1.03,  # 3% potential gain
            'second_target': current_close * 1.05,  # 5% potential gain
            'risk_reward_ratio': 1.0  # Balanced approach
        }
    }
    
    # Retrieve price levels based on current trend, default to sideways
    price_levels = price_strategies.get(trend, price_strategies['sideways'])
    
    # Round price levels to two decimal places
    for key in price_levels:
        if isinstance(price_levels[key], (int, float)):
            price_levels[key] = round(price_levels[key], 2)
    
    return price_levels

def analyze_market(df):
    """Perform market analysis and generate recommendations"""
    def analyze_trends(row):
        if row['close'] > row['EMA_20'] and row['SMA_20'] > row['SMA_50']:
            return 'uptrend'
        elif row['close'] < row['EMA_20'] and row['SMA_20'] < row['SMA_50']:
            return 'downtrend'
        return 'sideways'
    
    def make_decision(row):
        if row['Trend'] == 'uptrend' and row['MACD'] > row['Signal'] and row['RSI'] < 70:
            return 'Buy'
        elif row['Trend'] == 'downtrend' and row['MACD'] < row['Signal'] and row['RSI'] > 30:
            return 'Sell'
        return 'Hold'
    
    df['Trend'] = df.apply(analyze_trends, axis=1)
    df['Recommendation'] = df.apply(make_decision, axis=1)
    
    return df

def generate_swot(trend):
    """Generate SWOT analysis based on trend"""
    # (Same implementation as in previous version)
    swot_analysis = {
        'uptrend': {
            "Strengths": [
                "Strong upward momentum",
                "Positive market sentiment",
                "Higher highs and higher lows"
            ],
            "Weaknesses": [
                "Potential overbought conditions",
                "Risk of sudden reversals",
                "Increased volatility"
            ],
            "Opportunities": [
                "Breakout above resistance levels",
                "Momentum trading strategies",
                "Positive trend continuation"
            ],
            "Threats": [
                "Market corrections",
                "Profit booking pressure",
                "Overvaluation risks"
            ]
        },
        'downtrend': {
            "Strengths": [
                "Clear downward momentum",
                "Short selling opportunities",
                "Lower risk entry points"
            ],
            "Weaknesses": [
                "Negative market sentiment",
                "Potential oversold conditions",
                "Reduced trading volume"
            ],
            "Opportunities": [
                "Value investing entries",
                "Accumulation at support",
                "Counter-trend bounces"
            ],
            "Threats": [
                "Further price deterioration",
                "Extended bearish phase",
                "Loss of key support levels"
            ]
        },
        'sideways': {
            "Strengths": [
                "Stable price movement",
                "Reduced volatility",
                "Clear trading range"
            ],
            "Weaknesses": [
                "Lack of clear direction",
                "Limited trending opportunities",
                "Range-bound constraints"
            ],
            "Opportunities": [
                "Range trading strategies",
                "Breakout preparation",
                "Pattern formation trading"
            ],
            "Threats": [
                "False breakouts",
                "Extended consolidation",
                "Choppy price action"
            ]
        }
    }
    
    return swot_analysis.get(trend, swot_analysis['sideways'])

def prepare_bubble_response(df, last_record):
    """Prepare response data in Bubble-friendly format"""
    # (Similar to previous implementation, with added price levels)
    records = []
    
    for index, row in df.reset_index().iterrows():
        record = {
            "id": index + 1,
            "timestamp": row['timestamp'].strftime('%Y-%m-%d %H:%M:%S'),
            "price_data": {
                "open": float(row['open']),
                "high": float(row['high']),
                "low": float(row['low']),
                "close": float(row['close']),
                "volume": float(row['volume'])
            },
            # (Rest of the record remains the same)
        }
        records.append(record)
    
    # Calculate price levels
    price_levels = calculate_price_levels(df, last_record)
    
    # Generate market summary
    market_summary = {
        "last_updated": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "latest_price": float(last_record['close']),
        "daily_change": float(last_record['close'] - last_record['open']),
        "daily_change_percent": float((last_record['close'] - last_record['open']) / last_record['open'] * 100),
        "volume": float(last_record['volume']),
        "current_trend": last_record['Trend'],
        "recommendation": last_record['Recommendation'],
        "price_levels": price_levels
    }
    
    # Generate SWOT analysis
    swot = generate_swot(last_record['Trend'])
    
    return {
        "meta": {
            "total_records": len(records),
            "market_summary": market_summary,
            "swot_analysis": swot
        },
        "data": records
    }

# Rest of the Flask application remains the same as in the previous implementation
@app.route('/run-script', methods=['POST'])
def run_script():
    try:
        # Validate request
        is_valid, message = validate_request(request)
        if not is_valid:
            return make_response({"response": {"status": "error", "message": message}}), 400
        
        # Get data from request
        data_string = request.json['data']
        
        # Prepare DataFrame
        df = prepare_dataframe(data_string)
        
        # Calculate technical indicators
        df = calculate_technical_indicators(df)
        
        # Perform market analysis
        df = analyze_market(df)
        
        # Prepare Bubble-friendly response
        last_record = df.iloc[-1].to_dict()
        bubble_response = prepare_bubble_response(df, last_record)
        
        # Final response
        response = {
            "response": {
                "status": "success",
                "result": bubble_response
            }
        }
        
        resp = make_response(response)
        resp.headers['Content-Type'] = 'application/json'
        return resp, 200
    
    except Exception as e:
        print(f"Error processing request: {str(e)}")
        print(traceback.format_exc())
        response = {
            "response": {
                "status": "error",
                "message": "Internal server error",
                "details": str(e)
            }
        }
        return make_response(response), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)