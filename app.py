import os
from flask import Flask, request, jsonify, make_response
import pandas as pd
import pandas_ta as ta
import numpy as np
from scipy import stats
import traceback
import json

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def index():
    return make_response({"response": "Welcome to the Trading Analysis API!"})

@app.route('/health')
def health():
    return make_response({"response": {"status": "healthy"}}), 200

def calculate_technical_indicators(df):
    """Calculate comprehensive technical indicators for financial analysis"""
    
    # Multiple Period Moving Averages
    # SMA Calculations
    df['SMA_20'] = ta.sma(df['close'], length=20)
    df['SMA_50'] = ta.sma(df['close'], length=50)
    df['SMA_200'] = ta.sma(df['close'], length=200)
    
    # EMA Calculations
    df['EMA_20'] = ta.ema(df['close'], length=20)
    df['EMA_50'] = ta.ema(df['close'], length=50)
    df['EMA_200'] = ta.ema(df['close'], length=200)
    
    # Moving Average Crossover Signals
    df['Golden_Cross'] = np.where(df['SMA_50'] > df['SMA_200'], 1, 0)
    df['Death_Cross'] = np.where(df['SMA_50'] < df['SMA_200'], 1, 0)
    
    # Moving Average Trend Strength
    df['MA_Trend_Strength'] = (
        (df['close'] - df['SMA_200']) / df['SMA_200'] * 100
    )
    
    # Momentum Indicators
    df['RSI'] = ta.rsi(df['close'], length=14)
    macd = ta.macd(df['close'])
    df['MACD'] = macd['MACD_12_26_9']
    df['MACD_Signal'] = macd['MACDs_12_26_9']
    df['MACD_Hist'] = macd['MACDh_12_26_9']
    
    # Volatility Indicators
    df['ATR'] = ta.atr(df['high'], df['low'], df['close'], length=14)
    bb = ta.bbands(df['close'], length=20)
    df['BB_Upper'] = bb['BBU_20_2.0']
    df['BB_Middle'] = bb['BBM_20_2.0']
    df['BB_Lower'] = bb['BBL_20_2.0']
    
    # Volume Indicators
    df['OBV'] = ta.obv(df['close'], df['volume'])
    df['ADI'] = ta.ad(df['high'], df['low'], df['close'], df['volume'])
    df['Volume_SMA'] = df['volume'].rolling(window=20).mean()
    df['Relative_Volume'] = df['volume'] / df['Volume_SMA']
    
    # Price and Trend Calculations
    df['Price_ROC'] = ta.roc(df['close'], length=14)
    df['Price_Trend'] = df['close'].diff()
    df['Price_Trend_Direction'] = np.where(df['Price_Trend'] > 0, 1, -1)
    df['Daily_Return'] = df['close'].pct_change()
    df['Volatility'] = df['Daily_Return'].rolling(window=20).std()
    
    return df

def calculate_risk_level(df, i):
    """Calculate risk level based on multiple factors"""
    risk_score = 0
    
    # Volatility risk (0-2)
    vol_percentile = stats.percentileofscore(df['Volatility'].iloc[max(0, i-100):i+1], df['Volatility'].iloc[i])
    risk_score += vol_percentile / 50  # Convert to 0-2 scale
    
    # Trend strength risk (0-2)
    trend_strength = abs(df['EMA_20'].iloc[i] - df['EMA_50'].iloc[i]) / df['EMA_50'].iloc[i]
    risk_score += min(2, trend_strength * 100)
    
    # Volume risk (0-1)
    if df['Relative_Volume'].iloc[i] > 2:
        risk_score += 1
    
    # Moving Average Alignment Risk (0-2)
    ma_alignment_score = 0
    if df['close'].iloc[i] > df['SMA_200'].iloc[i]:
        ma_alignment_score += 0.5
    if df['SMA_50'].iloc[i] > df['SMA_200'].iloc[i]:
        ma_alignment_score += 0.5
    if df['EMA_20'].iloc[i] > df['EMA_50'].iloc[i]:
        ma_alignment_score += 0.5
    risk_score += ma_alignment_score
    
    # Map final score to risk levels
    if risk_score < 2:
        return 'LOW'
    elif risk_score < 3.5:
        return 'MEDIUM'
    else:
        return 'HIGH'

def generate_trading_signals(df):
    """Generate trading signals based on technical analysis"""
    
    df['Signal'] = 'HOLD'
    df['Signal_Strength'] = 0
    df['Signal_Reasons'] = ''
    
    for i in range(len(df)):
        reasons = []
        signal_strength = 0
        
        # Skip if not enough data
        if i < 200:  # Increased to account for 200 MA
            continue
            
        # 1. Long-term Trend Analysis (200 MA)
        if df['close'].iloc[i] > df['SMA_200'].iloc[i]:
            signal_strength += 1
            reasons.append("Price above 200 SMA")
        elif df['close'].iloc[i] < df['SMA_200'].iloc[i]:
            signal_strength -= 1
            reasons.append("Price below 200 SMA")
        
        # 2. Medium-term Trend Analysis (50 MA)
        if df['Golden_Cross'].iloc[i] == 1 and df['Golden_Cross'].iloc[i-1] == 0:
            signal_strength += 2
            reasons.append("Golden Cross formed")
        elif df['Death_Cross'].iloc[i] == 1 and df['Death_Cross'].iloc[i-1] == 0:
            signal_strength -= 2
            reasons.append("Death Cross formed")
        
        # 3. Short-term Trend Analysis (20 MA)
        if df['EMA_20'].iloc[i] > df['EMA_50'].iloc[i] and df['EMA_20'].iloc[i-1] <= df['EMA_50'].iloc[i-1]:
            signal_strength += 1
            reasons.append("EMA20 crossed above EMA50")
        elif df['EMA_20'].iloc[i] < df['EMA_50'].iloc[i] and df['EMA_20'].iloc[i-1] >= df['EMA_50'].iloc[i-1]:
            signal_strength -= 1
            reasons.append("EMA20 crossed below EMA50")
        
        # 4. Momentum Analysis
        if df['RSI'].iloc[i] < 30:
            signal_strength += 1
            reasons.append("RSI oversold")
        elif df['RSI'].iloc[i] > 70:
            signal_strength -= 1
            reasons.append("RSI overbought")
        
        if df['MACD'].iloc[i] > df['MACD_Signal'].iloc[i] and df['MACD'].iloc[i-1] <= df['MACD_Signal'].iloc[i-1]:
            signal_strength += 1
            reasons.append("MACD bullish crossover")
        elif df['MACD'].iloc[i] < df['MACD_Signal'].iloc[i] and df['MACD'].iloc[i-1] >= df['MACD_Signal'].iloc[i-1]:
            signal_strength -= 1
            reasons.append("MACD bearish crossover")
        
        # 5. Volume Analysis
        if df['Relative_Volume'].iloc[i] > 1.5 and df['Price_Trend_Direction'].iloc[i] > 0:
            signal_strength += 1
            reasons.append("High volume upward movement")
        elif df['Relative_Volume'].iloc[i] > 1.5 and df['Price_Trend_Direction'].iloc[i] < 0:
            signal_strength -= 1
            reasons.append("High volume downward movement")
        
        # 6. Trend Strength
        trend_strength = df['MA_Trend_Strength'].iloc[i]
        if abs(trend_strength) > 10:  # Strong trend
            signal_strength *= 1.2  # Amplify existing signal
            reasons.append(f"Strong {'upward' if trend_strength > 0 else 'downward'} trend")
        
        # 7. Bollinger Bands Analysis
        if df['close'].iloc[i] < df['BB_Lower'].iloc[i]:
            signal_strength += 1
            reasons.append("Price below lower Bollinger Band")
        elif df['close'].iloc[i] > df['BB_Upper'].iloc[i]:
            signal_strength -= 1
            reasons.append("Price above upper Bollinger Band")
        
        # Generate Final Signal
        df.loc[i, 'Signal_Strength'] = signal_strength
        df.loc[i, 'Signal_Reasons'] = '; '.join(reasons)
        
        if signal_strength >= 2:
            df.loc[i, 'Signal'] = 'BUY'
        elif signal_strength <= -2:
            df.loc[i, 'Signal'] = 'SELL'
        else:
            df.loc[i, 'Signal'] = 'HOLD'
        
        # Add Risk Assessment
        df.loc[i, 'Risk_Level'] = calculate_risk_level(df, i)

@app.route('/run-script', methods=['POST'])
def run_script():
    try:
        # Log the incoming request data for debugging
        print("Incoming request data:", request.json)
        
        # Check if the request is JSON
        if not request.is_json:
            return make_response({"response": {"status": "error", "message": "Request must be JSON"}}), 400
        
        data = request.json
        if 'data' not in data:
            return make_response({"response": {"status": "error", "message": "No 'data' key found in JSON"}}), 400
        
        data_string = data.get('data', '')
        if not data_string:
            return make_response({"response": {"status": "error", "message": "Empty data provided"}}), 400
        
        # Split the string into rows and group into columns
        rows = [r.strip() for r in data_string.split(',') if r.strip()]
        if len(rows) % 6 != 0:
            return make_response({
                "response": {
                    "status": "error",
                    "message": f"Invalid data format. Expected multiple of 6 elements, got {len(rows)}"
                }
            }), 400
        
        formatted_data = [rows[i:i+6] for i in range(0, len(rows), 6)]
        df = pd.DataFrame(formatted_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
        # Convert data types
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
        
        # Calculate indicators and signals
        df = calculate_technical_indicators(df)
        generate_trading_signals(df)
        
        # Convert DataFrame to response format
        result_list = df.to_dict(orient='records')
        result_object = {str(i): item for i, item in enumerate(result_list)}
        
        # Format datetime objects to string and handle NaN values
        for key in result_object:
            result_object[key]['timestamp'] = result_object[key]['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
            for field in result_object[key]:
                if pd.isna(result_object[key][field]):
                    result_object[key][field] = None
        
        response = {
            "response": {
                "status": "success",
                "total_records": len(result_list),
                "data": result_object
            }
        }
        
        resp = make_response(response)
        resp.headers['Content-Type'] = 'application/json'
        return resp, 200
        
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