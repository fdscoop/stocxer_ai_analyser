# File: financial_analysis_api.py

import os
from flask import Flask, request, jsonify, make_response
import pandas as pd
import pandas_ta as ta
import traceback

# Initialize Flask app
app = Flask(__name__)

# ---- Utility Functions ---- #

def validate_request_json(req):
    """Validate that the request contains JSON and has the required 'data' key."""
    if not req.is_json:
        return {"status": "error", "message": "Request must be JSON"}, 400
    
    data = req.json
    if 'data' not in data:
        return {"status": "error", "message": "No 'data' key found in JSON"}, 400
    
    if not data.get('data', ''):
        return {"status": "error", "message": "Empty data provided"}, 400
    
    return None, None

def process_data(data_string):
    """
    Parse the input data string, validate its format, and create a DataFrame.
    """
    rows = [r.strip() for r in data_string.split(',') if r.strip()]
    if len(rows) % 6 != 0:
        raise ValueError(f"Invalid data format. Expected multiple of 6 elements, got {len(rows)}")
    
    formatted_data = [rows[i:i+6] for i in range(0, len(rows), 6)]
    df = pd.DataFrame(formatted_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

    try:
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    except Exception as e:
        raise ValueError(f"Failed to parse timestamps: {str(e)}")

    # Ensure numeric columns are correctly converted
    try:
        df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
    except Exception as e:
        raise ValueError(f"Failed to parse numeric values: {str(e)}")

    return df

def calculate_indicators(df):
    """
    Calculate various technical indicators and additional custom logic.
    """
    if df.empty:
        raise ValueError("The DataFrame is empty. Cannot calculate indicators.")
    
    if 'close' not in df.columns:
        raise KeyError("The 'close' column is missing in the DataFrame.")

    # --- EMA Calculations ---
    for length in [9, 20, 50, 200]:
        df[f'EMA{length}'] = ta.ema(df['close'], length=length)
    
    # --- SMA Calculations ---
    for length in [9, 20, 50, 200]:
        df[f'SMA{length}'] = ta.sma(df['close'], length=length)
    
    # --- MACD Calculations ---
    macd = ta.macd(df['close'])
    df['MACD'] = macd['MACD_12_26_9']
    df['MACD_Signal'] = macd['MACDs_12_26_9']
    df['MACD_Hist'] = macd['MACDh_12_26_9']
    
    # --- RSI ---
    df['RSI'] = ta.rsi(df['close'], length=14)
    
    # --- Momentum ---
    df['Momentum'] = ta.mom(df['close'], length=10)
    
    # --- VWAP ---
    df['VWAP'] = ta.vwap(high=df['high'], low=df['low'], close=df['close'], volume=df['volume'])
    
    # --- ATR ---
    df['ATR'] = ta.atr(high=df['high'], low=df['low'], close=df['close'])
    
    # --- Bollinger Bands ---
    bollinger = ta.bbands(df['close'])
    df['BB_Upper'] = bollinger['BBU_20_2.0']
    df['BB_Middle'] = bollinger['BBM_20_2.0']
    df['BB_Lower'] = bollinger['BBL_20_2.0']
    
    # --- ADX ---
    adx = ta.adx(high=df['high'], low=df['low'], close=df['close'])
    df['ADX'] = adx['ADX_14']
    df['Plus_DI'] = adx['DMP_14']
    df['Minus_DI'] = adx['DMN_14']
    
    # --- CCI ---
    df['CCI'] = ta.cci(high=df['high'], low=df['low'], close=df['close'], length=20)
    
    # --- Stochastic Oscillator ---
    stoch = ta.stoch(high=df['high'], low=df['low'], close=df['close'])
    df['Stoch_K'] = stoch['STOCHk_14_3_3']
    df['Stoch_D'] = stoch['STOCHd_14_3_3']
    
    # --- Williams %R ---
    df['Williams_%R'] = ta.willr(high=df['high'], low=df['low'], close=df['close'], length=14)
    
    # --- Trend and SWOT Analysis ---
    df['Trend_Analysis'] = df.apply(analyze_trend, axis=1)
    df['SWOT'] = df.apply(perform_swot, axis=1)
    df['Recommendation'] = df.apply(generate_recommendation, axis=1)
    
    return df

def analyze_trend(row):
    """Perform trend analysis based on indicators."""
    trends = []
    if row['close'] > row['EMA200']:
        trends.append('Above EMA200 (Long-term Bullish)')
    if row['EMA20'] > row['EMA50']:
        trends.append('EMA20 above EMA50 (Medium-term Bullish)')
    if row['EMA9'] > row['EMA20']:
        trends.append('EMA9 above EMA20 (Short-term Bullish)')
    if row['MACD'] > row['MACD_Signal']:
        trends.append('MACD Bullish Crossover')
    if row['RSI'] > 70:
        trends.append('Overbought (RSI)')
    elif row['RSI'] < 30:
        trends.append('Oversold (RSI)')
    return ', '.join(trends) if trends else 'Neutral'

def perform_swot(row):
    """Perform SWOT analysis based on indicators."""
    swot = {'Strengths': [], 'Weaknesses': [], 'Opportunities': [], 'Threats': []}
    if row['close'] > row['EMA200']:
        swot['Strengths'].append('Strong long-term trend')
    if row['RSI'] > 50 and row['RSI'] < 70:
        swot['Strengths'].append('Good momentum')
    if row['close'] < row['EMA200']:
        swot['Weaknesses'].append('Weak long-term trend')
    if row['volume'] < row['volume']:
        swot['Weaknesses'].append('Low volume')
    if row['RSI'] < 30:
        swot['Opportunities'].append('Oversold condition')
    if row['MACD'] > row['MACD_Signal'] and row['MACD'] < 0:
        swot['Opportunities'].append('Potential reversal')
    if row['RSI'] > 70:
        swot['Threats'].append('Overbought condition')
    return swot

def generate_recommendation(row):
    """Generate trading recommendations based on analysis."""
    score = 0
    reasons = []
    if row['close'] > row['EMA200']:
        score += 1
        reasons.append('Price above EMA200')
    if row['EMA20'] > row['EMA50']:
        score += 1
        reasons.append('EMA20 above EMA50')
    if 30 < row['RSI'] < 70:
        score += 1
        reasons.append('RSI in healthy range')
    elif row['RSI'] <= 30:
        score += 2
        reasons.append('Oversold conditions')
    elif row['RSI'] >= 70:
        score -= 2
        reasons.append('Overbought conditions')
    if row['MACD'] > row['MACD_Signal']:
        score += 1
        reasons.append('MACD bullish crossover')
    else:
        score -= 1
        reasons.append('MACD bearish crossover')
    recommendation = (
        'Strong Buy' if score >= 3 else
        'Buy' if score > 0 else
        'Hold' if score == 0 else
        'Sell' if score > -3 else
        'Strong Sell'
    )
    return {'recommendation': recommendation, 'score': score, 'reasons': reasons}

# ---- API Route Handlers ---- #

@app.route('/')
def index():
    """Welcome endpoint."""
    return make_response({"response": "Welcome to the Financial Data Analysis API!"})

@app.route('/health')
def health():
    """Health check endpoint."""
    return make_response({"response": {"status": "healthy"}}), 200

@app.route('/run-script', methods=['POST'])
def run_script():
    """Process incoming data, apply analysis, and return the results."""
    try:
        error_response, status_code = validate_request_json(request)
        if error_response:
            return make_response({"response": error_response}), status_code
        data_string = request.json['data']
        df = process_data(data_string)
        df = calculate_indicators(df)
        result_list = df.to_dict(orient='records')
        result_object = {str(i): item for i, item in enumerate(result_list)}
        for key in result_object:
            result_object[key]['timestamp'] = result_object[key]['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
            for field in result_object[key]:
                if isinstance(result_object[key][field], float) and pd.isna(result_object[key][field]):
                    result_object[key][field] = None
        response = {
            "response": {
                "status": "success",
                "total_records": len(result_object),
                "data": result_object
            }
        }
        return make_response(response), 200
    except Exception as e:
        print(f"Error processing request: {str(e)}")
        print(traceback.format_exc())
        response = {"response": {"status": "error", "message": "Internal server error", "details": str(e)}}
        return make_response(response), 500

# ---- App Entry Point ---- #

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
