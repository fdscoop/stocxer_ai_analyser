import os
from flask import Flask, request, make_response
import pandas as pd
import pandas_ta as ta
import traceback
import json

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def index():
    return make_response({"response": "Welcome to the Financial Data Analysis API!"})

@app.route('/health')
def health():
    return make_response({"response": {"status": "healthy"}}), 200

@app.route('/run-script', methods=['POST'])
def run_script():
    try:
        print("Incoming request data:", request.json)
        
        if not request.is_json:
            response = {"response": {"status": "error", "message": "Request must be JSON"}}
            return make_response(response), 400

        data = request.json
        
        if 'data' not in data:
            response = {"response": {"status": "error", "message": "No 'data' key found in JSON"}}
            return make_response(response), 400
        
        data_string = data.get('data', '')
        
        if not data_string:
            response = {"response": {"status": "error", "message": "Empty data provided"}}
            return make_response(response), 400
        
        rows = [r.strip() for r in data_string.split(',') if r.strip()]
        if len(rows) % 6 != 0:
            response = {
                "response": {
                    "status": "error",
                    "message": f"Invalid data format. Expected multiple of 6 elements, got {len(rows)}"
                }
            }
            return make_response(response), 400
            
        formatted_data = [rows[i:i+6] for i in range(0, len(rows), 6)]
        df = pd.DataFrame(formatted_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
        # Convert and set timestamp as DatetimeIndex
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df.set_index('timestamp', inplace=True)

        if df.index.isnull().any():
            response = {"response": {"status": "error", "message": "Invalid or missing timestamps"}}
            return make_response(response), 400

        df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
        
        # Add technical indicators
        df['SMA_9'] = df['close'].rolling(window=9).mean()
        df['SMA_20'] = df['close'].rolling(window=20).mean()
        df['SMA_50'] = df['close'].rolling(window=50).mean()
        df['SMA_200'] = df['close'].rolling(window=200).mean()
        df['EMA_9'] = ta.ema(df['close'], length=9)
        df['EMA_20'] = ta.ema(df['close'], length=20)
        df['EMA_50'] = ta.ema(df['close'], length=50)
        df['EMA_200'] = ta.ema(df['close'], length=200)
        df['MACD'], df['Signal'], df['Histogram'] = ta.macd(df['close'])
        df['Momentum'] = ta.mom(df['close'])
        df['WVAMP'] = ta.vwap(df['high'], df['low'], df['close'], df['volume'])
        
        # Trend and decision analysis
        def analyze_trends(row):
            if row['close'] > row['EMA_9']:
                return 'upward'
            elif row['close'] < row['EMA_9']:
                return 'downward'
            return 'sideways'
        
        def make_decision(row):
            if row['Trend'] == 'upward' and row['MACD'] > row['Signal']:
                return 'Buy'
            elif row['Trend'] == 'downward' and row['MACD'] < row['Signal']:
                return 'Sell'
            return 'Hold'
        
        df['Trend'] = df.apply(analyze_trends, axis=1)
        df['Recommendation'] = df.apply(make_decision, axis=1)
        
        result_list = df.reset_index().to_dict(orient='records')  # Reset index to include 'timestamp' in JSON output
        result_object = {str(i): item for i, item in enumerate(result_list)}
        
        for key in result_object:
            result_object[key]['timestamp'] = result_object[key]['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
            for field in ['SMA_9', 'SMA_20', 'SMA_50', 'SMA_200', 'EMA_9', 'EMA_20', 'EMA_50', 'EMA_200', 
                          'MACD', 'Signal', 'Histogram', 'Momentum', 'WVAMP']:
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
    app.run(host='0.0.0.0', port=port)
