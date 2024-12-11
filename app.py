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

        # Convert columns to float and handle errors
        try:
            df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
        except ValueError as e:
            response = {"response": {"status": "error", "message": "Non-numeric values found in data", "details": str(e)}}
            return make_response(response), 400

        # Add technical indicators
        df['SMA_20'] = df['close'].rolling(window=20).mean()
        df['SMA_50'] = df['close'].rolling(window=50).mean()
        df['SMA_200'] = df['close'].rolling(window=200).mean()
        df['EMA_20'] = ta.ema(df['close'], length=20)
        df['EMA_50'] = ta.ema(df['close'], length=50)
        df['EMA_200'] = ta.ema(df['close'], length=200)

        macd_df = ta.macd(df['close'])
        df['MACD'] = macd_df['MACD_12_26_9']
        df['Signal'] = macd_df['MACDs_12_26_9']
        df['Histogram'] = macd_df['MACDh_12_26_9']
        df['RSI'] = ta.rsi(df['close'])
        df['ADX'] = ta.adx(df['high'], df['low'], df['close'])['ADX_14']

        # Calculate support and resistance levels (Pivot Points)
        df['Pivot'] = (df['high'] + df['low'] + df['close']) / 3
        df['Support_1'] = 2 * df['Pivot'] - df['high']
        df['Resistance_1'] = 2 * df['Pivot'] - df['low']

        # Trend and decision analysis
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

        # SWOT Analysis based on technical conditions
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

        swot_analysis = generate_swot(df.iloc[-1]['Trend'])

        # Generate report
        report = {
            "summary": {
                "current_trend": df.iloc[-1]['Trend'],
                "recommendation": df.iloc[-1]['Recommendation']
            },
            "technical_indicators": df.iloc[-1].to_dict(),
            "swot_analysis": swot_analysis
        }

        result_list = df.reset_index().to_dict(orient='records')  # Reset index to include 'timestamp' in JSON output
        result_object = {str(i): item for i, item in enumerate(result_list)}

        for key in result_object:
            result_object[key]['timestamp'] = result_object[key]['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
            for field in ['SMA_20', 'SMA_50', 'SMA_200', 'EMA_20', 'EMA_50', 'EMA_200', 'MACD', 'Signal', 'Histogram', 'RSI', 'ADX']:
                if pd.isna(result_object[key][field]):
                    result_object[key][field] = None

        response = {
            "response": {
                "status": "success",
                "total_records": len(result_list),
                "data": result_object,
                "report": report
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
