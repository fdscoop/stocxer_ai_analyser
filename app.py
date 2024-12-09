import os
from flask import Flask, request, jsonify
import pandas as pd
import numpy as np

# Optional: Use ta library for RSI if needed
from ta.momentum import RSIIndicator

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def index():
    return "Welcome to the Financial Data Analysis API!"

@app.route('/health')
def health():
    return jsonify({"status": "healthy"}), 200

@app.route('/run-script', methods=['POST'])
def run_script():
    try:
        data = request.json.get('data', '')
        if not data:
            return jsonify({"status": "error", "message": "No data provided"}), 400
        
        # Split the string into rows and group into columns
        rows = [r.strip() for r in data.split(',') if r.strip()]
        if len(rows) % 6 != 0:
            return jsonify({"status": "error", "message": "Invalid data format"}), 400
            
        formatted_data = [rows[i:i+6] for i in range(0, len(rows), 6)]
        
        # Create a DataFrame
        df = pd.DataFrame(formatted_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
        # Convert data types for numerical analysis
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
        
        # Calculate technical indicators using pandas methods
        df['SMA'] = df['close'].rolling(window=3).mean()
        
        # Calculate RSI using ta library
        rsi_indicator = RSIIndicator(df['close'])
        df['RSI'] = rsi_indicator.rsi()
        
        # Convert DataFrame to JSON for response
        result = df.to_dict(orient='records')
        return jsonify({"status": "success", "data": result})
        
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# Add this for Heroku deployment
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)