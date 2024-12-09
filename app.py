# -*- coding: utf-8 -*-
"""Untitled12.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1Bn0VEz8EC69MnSbPzbe3l2bzw20IW8O2
"""

from flask import Flask, request, jsonify
import pandas as pd
import pandas_ta as ta
import os

# Initialize Flask app
app = Flask(__name__)

# Route for root URL
@app.route('/')
def index():
    return "Welcome to the Flask API!"

# Flask route to receive and process data at /run-script
@app.route('/run-script', methods=['POST'])
def run_script():
    try:
        # Receive JSON data from POST request
        data = request.json.get('data', '')

        # Example input: "2024-12-01T10:00:00, 100.0, 110.0, 115.0, 95.0, 1000.0, ..."
        # Split the string into rows and group into columns
        rows = [r.strip() for r in data.split(',') if r.strip()]
        formatted_data = [rows[i:i+6] for i in range(0, len(rows), 6)]

        # Create a DataFrame
        df = pd.DataFrame(formatted_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

        # Convert data types for numerical analysis
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].astype(float)

        # Apply pandas_ta to calculate technical indicators
        df['SMA'] = ta.sma(df['close'], length=3)  # Simple Moving Average
        df['RSI'] = ta.rsi(df['close'])           # Relative Strength Index

        # Convert DataFrame to JSON for response
        result = df.to_dict(orient='records')

        return jsonify({"status": "success", "data": result})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

# Run the Flask app
if __name__ == "__main__":
    # Get the port from Heroku's environment or default to 5000 for local development
    port = int(os.environ.get("PORT", 5000))  # Heroku assigns a dynamic port
    app.run(debug=True, host="0.0.0.0", port=port)