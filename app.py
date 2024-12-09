from flask import Flask, request, jsonify
import pandas as pd
import pandas_ta as ta
import os

# Create the Flask application instance
app = Flask(__name__)

@app.route('/')
def index():
    return "Welcome to the Flask API!"

@app.route('/run-script', methods=['POST'])
def run_script():
    try:
        # Your existing code here
        data = request.json.get('data', '')
        # ... rest of your function ...
        return jsonify({"status": "success", "data": result})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# Remove the if __name__ == "__main__" block for Heroku
# Gunicorn will handle the running of the app