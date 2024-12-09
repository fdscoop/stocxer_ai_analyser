import os
from flask import Flask, request, jsonify
import pandas as pd
import pandas_ta as ta
import traceback

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def index():
    return jsonify({"response": "Welcome to the Financial Data Analysis API!"})

@app.route('/health')
def health():
    return jsonify({"response": {"status": "healthy"}}), 200

@app.route('/run-script', methods=['POST'])
def run_script():
    try:
        # Log the incoming request data for debugging
        print("Incoming request data:", request.json)
        print("Request content type:", request.content_type)
        
        # Check if the request is JSON
        if not request.is_json:
            return jsonify({
                "response": {
                    "status": "error", 
                    "message": "Request must be JSON",
                    "details": f"Received content type: {request.content_type}"
                }
            }), 400
        
        # Get data from JSON request
        data = request.json
        
        # Check if 'data' key exists
        if 'data' not in data:
            return jsonify({
                "response": {
                    "status": "error", 
                    "message": "No 'data' key found in JSON"
                }
            }), 400
        
        # Get the data value
        data_string = data.get('data', '')
        
        if not data_string:
            return jsonify({
                "response": {
                    "status": "error", 
                    "message": "Empty data provided"
                }
            }), 400
        
        # Split the string into rows and group into columns
        rows = [r.strip() for r in data_string.split(',') if r.strip()]
        if len(rows) % 6 != 0:
            return jsonify({
                "response": {
                    "status": "error", 
                    "message": f"Invalid data format. Expected multiple of 6 elements, got {len(rows)}"
                }
            }), 400
            
        formatted_data = [rows[i:i+6] for i in range(0, len(rows), 6)]
        
        # Create a DataFrame
        df = pd.DataFrame(formatted_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
        # Convert data types for numerical analysis
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
        
        # Apply pandas_ta to calculate technical indicators
        df['SMA'] = df['close'].rolling(window=3).mean()
        df['RSI'] = ta.rsi(df['close'])
        
        # Convert DataFrame to a list of dictionaries
        result_list = df.to_dict(orient='records')
        
        # Convert the list to an object with numeric keys
        result_object = {str(i): item for i, item in enumerate(result_list)}
        
        # Add metadata to help Bubble process the response
        response = {
            "response": {
                "status": "success",
                "total_records": len(result_list),
                "data": result_object
            }
        }
        
        # Log the response before returning (for debugging)
        print("Response:", response)
        return jsonify(response), 200
        
    except Exception as e:
        # Log the full traceback for server-side debugging
        print(f"Error processing request: {str(e)}")
        print(traceback.format_exc())
        
        return jsonify({
            "response": {
                "status": "error", 
                "message": "Internal server error",
                "details": str(e)
            }
        }), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)