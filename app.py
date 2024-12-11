import os
from flask import Flask, request, jsonify, make_response
import pandas as pd
import pandas_ta as ta
import traceback
import json 

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def index():
    """Root endpoint returning welcome message."""
    return make_response({"response": "Welcome to the Financial Data Analysis API!"})

@app.route('/health')
def health():
    """Health check endpoint."""
    return make_response({"response": {"status": "healthy"}}), 200

@app.route('/run-script', methods=['POST'])
def run_script():
    """
    Process financial data and perform technical analysis.
    """
    try:
        # Log the incoming request data for debugging
        print("Incoming request data:", request.json)
        
        # Check if the request is JSON
        if not request.is_json:
            response = {
                "response": {
                    "status": "error", 
                    "message": "Request must be JSON"
                }
            }
            return make_response(response), 400
        
        # Get data from JSON request
        data = request.json
        
        # Check if 'data' key exists
        if 'data' not in data:
            response = {
                "response": {
                    "status": "error", 
                    "message": "No 'data' key found in JSON"
                }
            }
            return make_response(response), 400
        
        # Get the data value
        data_string = data.get('data', '')
        
        if not data_string:
            response = {
                "response": {
                    "status": "error", 
                    "message": "Empty data provided"
                }
            }
            return make_response(response), 400
        
        # Split and clean the data
        rows = [r.strip() for r in data_string.split(',') if r.strip()]
        rows = [r.replace('\n', '').replace('\r', '') for r in rows]
        rows = [r for r in rows if r]  # Remove any empty strings
        
        # Validate minimum data points
        if len(rows) < 1200:  # Need at least 200 candles (6 values per candle)
            response = {
                "response": {
                    "status": "error", 
                    "message": "Insufficient data points. Need at least 200 candles for long-term analysis."
                }
            }
            return make_response(response), 400
        
        if len(rows) % 6 != 0:
            response = {
                "response": {
                    "status": "error", 
                    "message": f"Invalid data format. Expected multiple of 6 elements, got {len(rows)}"
                }
            }
            return make_response(response), 400
            
        formatted_data = [rows[i:i+6] for i in range(0, len(rows), 6)]
        
        # Create DataFrame
        df = pd.DataFrame(formatted_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

        # Explicitly convert timestamp to datetime and set as index
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)

        # Convert data types
        df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].astype(float)

        # Rest of the code remains the same...
        # Calculate EMAs
        df['EMA9'] = ta.ema(df['close'], length=9)
        df['EMA20'] = ta.ema(df['close'], length=20)
        df['EMA50'] = ta.ema(df['close'], length=50)
        df['EMA200'] = ta.ema(df['close'], length=200)

        # (... rest of the existing code remains the same ...)

        # Modify the result conversion to reset index
        df_reset = df.reset_index()
        result_list = df_reset.to_dict(orient='records')

        # Convert the list to an object with numeric keys
        result_object = {str(i): item for i, item in enumerate(result_list)}

        # Format datetime objects and handle NaN values
        for key in result_object:
            result_object[key]['timestamp'] = result_object[key]['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
            for field in result_object[key]:
                if isinstance(result_object[key][field], float) and pd.isna(result_object[key][field]):
                    result_object[key][field] = None

        # Create summary from latest data point
        latest_record = result_object[str(len(result_object)-1)]
        summary = {
            "current_price": latest_record['close'],
            "recommendation": latest_record['Recommendation'],
            "trend": latest_record['Trend_Analysis'],
            "key_levels": {
                "EMA200": latest_record['EMA200'],
                "EMA50": latest_record['EMA50'],
                "VWAP": latest_record['VWAP']
            },
            "swot": latest_record['SWOT'],
            "technical_indicators": {
                "RSI": latest_record['RSI'],
                "MACD": latest_record['MACD'],
                "ATR": latest_record['ATR']
            }
        }

        # Create the response object
        response = {
            "response": {
                "status": "success",
                "total_records": len(result_list),
                "summary": summary,
                "data": result_object
            }
        }
        
        # Create response with proper headers
        resp = make_response(response)
        resp.headers['Content-Type'] = 'application/json'
        resp.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
        resp.headers['Pragma'] = 'no-cache'
        resp.headers['Expires'] = '0'
        return resp, 200
        
    except ValueError as ve:
        print(f"Value Error in calculations: {str(ve)}")
        print(traceback.format_exc())
        response = {
            "response": {
                "status": "error",
                "message": "Error in calculations",
                "details": str(ve)
            }
        }
        return make_response(response), 422
        
    except pd.errors.EmptyDataError:
        response = {
            "response": {
                "status": "error",
                "message": "Empty or invalid DataFrame"
            }
        }
        return make_response(response), 422
        
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