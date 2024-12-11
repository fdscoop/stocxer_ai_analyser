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

        # Convert data types
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].astype(float)

        # Explicitly create a RangeIndex
        df = df.reset_index(drop=True)

        # Calculate EMAs
        df['EMA9'] = ta.ema(df['close'], length=9)
        df['EMA20'] = ta.ema(df['close'], length=20)
        df['EMA50'] = ta.ema(df['close'], length=50)
        df['EMA200'] = ta.ema(df['close'], length=200)

        # Calculate SMAs
        df['SMA9'] = ta.sma(df['close'], length=9)
        df['SMA20'] = ta.sma(df['close'], length=20)
        df['SMA50'] = ta.sma(df['close'], length=50)
        df['SMA200'] = ta.sma(df['close'], length=200)

        # Calculate MACD
        macd = ta.macd(df['close'])
        df['MACD'] = macd['MACD_12_26_9']
        df['MACD_Signal'] = macd['MACDs_12_26_9']
        df['MACD_Hist'] = macd['MACDh_12_26_9']

        # Calculate RSI
        df['RSI'] = ta.rsi(df['close'], length=14)

        # Calculate Momentum
        df['Momentum'] = ta.mom(df['close'], length=10)

        # Calculate VWAP
        df['VWAP'] = ta.vwap(high=df['high'], low=df['low'], close=df['close'], volume=df['volume'])

        # Calculate Average True Range (ATR)
        df['ATR'] = ta.atr(high=df['high'], low=df['low'], close=df['close'])

        # Calculate Bollinger Bands
        bollinger = ta.bbands(df['close'])
        df['BB_Upper'] = bollinger['BBU_20_2.0']
        df['BB_Middle'] = bollinger['BBM_20_2.0']
        df['BB_Lower'] = bollinger['BBL_20_2.0']

        # Trend Analysis
        def analyze_trend(row):
            trends = []
            
            # EMA Trends
            if row['close'] > row['EMA200']:
                trends.append('Above EMA200 (Long-term Bullish)')
            if row['EMA20'] > row['EMA50']:
                trends.append('EMA20 above EMA50 (Medium-term Bullish)')
            if row['EMA9'] > row['EMA20']:
                trends.append('EMA9 above EMA20 (Short-term Bullish)')
                
            # MACD Analysis
            if row['MACD'] > row['MACD_Signal']:
                trends.append('MACD Bullish Crossover')
            
            # RSI Analysis
            if row['RSI'] > 70:
                trends.append('Overbought (RSI)')
            elif row['RSI'] < 30:
                trends.append('Oversold (RSI)')
                
            return ', '.join(trends) if trends else 'Neutral'

        df['Trend_Analysis'] = df.apply(analyze_trend, axis=1)

        # SWOT Analysis
        def perform_swot(row):
            swot = {
                'Strengths': [],
                'Weaknesses': [],
                'Opportunities': [],
                'Threats': []
            }
            
            # Strengths
            if row['close'] > row['EMA200']:
                swot['Strengths'].append('Strong long-term trend')
            if row['RSI'] > 50 and row['RSI'] < 70:
                swot['Strengths'].append('Good momentum')
                
            # Weaknesses
            if row['close'] < row['EMA200']:
                swot['Weaknesses'].append('Weak long-term trend')
            if row['volume'] < df['volume'].mean():
                swot['Weaknesses'].append('Low volume')
                
            # Opportunities
            if row['RSI'] < 30:
                swot['Opportunities'].append('Oversold condition')
            if row['MACD'] > row['MACD_Signal'] and row['MACD'] < 0:
                swot['Opportunities'].append('Potential reversal')
                
            # Threats
            if row['RSI'] > 70:
                swot['Threats'].append('Overbought condition')
            if row['ATR'] > df['ATR'].mean() * 1.5:
                swot['Threats'].append('High volatility')
                
            return swot

        # Trading Decision Logic
        def generate_recommendation(row):
            score = 0
            reasons = []
            
            # Trend-following signals
            if row['close'] > row['EMA200']:
                score += 1
                reasons.append('Price above EMA200')
            if row['EMA20'] > row['EMA50']:
                score += 1
                reasons.append('EMA20 above EMA50')
                
            # Momentum signals
            if 30 < row['RSI'] < 70:
                score += 1
                reasons.append('RSI in healthy range')
            elif row['RSI'] <= 30:
                score += 2
                reasons.append('Oversold conditions')
            elif row['RSI'] >= 70:
                score -= 2
                reasons.append('Overbought conditions')
                
            # MACD signals
            if row['MACD'] > row['MACD_Signal']:
                score += 1
                reasons.append('MACD bullish crossover')
            else:
                score -= 1
                reasons.append('MACD bearish crossover')
                
            # Generate recommendation
            if score >= 3:
                recommendation = 'Strong Buy'
            elif score > 0:
                recommendation = 'Buy'
            elif score == 0:
                recommendation = 'Hold'
            elif score > -3:
                recommendation = 'Sell'
            else:
                recommendation = 'Strong Sell'
                
            return {
                'recommendation': recommendation,
                'score': score,
                'reasons': reasons
            }

        # Apply analyses to DataFrame
        df['SWOT'] = df.apply(perform_swot, axis=1)
        df['Recommendation'] = df.apply(generate_recommendation, axis=1)

        # Convert DataFrame to list of dictionaries
        result_list = df.to_dict(orient='records')

        # Convert the list to an object with numeric keys
        result_object = {str(i): item for i, item in enumerate(result_list)}

        # Format datetime objects and handle NaN values
        for key in result_object:
            # Ensure timestamp is converted to string
            result_object[key]['timestamp'] = result_object[key]['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
            
            # Handle Recommendation dictionary
            if isinstance(result_object[key]['Recommendation'], dict):
                result_object[key]['Recommendation'] = {
                    'recommendation': result_object[key]['Recommendation'].get('recommendation', 'Unknown'),
                    'score': result_object[key]['Recommendation'].get('score', 0),
                    'reasons': result_object[key]['Recommendation'].get('reasons', [])
                }
            
            # Handle NaN values
            for field in result_object[key]:
                if isinstance(result_object[key][field], float) and pd.isna(result_object[key][field]):
                    result_object[key][field] = None

        # Create summary from latest data point
        latest_record = result_object[str(len(result_object)-1)]
        summary = {
            "current_price": latest_record['close'],
            "recommendation": latest_record['Recommendation']['recommendation'],
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