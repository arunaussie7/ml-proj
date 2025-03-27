from flask import Flask, render_template
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import numpy as np

app = Flask(__name__)

# Load ML model
model = joblib.load("ml_model.pkl")

# Load XAUUSD data
def get_chart():
    df = pd.read_csv("xauusd_M5.csv")
    
    plt.figure(figsize=(12, 6))
    plt.plot(df['close'], label="XAUUSD Close Price", color='gold')
    plt.xlabel("Candles")
    plt.ylabel("Price (USD)")
    plt.title("XAUUSD Price Chart (5-Minute Interval)")
    plt.grid(True)
    plt.legend()
    
    # Save the chart as an image
    plt.savefig("static/xauusd_chart.png")
    plt.close()

# Function to make predictions
def predict_next_candle():
    df = pd.read_csv("xauusd_M5.csv")
    latest_data = df.iloc[-1][['open', 'high', 'low', 'close', 'tick_volume']].values.reshape(1, -1)
    
    prediction = model.predict_proba(latest_data)[0]
    bullish_prob = round(prediction[1] * 100, 2)  # Probability of bullish candle
    bearish_prob = round(prediction[0] * 100, 2)  # Probability of bearish candle
    
    return bullish_prob, bearish_prob

@app.route("/")
def home():
    get_chart()  # Generate chart
    bullish, bearish = predict_next_candle()  # Get predictions
    return render_template("index.html", bullish=bullish, bearish=bearish)

if __name__ == "__main__":
    app.run(debug=True)
