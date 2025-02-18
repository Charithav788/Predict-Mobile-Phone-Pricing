from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the trained model and scaler
with open("mobile_price_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("scaler.pkl", "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

# Serve index.html
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get JSON data from request
        data = request.get_json()

        # Convert to DataFrame
        df = pd.DataFrame([data])

        # Ensure columns match training data
        expected_columns = ["battery_power", "blue", "clock_speed", "dual_sim", "fc", "four_g", 
                            "int_memory", "m_dep", "mobile_wt", "n_cores", "pc", "px_height", 
                            "px_width", "ram", "sc_h", "sc_w", "talk_time", "three_g", 
                            "touch_screen", "wifi"]

        # Align the columns to match the training dataset
        df = df[expected_columns]

        # Scale features using the loaded scaler
        df_scaled = scaler.transform(df)

        # Make prediction
        prediction = model.predict(df_scaled)[0]

        # Define price range labels
        price_labels = {0:"0\t" "Low Cost", 1:"1\t" "Medium Cost", 2: "2\t""High Cost", 3:"3\t" "Premium"}

        # Get label from prediction
        price_label = price_labels.get(int(prediction), "Unknown")

        return jsonify({"price_range": price_label})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
