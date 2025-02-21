from flask import Flask, jsonify, request
import numpy as np
import tensorflow as tf
import joblib

# Load the trained model without compiling it to avoid serialization issues
model = tf.keras.models.load_model('fishpond_model.h5', compile=False)

# Load the scaler
scaler = joblib.load('scaler.pkl')

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Expect JSON input with past sensor data
        data = request.json.get("past_data")
        if not data:
            return jsonify({"error": "No input data provided"}), 400

        # Convert input to NumPy array and reshape to (1, 72, 5)
        past_data = np.array(data).reshape(1, 72, 5)

        # Scale the input data using the saved scaler
        scaled_data = scaler.transform(past_data.reshape(-1, 5)).reshape(1, 72, 5)

        # Make predictions
        predictions = model.predict(scaled_data)

        # Inverse transform predictions to get real-world values
        predicted_values = scaler.inverse_transform(predictions.reshape(-1, 5)).reshape(1, 72, 5)

        # Format response JSON
        response = {
            "Temperature (°C)": predicted_values[0, :, 0].tolist(),
            "pH Level": predicted_values[0, :, 1].tolist(),
            "Ammonia (ppm)": predicted_values[0, :, 2].tolist(),
            "Dissolved Oxygen (mg/L)": predicted_values[0, :, 3].tolist(),
            "Salinity (PSU)": predicted_values[0, :, 4].tolist()
        }
        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
