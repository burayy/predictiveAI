from flask import Flask, jsonify, request
import numpy as np
import tensorflow as tf
import joblib

# Load the model and scaler
model = tf.keras.models.load_model('model.h5')
scaler = joblib.load('scaler.pkl')

app = Flask(__name__)

@app.route('/predict', methods=['GET'])
def predict():
    # Simulate receiving past sensor data from a request (in reality, you might use request.args or request.json)
    # For this example, we'll simulate with random values
    past_data = np.random.rand(1, 72, 5)  # Assuming 72 hours of data, with 5 features

    # Scale the input data
    scaled_data = scaler.transform(past_data.reshape(-1, 5))

    # Make predictions (72 hours into the future)
    predictions = model.predict(scaled_data)
    
    # Inverse scale the predictions back to original scale
    predicted_values = scaler.inverse_transform(predictions.reshape(-1, 5))

    # Reshape for 72 hours prediction
    predicted_values = predicted_values.reshape(1, 72, 5)

    # Prepare the output in a format suitable for the frontend
    response = {
        "Temperature (°C)": predicted_values[0, :, 0].tolist(),
        "pH Level": predicted_values[0, :, 1].tolist(),
        "Ammonia (ppm)": predicted_values[0, :, 2].tolist(),
        "Dissolved Oxygen (mg/L)": predicted_values[0, :, 3].tolist(),
        "Salinity (PSU)": predicted_values[0, :, 4].tolist()
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
