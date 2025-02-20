import os  
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
    past_data = np.random.rand(1, 72, 5)  
    scaled_data = scaler.transform(past_data.reshape(-1, 5))  
    predictions = model.predict(scaled_data)  
    predicted_values = scaler.inverse_transform(predictions.reshape(-1, 5))  
    predicted_values = predicted_values.reshape(1, 72, 5)  

    response = {  
        "Temperature (°C)": predicted_values[0, :, 0].tolist(),  
        "pH Level": predicted_values[0, :, 1].tolist(),  
        "Ammonia (ppm)": predicted_values[0, :, 2].tolist(),  
        "Dissolved Oxygen (mg/L)": predicted_values[0, :, 3].tolist(),  
        "Salinity (PSU)": predicted_values[0, :, 4].tolist()  
    }  
    return jsonify(response)  

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))  # Render assigns a PORT dynamically
    print(f"Starting server on port {port}...")
    app.run(host='0.0.0.0', port=port, debug=True)

