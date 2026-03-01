import os
import sys
import torch
import joblib
import numpy as np
from flask import Flask, request, jsonify, render_template

# Add models directory to path so app can access FraudClassifier
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from models.classifier import FraudClassifier

app = Flask(__name__, 
            template_folder='../frontend/templates', 
            static_folder='../frontend/static')

# Initialize variables
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'fraud_model.pth')
SCALER_PATH = os.path.join(os.path.dirname(__file__), 'scaler.pkl')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_features = 30  # As per the dataset (Time, V1..V28, Amount)

# Load model and scaler
model = FraudClassifier(input_dim=num_features).to(device)
scaler = None

if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    scaler = joblib.load(SCALER_PATH)
    print("Model and Scaler loaded successfully.")
else:
    print("WARNING: Model or Scaler not found. You need to train the classifier first.")

@app.route("/")
def home():
    """Renders the payment application"""
    return render_template('payment.html')

@app.route("/analysis")
def analysis():
    """Renders the frontend application"""
    return render_template('index.html')

@app.route("/predict", methods=["POST"])
def predict():
    """Accepts JSON transaction features and returns fraud probability and prediction"""
    if scaler is None:
        return jsonify({"error": "Model or Scaler not loaded on server."}), 500
        
    try:
        data = request.json
        features = data.get('features')
        payer_ip = data.get('payer_ip', request.remote_addr)
        
        print(f"\\n[PREDICTION REQUEST] Analyzing transaction from IP: {payer_ip}")
        
        if not features or len(features) != num_features:
            return jsonify({
                "error": f"Invalid input. Expected an array of {num_features} numerical features."
            }), 400
            
        # Convert to numpy array and scale
        features_arr = np.array(features).reshape(1, -1)
        features_scaled = scaler.transform(features_arr)
        
        # Convert to Pytorch Tensor
        features_tensor = torch.Tensor(features_scaled).to(device)
        
        # Prediction
        with torch.no_grad():
            probability = model(features_tensor).item()  # float probability
            
        prediction_label = "Fraud" if probability > 0.5 else "Normal"
        print(f"-> Prediction Result for IP {payer_ip}: {prediction_label} ({probability:.4f})\\n")
        
        return jsonify({
            "fraud_probability": round(probability, 4),
            "prediction": prediction_label,
            "payer_ip_logged": payer_ip
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)
