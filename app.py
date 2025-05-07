import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle,joblib

# Create flask app
app = Flask(__name__)  

model = pickle.load(open("best_model.pkl", "rb"))
try:
    scaler = joblib.load("scaler.pkl")
    print("Scaler Loaded Successfully")
except Exception as e:
    print("Scaler Loading Error:", e)
@flask_app.route("/")
def Home():
    return render_template("index.html")

@flask_app.route("/predict", methods=["POST"])
def predict():
    try:
        # Debugging: Print input data
        form_data = request.form.to_dict()
        print("Raw Form Data:", form_data)

        # Convert input to float
        float_features = [float(x) for x in request.form.values()]
        
        # Feature Engineering: Add 'Contraction_Risk_Interaction'
        contraction_risk = float_features[0] * float_features[5]  # Count Contraction * Risk Factor Score
        float_features.append(contraction_risk)

        # Convert to numpy array
        features = np.array(float_features).reshape(1, -1)

        # Apply scaling
        features_scaled = scaler.transform(features)  

        print("Processed Features (Scaled):", features_scaled)

        # Model Prediction
        prediction = model.predict(features_scaled)
        print("Model Prediction:", prediction)

        return render_template("index.html", prediction_text=f"The baby is {prediction[0]} ({'Preterm' if prediction[0]==1 else 'Not Preterm'})")
    except Exception as e:
        print("Error in Prediction:", e)
        return render_template("index.html", prediction_text="Error in prediction!")
    
if __name__ == "__main__":
    flask_app.run(debug=True)
