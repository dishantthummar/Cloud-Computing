# app.py

from flask import Flask, request, render_template
import numpy as np
import joblib

# Load the trained model
model_path = 'model.pkl'
model = joblib.load(model_path)

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form input values
        cgpa = float(request.form.get('cgpa', 0))
        iq = float(request.form.get('iq', 0))

        # Combine features into numpy array
        features = np.array([[cgpa, iq]])

        # Make prediction
        prediction = model.predict(features)[0]

        # Map prediction to label
        result = 'Placed ' if prediction == 1 else 'Not Placed '

        return render_template('index.html', prediction_text=f'Prediction: {result}')

    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')

if __name__ == "__main__":
    app.run(debug=True)
