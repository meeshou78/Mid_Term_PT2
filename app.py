# app.py

from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load model and label mapping
model = joblib.load("model.pkl")
mapping = joblib.load("race_mapping.pkl")

@app.route("/", methods=["GET", "POST"])
def predict():
    prediction = None
    if request.method == "POST":
        try:
            math = float(request.form["math"])
            reading = float(request.form["reading"])
            writing = float(request.form["writing"])
            
            input_data = np.array([[math, reading, writing]])
            pred_code = model.predict(input_data)[0]
            prediction = mapping[pred_code]
        except Exception as e:
            prediction = f"Error: {str(e)}"

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
