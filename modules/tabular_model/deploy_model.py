
from flask import Flask, request, jsonify, render_template_string
import joblib
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)


def load_model():
    processed_dir = "data/processed/labs_clinical/arthritis_profile_processed"
    model = joblib.load(os.path.join(processed_dir, "trained_model.pkl"))
    scaler = joblib.load(os.path.join(processed_dir, "scaler.pkl"))
    return model, scaler

model, scaler = load_model()

def predict_arthritis(data_df):
    features = ['ESRh', 'ESRo', 'CRP', 'Uric_Acid', 'ASO', 'age_binned', 'sex_encoded']
    X_new = data_df[features]
    X_new_scaled = scaler.transform(X_new)
    predictions = model.predict(X_new_scaled)
    probabilities = model.predict_proba(X_new_scaled)[:, 1]
    return int(predictions[0]), float(probabilities[0])



HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>Arthritis Prediction</title>
    <style>
        body {
            font-family: 'Segoe UI', sans-serif;
            background: linear-gradient(to right, #e3f2fd, #bbdefb);
            text-align: center;
            padding: 40px;
        }
        .container {
            background: white;
            width: 400px;
            margin: auto;
            padding: 25px;
            border-radius: 15px;
            box-shadow: 0 0 15px rgba(0,0,0,0.2);
        }
        input {
            width: 80%;
            padding: 10px;
            margin: 8px;
            border-radius: 5px;
            border: 1px solid #aaa;
        }
        button {
            background: #1565c0;
            color: white;
            padding: 10px 20px;
            border-radius: 8px;
            border: none;
            cursor: pointer;
        }
        button:hover {
            background: #0d47a1;
        }
        .result {
            margin-top: 20px;
            font-size: 18px;
        }
        .emoji {
            font-size: 45px;
        }
        .bar {
            width: 80%;
            background-color: #ddd;
            border-radius: 10px;
            margin: auto;
        }
        .bar-inner {
            height: 20px;
            border-radius: 10px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h2>🩺 Arthritis Prediction App</h2>
        <form action="/predict" method="post">
            <input type="number" step="any" name="ESRh" placeholder="ESRh" required><br>
            <input type="number" step="any" name="ESRo" placeholder="ESRo" required><br>
            <input type="number" step="any" name="CRP" placeholder="CRP" required><br>
            <input type="number" step="any" name="Uric_Acid" placeholder="Uric Acid" required><br>
            <input type="number" step="any" name="ASO" placeholder="ASO" required><br>
            <input type="number" name="age_binned" placeholder="Age Binned" required><br>
            <input type="number" name="sex_encoded" placeholder="Sex Encoded (0=F,1=M)" required><br>
            <button type="submit">🔍 Predict</button>
        </form>

        {% if result %}
        <div class="result">
            <h3>🧠 Arthritis Prediction Result</h3>
            <p>Prediction: <b>{{ result.pred_text }}</b> <span class="emoji">{{ result.emoji }}</span></p>
            
            <div class="bar">
                <div class="bar-inner" 
                     style="width: {{ (result.probability * 100)|round(0) }}%; 
                            background-color: {{ result.bar_color }};">
                </div>
            </div>

            <p>Probability: {{ result.probability }}</p>
            <br>
            <a href="/">🔁 Try Again</a>
        </div>
        {% endif %}
    </div>
</body>
</html>
"""



@app.route('/', methods=['GET'])
def home():
    return render_template_string(HTML_PAGE)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = {col: float(request.form[col]) for col in request.form}
        data_df = pd.DataFrame([data])
        pred, prob = predict_arthritis(data_df)
        pred_text = "Arthritis Detected" if pred == 1 else "No Arthritis"
        emoji = "✅" if pred == 1 else "❌"
        bar_color = "#d32f2f" if prob > 0.6 else "#4caf50"
        result = {"pred_text": pred_text, "emoji": emoji, "probability": round(prob, 2), "bar_color": bar_color}
        return render_template_string(HTML_PAGE, result=result)
    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == '__main__':
    app.run(debug=True)
