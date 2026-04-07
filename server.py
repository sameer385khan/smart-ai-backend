from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.linear_model import LogisticRegression
import json
import os

app = Flask(__name__)
CORS(app)

DATA_FILE = "data.json"
SEQ = 3

# ===== LOAD DATA =====
if os.path.exists(DATA_FILE):
    with open(DATA_FILE, "r") as f:
        train_data = json.load(f)
else:
    train_data = [0,1,0,1,0,1,1,0,1]

# ===== TRAIN MODEL =====
def train_model(data):
    if len(data) < SEQ + 1:
        return None

    X, y = [], []
    for i in range(len(data) - SEQ):
        X.append(data[i:i+SEQ])
        y.append(data[i+SEQ])

    model = LogisticRegression()
    model.fit(X, y)
    return model

model = train_model(train_data)

# ===== API =====
@app.route("/")
def home():
    return "AI Backend Running"

@app.route("/predict", methods=["POST"])
def predict():
    global model, train_data

    try:
        data = request.json.get("data", [])

        if len(data) < SEQ:
            return jsonify({"error": "Need more data"})

        numeric = [0 if x=="dragon" else 1 for x in data]

        # Save data
        train_data.extend(numeric)
        train_data = train_data[-200:]

        with open(DATA_FILE, "w") as f:
            json.dump(train_data, f)

        # Retrain model
        model = train_model(train_data)

        # ML prediction
        last_seq = numeric[-SEQ:]
        ml_pred = model.predict([last_seq])[0]

        # Logic
        recent = data[-5:]
        trend = max(set(recent), key=recent.count)

        last = data[-1]
        streak = 1
        for i in range(len(data)-2, -1, -1):
            if data[i] == last:
                streak += 1
            else:
                break

        momentum = sum(numeric[-3:])
        reversal = streak >= 4

        score = 0
        score += 0.5 if ml_pred == 1 else -0.5
        score += 0.4 if trend == "tiger" else -0.4
        score += 0.3 if momentum >= 2 else -0.3

        if reversal:
            score *= -1

        prediction = "tiger" if score > 0 else "dragon"

        conf = abs(score)
        if conf > 0.8:
            confidence = "High 🔥"
        elif conf > 0.4:
            confidence = "Medium ⚡"
        else:
            confidence = "Low ⚠️"

        return jsonify({
            "prediction": prediction,
            "trend": trend,
            "streak": streak,
            "momentum": momentum,
            "reversal": reversal,
            "confidence": confidence
        })

    except Exception as e:
        return jsonify({"error": str(e)})

# ===== RUN SERVER =====
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)