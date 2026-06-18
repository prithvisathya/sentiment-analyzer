import pickle
import re
import boto3
import os
from flask import Flask, request, jsonify

# ── Load model from S3 ───────────────────────────────────────────
# On AWS Lambda there's no local filesystem with our model files
# so we download them from S3 at startup instead
s3 = boto3.client("s3", region_name="us-west-1")
BUCKET = "sentiment-analyzer-prithvi"

def load_from_s3(key):
    response = s3.get_object(Bucket=BUCKET, Key=key)
    return pickle.loads(response["Body"].read())

print("Loading model from S3...")
model = load_from_s3("model.pkl")
vectorizer = load_from_s3("vectorizer.pkl")
print("Model and vectorizer loaded successfully")

# ── 2. Create the Flask app ───────────────────────────────────────
# Flask is a lightweight web framework — it listens for HTTP requests
# and routes them to the right function based on the URL and method
app = Flask(__name__)

# ── 3. Preprocessing (same as train.py — must be identical) ──────
def preprocess(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ── 4. Define the /analyze endpoint ──────────────────────────────
# @app.route is a decorator — it tells Flask which URL triggers this function
# methods=["POST"] means this endpoint only accepts POST requests
# GET would be for fetching data, POST is for sending data to be processed
@app.route("/analyze", methods=["POST"])
def analyze():
    # request.get_json() parses the JSON body sent by the client
    data = request.get_json()

    # Validate the input — always check your inputs in real APIs
    if not data or "text" not in data:
        return jsonify({
            "error": "Missing 'text' field in request body"
        }), 400  # 400 = Bad Request

    text = data["text"]

    if not isinstance(text, str) or len(text.strip()) == 0:
        return jsonify({
            "error": "Text must be a non-empty string"
        }), 400

    # Run the ML pipeline
    clean = preprocess(text)
    vec = vectorizer.transform([clean])
    label = model.predict(vec)[0]
    confidence = float(model.predict_proba(vec).max())

    # Return a structured JSON response
    # jsonify() converts a Python dict to a proper JSON HTTP response
    return jsonify({
        "text": text,
        "sentiment": label,
        "confidence": round(confidence, 4),
        "status": "success"
    }), 200  # 200 = OK

# ── 5. Health check endpoint ──────────────────────────────────────
# Standard practice — lets you verify the API is running
# Genesys and most SaaS companies use these in production
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "model": "logistic_regression",
        "version": "1.0.0"
    }), 200

@app.route("/prod/analyze", methods=["POST"])
def analyze_prod():
    return analyze()

@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "404", "path": request.path}), 404 

# ── 6. Start the server ───────────────────────────────────────────
if __name__ == "__main__":
    # debug=True auto-reloads when you change the code
    # In production you'd set debug=False
    app.run(debug=True, port=5000)

# Required for AWS Lambda
from apig_wsgi import make_lambda_handler
handler = make_lambda_handler(app) 

    