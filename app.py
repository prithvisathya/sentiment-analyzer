import pickle
import re
import boto3
import os
from flask import Flask, request, jsonify

#Load model from S3
s3 = boto3.client("s3", region_name="us-west-1")
BUCKET = "sentiment-analyzer-prithvi"

def load_from_s3(key):
    response = s3.get_object(Bucket=BUCKET, Key=key)
    return pickle.loads(response["Body"].read())

print("Loading model from S3...")
model = load_from_s3("model.pkl")
vectorizer = load_from_s3("vectorizer.pkl")
print("Model and vectorizer loaded successfully")

#Create the Flask app
app = Flask(__name__)

#Preprocessing
def preprocess(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

@app.route("/analyze", methods=["POST"])
def analyze():
    # request.get_json() parses the JSON body sent by the client
    data = request.get_json()

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
    return jsonify({
        "text": text,
        "sentiment": label,
        "confidence": round(confidence, 4),
        "status": "success"
    }), 200  # 200 = OK

#Health check endpoint
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

if __name__ == "__main__":
    app.run(debug=True, port=5000)

# Required for AWS Lambda
from apig_wsgi import make_lambda_handler
handler = make_lambda_handler(app) 

    