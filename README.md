# Customer Sentiment Analyzer API

A REST API that classifies the sentiment of text as positive or negative using a logistic regression model trained on 25,000 IMDB reviews.

## Tech Stack
- Python, scikit-learn, Flask
- TF-IDF vectorization
- Deployed on AWS Lambda + API Gateway

## API Usage

### Analyze sentiment
POST /analyze

Request:
{
  "text": "This product is absolutely fantastic"
}

Response:
{
  "sentiment": "positive",
  "confidence": 0.91,
  "status": "success",
  "text": "This product is absolutely fantastic"
}

### Health check
GET /health

## Run locally

1. Clone the repo
2. Create a virtual environment: python3 -m venv venv
3. Activate it: source venv/bin/activate
4. Install dependencies: pip install -r requirements.txt
5. Train the model: python3 train.py
6. Start the API: python3 app.py
7. API runs at http://127.0.0.1:5000

## Model Performance
- Dataset: 25,000 IMDB movie reviews
- Accuracy: 82.48%
- Algorithm: Logistic Regression with TF-IDF features
