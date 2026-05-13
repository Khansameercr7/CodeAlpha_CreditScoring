# Deployment Guide

## Quick Deployment Options

This guide covers multiple ways to deploy the Credit Scoring system to production.

---

## Option 1: Streamlit Cloud (Recommended for Dashboard)

**Best for:** Quick, free deployment of the interactive dashboard

### Prerequisites
- GitHub account with repository pushed
- Streamlit account (free)

### Steps

1. **Go to Streamlit Cloud**
   ```
   https://streamlit.io/cloud
   ```

2. **Sign in with GitHub**
   - Click "New app"
   - Select your repository: `CodeAlpha_CreditScoring`
   - Main file path: `app.py`
   - Python version: 3.9+

3. **Configure Secrets**
   - Create `.streamlit/secrets.toml` in your repo:
   ```toml
   data_path = "data/credit_data.csv"
   model_path = "outputs/models"
   ```

4. **Deploy**
   - Click "Deploy"
   - Your dashboard will be live at: `https://[your-app-name].streamlit.app`

5. **Sharing**
   - Share the URL with stakeholders
   - Anyone can access the live dashboard

### Cost
- **Free tier:** Limited computational resources, perfect for demos
- **Pro tier:** Enhanced performance (optional)

---

## Option 2: Docker Container (Recommended for Production)

**Best for:** Containerized deployment to any cloud platform

### Step 1: Create Dockerfile

Create `Dockerfile` in project root:

```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Expose ports
EXPOSE 8501 8000

# Set environment variables
ENV STREAMLIT_SERVER_PORT=8501
ENV PYTHONUNBUFFERED=1

# Run Streamlit by default
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### Step 2: Create docker-compose.yml

```yaml
version: '3.8'

services:
  streamlit-dashboard:
    build: .
    ports:
      - "8501:8501"
    volumes:
      - ./data:/app/data
      - ./outputs:/app/outputs
    environment:
      - STREAMLIT_SERVER_PORT=8501
```

### Step 3: Build and Run Locally

```bash
# Build image
docker build -t credit-scoring:latest .

# Run container
docker run -p 8501:8501 credit-scoring:latest
```

Visit `http://localhost:8501`

### Step 4: Push to Container Registry

```bash
# Tag image
docker tag credit-scoring:latest [your-username]/credit-scoring:latest

# Login to Docker Hub
docker login

# Push image
docker push [your-username]/credit-scoring:latest
```

### Step 5: Deploy to Cloud

**AWS ECS:**
```bash
# Push to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin [account].dkr.ecr.us-east-1.amazonaws.com
docker tag credit-scoring [account].dkr.ecr.us-east-1.amazonaws.com/credit-scoring
docker push [account].dkr.ecr.us-east-1.amazonaws.com/credit-scoring

# Create ECS service pointing to image
```

**Google Cloud Run:**
```bash
gcloud run deploy credit-scoring --image gcr.io/[project]/credit-scoring --platform managed
```

**Azure Container Instances:**
```bash
az container create --resource-group mygroup --name credit-scoring --image [username]/credit-scoring --port 8501
```

---

## Option 3: REST API with FastAPI (Recommended for Integration)

**Best for:** Server-to-server integration, mobile/web apps

### Step 1: Create API Server

Create `api_server.py`:

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
from src.models.predictor import CreditPredictor
from src.utils.model_io import load_all

app = FastAPI(title="Credit Scoring API", version="1.0.0")

# Load model at startup
models, scaler, feature_names = load_all("outputs/models")
predictor = CreditPredictor(
    models["XGBoost"],
    scaler,
    feature_names
)

class ApplicantData(BaseModel):
    """Loan applicant information"""
    person_age: int
    person_income: int
    person_home_ownership: str
    person_emp_length: float
    loan_intent: str
    loan_grade: str
    loan_amnt: int
    loan_int_rate: float
    loan_pct_income: float
    cb_person_default_on_file: int
    cb_preson_cred_history_length: int

class PredictionResponse(BaseModel):
    """Prediction result"""
    default_probability: float
    risk_score: float
    risk_band: str
    recommendation: str
    key_risk_factors: list

@app.get("/health")
def health():
    """Health check endpoint"""
    return {"status": "healthy", "model": "credit-scoring-v1"}

@app.post("/predict", response_model=PredictionResponse)
def predict(applicant: ApplicantData):
    """
    Predict credit risk for an applicant
    
    Parameters:
    -----------
    applicant : ApplicantData
        Applicant financial and demographic information
    
    Returns:
    --------
    PredictionResponse
        Default probability, risk score, recommendation
    """
    try:
        result = predictor.predict(
            applicant.dict(),
            threshold=0.5
        )
        
        return PredictionResponse(
            default_probability=result["probability"],
            risk_score=result["risk_score"],
            risk_band=result["risk_band"],
            recommendation=result["recommendation"],
            key_risk_factors=result["key_risk_factors"]
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/batch-predict")
def batch_predict(applicants: list[ApplicantData]):
    """
    Predict credit risk for multiple applicants
    """
    results = []
    for applicant in applicants:
        result = predictor.predict(applicant.dict(), threshold=0.5)
        results.append(result)
    return {"predictions": results, "count": len(results)}

@app.get("/model-info")
def model_info():
    """Get model metadata"""
    return {
        "model": "XGBoost",
        "roc_auc": 0.937,
        "f1_score": 0.758,
        "accuracy": 0.888,
        "features": len(feature_names),
        "training_data": "32,581 loan applications"
    }
```

### Step 2: Create Docker for API

Update `Dockerfile`:

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt fastapi uvicorn

COPY . .

EXPOSE 8000

CMD ["uvicorn", "api_server:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Step 3: Run API

```bash
# Local testing
uvicorn api_server:app --reload

# Production
gunicorn -w 4 -b 0.0.0.0:8000 api_server:app
```

### Step 4: Test API

```bash
# Health check
curl http://localhost:8000/health

# Single prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "person_age": 35,
    "person_income": 85000,
    "person_home_ownership": "RENT",
    "person_emp_length": 10,
    "loan_intent": "PERSONAL",
    "loan_grade": "A",
    "loan_amnt": 15000,
    "loan_int_rate": 5.5,
    "loan_pct_income": 0.18,
    "cb_person_default_on_file": 0,
    "cb_preson_cred_history_length": 8
  }'

# Batch prediction
curl -X POST http://localhost:8000/batch-predict \
  -H "Content-Type: application/json" \
  -d '[...]'
```

---

## Option 4: AWS Lambda (Serverless)

**Best for:** Scalable, pay-per-use deployment

### Step 1: Prepare Lambda Package

```bash
pip install -r requirements.txt -t lambda_deployment/

# Copy Lambda handler
cp lambda_handler.py lambda_deployment/

# Zip everything
cd lambda_deployment
zip -r ../lambda_function.zip .
```

### Step 2: Create Lambda Handler

Create `lambda_handler.py`:

```python
import json
import joblib
from src.models.predictor import CreditPredictor

# Load model once at cold start
models, scaler, feature_names = joblib.load('models.pkl'), joblib.load('scaler.pkl'), joblib.load('features.pkl')
predictor = CreditPredictor(models["XGBoost"], scaler, feature_names)

def lambda_handler(event, context):
    """AWS Lambda handler for credit prediction"""
    try:
        applicant = json.loads(event['body'])
        result = predictor.predict(applicant, threshold=0.5)
        
        return {
            'statusCode': 200,
            'body': json.dumps(result)
        }
    except Exception as e:
        return {
            'statusCode': 400,
            'body': json.dumps({'error': str(e)})
        }
```

### Step 3: Deploy to AWS

```bash
# Create Lambda function
aws lambda create-function \
  --function-name credit-scoring \
  --runtime python3.9 \
  --role arn:aws:iam::ACCOUNT:role/lambda-role \
  --handler lambda_handler.lambda_handler \
  --zip-file fileb://lambda_function.zip

# Create API Gateway endpoint
# (Use AWS Console for easier setup)
```

---

## Option 5: Google Cloud Vertex AI

**Best for:** Enterprise ML deployment with monitoring

```bash
# Submit training job
gcloud ai custom-jobs create \
  --region us-central1 \
  --training-container-image-uri gcr.io/[project]/credit-scoring:latest \
  --config train_config.yaml

# Deploy model endpoint
gcloud ai endpoints create credit-scoring-endpoint --region us-central1
gcloud ai endpoints deploy-model credit-scoring-endpoint \
  --deployed-model-display-name credit-scoring-v1 \
  --model projects/[project]/locations/us-central1/models/[model-id]
```

---

## Environment Variables

Create `.env` for all deployment options:

```bash
# Database
DATABASE_URL=postgresql://user:password@localhost/credit_db

# Model paths
MODEL_PATH=outputs/models
DATA_PATH=data/credit_data.csv

# API
API_HOST=0.0.0.0
API_PORT=8000

# Logging
LOG_LEVEL=INFO
LOG_FILE=logs/app.log

# Monitoring
SENTRY_DSN=https://...
```

---

## Monitoring and Logging

### Basic Monitoring

```python
import logging
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

@app.post("/predict")
def predict(applicant: ApplicantData):
    try:
        result = predictor.predict(applicant.dict())
        logger.info(f"Prediction: {result['risk_band']}")
        return result
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise
```

### Advanced Monitoring (Production)

- **Sentry:** Error tracking
- **DataDog:** Performance metrics
- **Prometheus:** System monitoring
- **ELK Stack:** Log aggregation

---

## Performance Optimization

### Model Optimization

```python
# Use ONNX for faster inference
from skl2onnx import convert_sklearn
from onnxruntime import InferenceSession

# Convert model
onnx_model = convert_sklearn(xgboost_model)
sess = InferenceSession("model.onnx")

# Inference
results = sess.run(None, {"X": features_array})
```

### Caching

```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def get_features(applicant_hash):
    """Cache feature engineering results"""
    pass
```

---

## Cost Comparison

| Platform | Cost | Best For |
|----------|------|----------|
| Streamlit Cloud | Free (+ Pro) | Quick demos |
| Docker + EC2 | ~$5-50/month | Full control |
| Lambda | Pay-per-request | Sporadic usage |
| Cloud Run | ~$10-100/month | Balanced |
| Vertex AI | ~$50-500/month | Enterprise |

---

## Deployment Checklist

- [ ] Code tested locally
- [ ] All dependencies in requirements.txt
- [ ] Environment variables configured
- [ ] Model files present
- [ ] Logging configured
- [ ] Error handling in place
- [ ] Security measures (API keys, data privacy)
- [ ] Monitoring setup
- [ ] Backup/recovery plan
- [ ] Documentation updated
- [ ] Performance tested
- [ ] Scaling plan in place

---

## Troubleshooting

### Model Loading Fails
```bash
# Check model files exist
ls -la outputs/models/

# Verify joblib version compatibility
pip install --upgrade joblib
```

### Out of Memory
- Use batch processing instead of loading all data
- Consider model quantization
- Increase container memory limits

### Slow Predictions
- Optimize feature engineering
- Use caching
- Consider model distillation
- Check computational resources

---

**Choose the deployment option that best fits your needs:**
- Learning/Demo → Streamlit Cloud
- Production Dashboard → Docker + Cloud Run
- API Integration → FastAPI + Lambda
- Enterprise → Vertex AI or Kubernetes

