
# Task 15: End-to-End Deep Learning Production Deployment

## Project Overview

An IT Incident Classification System using TensorFlow/Keras,
Flask, Streamlit, Docker, and Kubernetes.

The model classifies incident descriptions into Network,
Hardware, Software, Security, and Access categories.

## Architecture

Streamlit UI -> Flask REST API -> Saved tokenizer -> LSTM model

Docker packages the services. Kubernetes deploys and manages
the application containers.

## Prerequisites

- Python 3.11
- Docker Desktop
- Minikube
- kubectl
- Git (optional)

## Install

Create and activate a virtual environment:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Train the model

```powershell
python train_model.py
```

Training generates the model, tokenizer, category labels,
evaluation metrics, training history, and confusion matrix.

The included data is illustrative. Its evaluation results
must not be treated as production performance.

## Run locally

Terminal 1:

```powershell
python app.py
```

Terminal 2:

```powershell
$env:API_URL="http://127.0.0.1:5000"
streamlit run streamlit_app.py
```

Open http://localhost:8501.

## Run tests

```powershell
python -m unittest discover -s tests -v
```

## Docker

```powershell
docker compose -f compose.yaml build
docker compose -f compose.yaml up -d
docker compose -f compose.yaml ps
```

Open http://localhost:8501.

Stop containers:

```powershell
docker compose -f compose.yaml down
```

## Kubernetes with Minikube

Start a cluster:

```powershell
minikube start --cpus=4 --memory=6144
kubectl get nodes
```

Build images:

```powershell
docker build -f Dockerfile.api -t incident-api:1.0 .
docker build -f Dockerfile.ui -t incident-ui:1.0 .
```

Load images:

```powershell
minikube image load incident-api:1.0
minikube image load incident-ui:1.0
```

Deploy:

```powershell
kubectl apply -f k8s/api.yaml
kubectl apply -f k8s/ui.yaml
```

Verify:

```powershell
kubectl get deployments
kubectl get pods
kubectl get services
kubectl rollout status deployment/incident-api
kubectl rollout status deployment/incident-ui
```

Open the application:

```powershell
minikube service incident-ui
```

## Scaling and monitoring

```powershell
kubectl scale deployment/incident-api --replicas=3
kubectl get pods
kubectl describe deployment incident-api
kubectl logs deployment/incident-api --tail=100
kubectl top pods
```

If metrics are unavailable:

```powershell
minikube addons enable metrics-server
```

## Rolling update

Build and load a new API image:

```powershell
docker build -f Dockerfile.api -t incident-api:1.1 .
minikube image load incident-api:1.1
kubectl set image deployment/incident-api api=incident-api:1.1
kubectl rollout status deployment/incident-api
```

Rollback:

```powershell
kubectl rollout undo deployment/incident-api
```

## Deliverables

- Training source code
- Trained model and preprocessing artifacts
- Flask API
- Streamlit frontend
- Docker configuration
- Kubernetes manifests
- Automated API tests
- Evaluation reports and screenshots
