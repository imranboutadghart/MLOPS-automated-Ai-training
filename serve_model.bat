@echo off
REM Simple script to serve model locally with MLflow

echo Starting MLflow Model Server...
echo.

set MLFLOW_TRACKING_URI=http://localhost:5000
set AWS_ACCESS_KEY_ID=minioadmin
set AWS_SECRET_ACCESS_KEY=minioadmin
set MLFLOW_S3_ENDPOINT_URL=http://localhost:9000

REM Serve the production model
mlflow models serve -m "models:/titanic_classifier/Production" -p 8000 --env-manager local --no-conda

echo.
echo Model server stopped.
pause
