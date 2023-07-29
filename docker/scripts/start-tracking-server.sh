#!/bin/bash

mlflow server \
  --backend-store-uri "${MLFLOW_BACKEND_STORE}" \
  --default-artifact-root "${MLFLOW_ARTIFACT_STORE}" \
  --host 0.0.0.0 \
  --port "${LOCAL_DEV_MLFLOW_SERVER_PORT}"
