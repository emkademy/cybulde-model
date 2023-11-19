#!/usr/bin/env bash

set -o errexit
set -o pipefail
set -o nounset

if [[ "${IS_PROD_ENV}" == "true" ]]; then
	/usr/local/gcloud/google-cloud-sdk/bin/gcloud compute ssh "${VM_NAME}" --zone "${ZONE}" --tunnel-through-iap -- -4 -N -L ${PROD_MLFLOW_SERVER_PORT}:localhost:${PROD_MLFLOW_SERVER_PORT}
else
	/start-prediction-service.sh &
	/start-tracking-server.sh &
	tail -F anything
fi
