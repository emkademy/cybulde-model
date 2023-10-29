#!/bin/bash
set -euo pipefail
IFS=$'\n\t'

export NCCL_ASYNC_ERROR_HANDLING=1
export GCP_LOGGING_ENABLED="TRUE"

INSTANCE_GROUP_NAME=$(curl --silent --fail http://metadata.google.internal/computeMetadata/v1/instance/attributes/instance_group_name -H "Metadata-Flavor: Google" || echo "42")
DOCKER_IMAGE=$(curl --silent --fail http://metadata.google.internal/computeMetadata/v1/instance/attributes/docker_image -H "Metadata-Flavor: Google" || echo "42")
ZONE=$(curl --silent http://metadata.google.internal/computeMetadata/v1/instance/attributes/zone -H "Metadata-Flavor: Google")
PYTHON_HASH_SEED=$(curl --silent --fail http://metadata.google.internal/computeMetadata/v1/instance/attributes/python_hash_seed -H "Metadata-Flavor: Google" || echo "42")
MLFLOW_TRACKING_URI=$(curl --silent http://metadata.google.internal/computeMetadata/v1/instance/attributes/mlflow_tracking_uri -H "Metadata-Flavor: Google")
NODE_COUNT=$(curl --silent http://metadata.google.internal/computeMetadata/v1/instance/attributes/node_count -H "Metadata-Flavor: Google")
DISKS=$(curl --silent http://metadata.google.internal/computeMetadata/v1/instance/attributes/disks -H "Metadata-Flavor: Google")

echo -e "TRAINING: instance group name: ${INSTANCE_GROUP_NAME}, docker image: ${DOCKER_IMAGE}, node count: ${NODE_COUNT}, python hash seed: ${PYTHON_HASH_SEED}"

echo "============= Installing Nvidia Drivers ==============="
apt-get update && /opt/deeplearning/install-driver.sh

echo "============= Downloading docker image ==============="
gcloud auth configure-docker --quiet europe-west4-docker.pkg.dev
time docker pull "${DOCKER_IMAGE}"

echo "============= TRAINING: start ==============="
docker run --init --rm --gpus all --ipc host --user root --hostname "$(hostname)" --privileged \
	--log-driver=gcplogs \
	-e PYTHONHASHSEED="${PYTHON_HASH_SEED}" \
	-e MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI}" \
	-e TOKENIZERS_PARALLELISM=false \
	${DOCKER_IMAGE} \
	torchrun \
	--nnodes="${NODE_COUNT}" \
	--nproc_per_node='gpu' \
	cybulde/run_tasks.py || echo '================ TRAINING: job failed ==============='

echo "============= Cleaning up ==============="
gcloud compute instance-groups managed delete --quiet "${INSTANCE_GROUP_NAME}" --zone "${ZONE}"
