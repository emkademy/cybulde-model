#!/bin/bash

set -euo pipefail
IFS=$'\n\t'

export GCP_LOGGING_ENABLED="TRUE"

INSTANCE_GROUP_NAME=$(curl --silent --fail http://metadata.google.internal/computeMetadata/v1/instance/attributes/instance_group_name -H "Metadata-Flavor: Google")
ZONE=$(curl --silent http://metadata.google.internal/computeMetadata/v1/instance/attributes/zone -H "Metadata-Flavor: Google")
PYTHON_HASH_SEED=$(curl --silent --fail http://metadata.google.internal/computeMetadata/v1/instance/attributes/python_hash_seed -H "Metadata-Flavor: Google" || echo "42")
NODE_COUNT=$(curl --silent http://metadata.google.internal/computeMetadata/v1/instance/attributes/node_count -H "Metadata-Flavor: Google")
DISKS=$(curl --silent http://metadata.google.internal/computeMetadata/v1/instance/attributes/disks -H "Metadata-Flavor: Google")

INSTANCE_GROUP_NAME=$(echo ${INSTANCE_GROUP_NAME} | tr '[:upper:]' '[:lower:]')

echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
echo "HELLO WORLD!"
echo "INSTANCE_GROUP_NAME=${INSTANCE_GROUP_NAME}"
echo "ZONE=${ZONE}"
echo "PYTHON_HASH_SEED=${PYTHON_HASH_SEED}"
echo "NODE_COUNT=${NODE_COUNT}"
echo "DISKS=${DISKS}"
echo "EVERYTHING WENT WELL!!"
echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"

echo "Deleting instance group ${INSTANCE_GROUP_NAME}"
gcloud compute instance-groups managed delete --quiet "${INSTANCE_GROUP_NAME}" --zone "${ZONE}"
