#!/usr/bin/env bash

set -o errexit
set -o pipefail
set -o nounset

if [[ "${IS_PROD_ENV}" == "true" ]]; then
  echo "production service"
else
  /start-tracking-server.sh &
  tail -F anything
fi

