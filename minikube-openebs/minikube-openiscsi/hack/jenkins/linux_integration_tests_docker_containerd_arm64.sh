#!/bin/bash

# Copyright 2019 The Kubernetes Authors All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# This script runs the integration tests on a Linux machine for the KVM Driver

# The script expects the following env variables:
# MINIKUBE_LOCATION: GIT_COMMIT from upstream build.
# COMMIT: Actual commit ID from upstream build
# EXTRA_BUILD_ARGS (optional): Extra args to be passed into the minikube integrations tests
# access_token: The Github API access token. Injected by the Jenkins credential provider.
set -e

mkdir -p cron && gsutil -qm rsync "gs://minikube-builds/${MINIKUBE_LOCATION}/cron" cron || echo "FAILED TO GET CRON FILES"
sudo install cron/cleanup_and_reboot_Linux.sh /etc/cron.hourly/cleanup_and_reboot || echo "FAILED TO INSTALL CLEANUP"

# removing possible left over docker containers from previous runs
docker rm -f -v "$(docker ps -aq)" >/dev/null 2>&1 || true

ARCH="arm64" \
OS="linux" \
DRIVER="docker" \
JOB_NAME="$JOB_NAME" \
CONTAINER_RUNTIME="containerd" \
EXTERNAL="yes" \
source ./common.sh
