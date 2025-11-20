#!/usr/bin/env bash
set -euo pipefail

BASE="233532778289.dkr.ecr.us-west-2.amazonaws.com/coverity-assist:dev1"
TAG="233532778289.dkr.ecr.us-west-2.amazonaws.com/coverity-assist:dev-$(date -u +%Y%m%d-%H%M)-overlay"

# login (stderr for chatter)
aws ecr get-login-password --region us-west-2 \
  | docker login --username AWS --password-stdin 233532778289.dkr.ecr.us-west-2.amazonaws.com 1>&2

# build + push (stderr for chatter)
docker build -t "$TAG" --build-arg BASE="$BASE" -f Dockerfile . 1>&2
docker push "$TAG" 1>&2

# Print only the tag so you can do: IMG=$(./redeploy.sh)
echo "$TAG"
