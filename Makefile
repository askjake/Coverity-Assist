# Makefile for Coverity-Assist X App image

REGION   ?= us-west-2
ACCOUNT  ?= 233532778289
REPO     ?= coverity-assist
ECR      ?= $(ACCOUNT).dkr.ecr.$(REGION).amazonaws.com/$(REPO)
TAG      ?= 0.0.1
IMAGE    ?= $(ECR):$(TAG)

.PHONY: login image push run-local

login:
	aws ecr get-login-password --region $(REGION) \
	  | docker login --username AWS --password-stdin \
	    $(ACCOUNT).dkr.ecr.$(REGION).amazonaws.com

image:
	docker build -t $(IMAGE) .

push: image
	$(MAKE) login
	docker push $(IMAGE)

run-local:
	uvicorn app:app --host 0.0.0.0 --port 8000 --reload
