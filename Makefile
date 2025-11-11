ECR_REPO   := 233532778289.dkr.ecr.us-west-2.amazonaws.com/coverity-assist
AWS_REGION := us-west-2

# You can override TAG on the command line, e.g. `make TAG=dev1 docker-push`
TAG ?= dev
IMAGE_TAG := $(ECR_REPO):$(TAG)

docker-build:
	docker build -t $(IMAGE_TAG) .

docker-login:
	aws ecr get-login-password --region $(AWS_REGION) \
	  | docker login --username AWS --password-stdin 233532778289.dkr.ecr.us-west-2.amazonaws.com

docker-push: docker-build docker-login
	docker push $(IMAGE_TAG)

print-image:
	@echo $(IMAGE_TAG)
