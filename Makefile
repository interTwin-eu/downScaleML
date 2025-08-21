# Makefile for Docker-based openEO downScaleML setup

IMAGE_NAME := front  # Change to 'new' if needed
TEST_FILE ?= /app/tests/
AWS_ENV_VARS := -e AWS_ACCESS_KEY_ID=$(AWS_ACCESS_KEY_ID) -e AWS_SECRET_ACCESS_KEY=$(AWS_SECRET_ACCESS_KEY)
VOLUME_MAP := -v $(PWD)/app/test_data:/app/test_data


.PHONY: build run test setup clean shell

## Build the Docker image
build:
	docker build -t $(IMAGE_NAME) .

## Setup: build the image and run the tests
setup: build run

## Run a specific test (make test FILE=/app/tests/my_test.py)
run:
	docker run -it $(AWS_ENV_VARS) $(VOLUME_MAP) $(IMAGE_NAME) pytest /app/tests/pytest_private_seas5.py -v -s

## Open an interactive shell with micromamba env activated
shell:
	docker run -it $(AWS_ENV_VARS) $(VOLUME_MAP) $(IMAGE_NAME) bash

## Clean all containers based on the image
clean:
	-docker ps -a -q --filter ancestor=$(IMAGE_NAME) | xargs -r docker rm -f
