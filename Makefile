.PHONY: build stop rm run bash test

build: stop rm
	DOCKER_BUILDKIT=1 docker build -t modestpy .

stop:
	-docker stop modestpy_container

rm:
	-docker rm modestpy_container

run:
	docker run \
		-t \
		-d \
		--name modestpy_container \
		modestpy

bash:
	docker exec -ti modestpy_container bash

test:
	docker exec -ti modestpy_container python3 modestpy/test/run.py
