FROM ubuntu:18.04

WORKDIR /modestpy

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update
RUN apt-get install -y libgfortran3 gcc g++
RUN apt-get install -y python3 python3-pip
RUN apt-get install -y libjpeg8-dev zlib1g-dev

COPY . .
RUN python3 -m pip install -U pip
RUN python3 -m pip install .
ENTRYPOINT ["/bin/bash"]
