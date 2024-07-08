FROM python:latest as firstBase

RUN apt-get update && apt-get install -y python3-pip gdal-bin libgdal-dev

RUN python3 -m pip install earthdaily








