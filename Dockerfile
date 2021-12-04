FROM ubuntu:20.04

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y python3-pip libglib2.0-0 libgl1
RUN mkdir /app
COPY . /app/
WORKDIR /app
RUN pip3 install -r requirements.txt
CMD ["bash", "start.sh"]
