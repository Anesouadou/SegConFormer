FROM nvidia/cuda:11.2.2-cudnn8-runtime-ubuntu20.04

RUN chmod 1777 /tmp && chmod 1777 /var/tmp
RUN apt-get update
RUN apt-get upgrade -y 
RUN apt-get install -y vim curl ca-certificates amqp-tools wget graphviz
RUN apt-get -y install git

RUN apt-get install -y python3
RUN apt-get install -y python3-pip

COPY requirements.txt .
RUN pip install -r requirements.txt
