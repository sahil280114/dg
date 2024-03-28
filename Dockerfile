FROM runpod/pytorch:3.10-2.0.1-120-devel

WORKDIR /workspace

RUN apt-get update && apt-get install -y git python3 python3-pip

ADD requirements.txt .

RUN pip3 install -r requirements.txt

ADD . .

ENV GOOGLE_APPLICATION_CREDENTIALS=gcs.json


CMD python3 main.py