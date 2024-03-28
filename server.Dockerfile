FROM python:3.9-slim

WORKDIR /workspace

RUN apt-get update && apt-get install -y git 

ADD requirements.txt .

RUN pip3 install -r requirements.txt
ADD . .
ENV GOOGLE_APPLICATION_CREDENTIALS=gcs.json

CMD python3 server.py