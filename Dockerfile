FROM ubuntu:20.04
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update -y && \
    apt-get install -y python3-pip python3.9-dev

COPY ./requirements.txt .
RUN mkdir models
WORKDIR models
RUN mkdir bert-base-cased
WORKDIR bert-base-cased
COPY models/bert-base-cased/checkpoint-1910 ./checkpoint-1910
WORKDIR /
RUN mkdir data
WORKDIR data
COPY data/taxonomy_mappings.json .
WORKDIR /
COPY predict.py .
COPY preprocess.py .
COPY train.py .
COPY flask_app.py .

RUN pip3 install -r requirements.txt

ENTRYPOINT ["python3"]
CMD ["flask_app.py"]