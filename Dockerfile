FROM python:3.8.12-buster

RUN pip3  install --upgrade pip


RUN pip3 install pytest


COPY . workdir
WORKDIR workdir

CMD pytests tests
