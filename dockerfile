FROM python:3.10-slim-buster

WORKDIR /python-docker

RUN apt-get update 

COPY . /app
WORKDIR /app

# COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

EXPOSE 8000

CMD ["gunicorn", "app:app"]