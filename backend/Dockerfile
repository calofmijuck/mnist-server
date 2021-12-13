FROM ubuntu:18.04

RUN apt-get update -y && apt-get upgrade -y
RUN apt-get install python3 python3-pip git -y

RUN git clone https://github.com/calofmijuck/mnist-server.git
RUN pip3 install --upgrade pip

RUN pip3 install -r mnist-server/requirements.txt

COPY cnn/params.pkl mnist-server/backend/cnn/params.pkl

CMD ["mnist-server/backend/run.sh"]
