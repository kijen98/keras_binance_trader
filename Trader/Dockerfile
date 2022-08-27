FROM tensorflow/tensorflow

RUN apt-get update -y
RUN apt-get upgrade -y

RUN pip install --upgrade pip
RUN pip install schedule
RUN pip install python-dateutil

RUN pip install matplotlib
RUN pip install pandas
RUN pip install python-binance

WORKDIR /app

ENTRYPOINT ["/bin/bash", "-c", "python Trader.py"]