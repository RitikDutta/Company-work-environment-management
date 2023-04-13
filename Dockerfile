FROM tensorflow/tensorflow:latest


COPY . /app
WORKDIR /app

RUN pip install --no-cache-dir -r requirements.txt

CMD python application.py

