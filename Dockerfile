FROM python:3.10.12
COPY . /app
WORKDIR /app 
RUN pip install -r requirements.txt
CMD python template_main.py