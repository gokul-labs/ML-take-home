FROM python:3.10

WORKDIR ml-server/

COPY app/ ./app/
COPY mlmodels/finetuned_model_2_cpu.pt ./mlmodels/finetuned_model_2_cpu.pt
COPY mlmodels/finetuned_filter_model_1_cpu.pt ./mlmodels/finetuned_filter_model_1_cpu.pt
COPY api_server.py .
COPY run.sh .
COPY train.py .
COPY Pipfile .
COPY Pipfile.lock .
COPY requirements.txt .

EXPOSE 8001
#RUN pip install pipenv
#RUN pipenv install --deploy
#RUN pipenv shell
RUN pip install -r requirements.txt
CMD ["sh", "run.sh"]