FROM python:3.10

WORKDIR ml-server/

COPY app/ .
COPY mlmodels/finetuned_model_2_cpu.pt .
COPY mlmodels/finetuned_filter_model_1_cpu.pt .
COPY api_server.py .
COPY Pipfile .
COPY Pipfile.lock .
COPY run.sh .
COPY train.py .

EXPOSE 8001
RUN sudo pip install pipenv
RUN sudo pipenv install --system
RUN sudo pipenv shell
CMD ["sh", "run.sh"]