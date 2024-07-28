FROM python:3.10-slim

WORKDIR /app

# copy files of saved_models api_server and src;
COPY . ./app

# remove some dev dependencies.
RUN sed -i -E 's/.*matplot.*|.*wandb.*|.*lightning.*//' ./app/requirements.txt && \
    pip install -r ./app/requirements.txt

ENV PYTHONPATH ./app

# Make port 5000 available to the world outside this container
EXPOSE 5000

CMD ["python", "./app/api_server/app.py"]