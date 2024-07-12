FROM python:3.10-slim

WORKDIR /app

# copy files of saved_models api_server and src;
COPY . ./app

# will install the torch-cpu dependencies manually so remove
# their entries from the requirements file;
RUN sed -i -E 's/.*torch.*|.*lightning.*//' ./app/requirements.txt && \
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install -r ./app/requirements.txt

ENV PYTHONPATH ./app

# Make port 5000 available to the world outside this container
EXPOSE 5000

CMD ["python", "./app/api_server/app.py"]