# Finetuning BERT using LoRA.

## Training the model:
* Frist, cd in the root of the repo.
* Second, run `export PYTHONPATH=.`
* Third, run `wandb login` to login to the wandb with your api key.
* Fourth, run `export WANDB_START_METHOD="thread"` otherwise some weird threading exception occurs. For more info see this <a href="https://github.com/wandb/wandb/issues/3223#issuecomment-1032820724">issue</a>.
* Fifth, set tokenizers env variable to avoid warnings 

    ```bash
    export TOKENIZERS_PARALLELISM=true
    ```

## Local testing with Docker:
* Build the image from the root of the dir:
    ```bash
    docker build -t flask-pytorch-model -f api_server/Dockerfile .
    ```
* Run the container, should start the server:
    ```bash
    docker run -p 5000:5000 --name flask-cont-1 flask-pytorch-model
    ```
* From another terminal, test the app:
    This should be positive sentiment

    ```bash
    curl -X POST http://0.0.0.0:5000/predict -H "Content-Type: application/json" -d '{"text": "The movie was wonderful!"}'
    ```

    This should be negative sentiment

    ```bash
    curl -X POST http://0.0.0.0:5000/predict -H "Content-Type: application/json" -d '{"text": "The movie was awful!"}'
    ```
* I got the following responses respectively:
    First result

    ```bash
    {"body":"{\"sentiment\": [\"positive\"]}","headers":{"Content-Type":"application/json"},"statusCode":200}
    ```

    Second result:
    
    ```bash
    {"body":"{\"sentiment\": [\"negative\"]}","headers":{"Content-Type":"application/json"},"statusCode":200}
    ```