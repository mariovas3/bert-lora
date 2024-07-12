# Finetuning BERT using LoRA.

Parameters from Lightning training when lora rank is 8 and I use lora weights on query, value and feedforward matrices of transformer blocks.

```bash
  | Name             | Type       | Params | Mode 
--------------------------------------------------------
0 | sbert            | BertModel  | 23.0 M | eval 
1 | lora_module_list | ModuleList | 258 K  | train
2 | mlp              | MLP        | 58.1 K | train
--------------------------------------------------------
316 K     Trainable params
22.7 M    Non-trainable params
23.0 M    Total params
```

The trainable parameters are about `1.37%` of total parameters.

## Training the model:
* Frist, cd in the root of the repo.
* Second, run `export PYTHONPATH=.`
* Third, run `wandb login` to login to the wandb with your api key.
* Fourth, run `export WANDB_START_METHOD="thread"` otherwise some weird threading exception occurs. For more info see this <a href="https://github.com/wandb/wandb/issues/3223#issuecomment-1032820724">issue</a>.
* Fifth, set tokenizers env variable to avoid warnings 

    ```bash
    export TOKENIZERS_PARALLELISM=true
    ```
* Training command:

    ```bash
    python src/run_experiment.py fit --config fit_config.yaml --trainer.accelerator=gpu --trainer.devices=1 --trainer.max_epochs=12 --trainer.check_val_every_n_epoch=2 --trainer.log_every_n_step=25 --data.num_workers=4 --my_model_checkpoint.every_n_epochs=2 --model.lora_alpha=1
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