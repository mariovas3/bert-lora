# Finetuning BERT using LoRA.

Parameters from Lightning training when lora rank is 16 and I use lora weights on query, value and feedforward matrices of transformer blocks.

```bash
  | Name             | Type       | Params | Mode
--------------------------------------------------------
0 | sbert            | BertModel  | 23.2 M | eval
1 | lora_module_list | ModuleList | 516 K  | train
2 | mlp              | MLP        | 58.1 K | train
--------------------------------------------------------
574 K     Trainable params
22.7 M    Non-trainable params
23.3 M    Total params
```

The trainable parameters are about `2.46%` of total parameters.

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
    python src/run_experiment.py fit --config fit_config.yaml --trainer.accelerator=gpu --trainer.devices=1 --trainer.max_epochs=12 --trainer.check_val_every_n_epoch=2 --trainer.log_every_n_step=25 --data.num_workers=4 --my_model_checkpoint.every_n_epochs=2 --model.lora_alpha=1 --model.lora_rank=16 --model.lr=1e-3
    ```

* After checking various configurations for the rank (8, 16, 32) and alpha parameter of LoRA (0.25, 1, 8), I found that the algorithm is not very sensitive to the rank but is very sensitive to alpha. The configuration with `lora_alpha=1` and `lora_rank=16` seemed to work well - achieving `89.58%` accuracy.

* Each experiment was run for 12 epochs taking 28 minutes per run on a L4 GPU from a Lightning Studio.

* The training curves from wandb are given below:

	<img src="./assets/imgs/train-curves.png"/>

* I also got a good GPU utilisation of ~ 96%:

    <img src="./assets/imgs/gpu-utilisation.png"/>

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
