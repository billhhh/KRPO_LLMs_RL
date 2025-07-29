# Kalman Filter Enhanced Group Relative Policy Optimization for Language Model Reasoning

The code repository for paper "Kalman Filter Enhanced Group Relative Policy Optimization for Language Model Reasoning".

## Environment Setup
### Create conda env

```
conda create --name krpo python=3.12 -y
conda activate krpo
```

### Install dependencies

```
conda install -c conda-forge cudatoolkit cudatoolkit-dev -y
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
```

The base model is `Llama-3.2-1B-Instruct`, please follow the huggingface to fetch the model https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct

## Model training and evaluation
### Model training

For model training, the commandline is:

```commandline
bash run.sh [GPU id]
```

For instance:

```commandline
bash run.sh 0
```

### Model evaluation

For model evaluation, the resume path of the tested model can be specified in the `eval_krpo.sh` file. The evaluation can be performed with:

```commandline
bash eval.sh [GPU id]
```

For example:

```commandline
bash eval.sh 0
```

In both train.py and eval.py, group_advantages_baseline() function is how the baseline model gets group advantages.


### Additional Base models and Data

Performed more experiments with base models Qwen2.5-0.5B-Instruct and Qwen2.5-1.5B-Instruct on Normal level difficulty Arithmetic questions:

![base_model](https://raw.githubusercontent.com/billhhh/KRPO_LLMs_RL/refs/heads/main/logs/W%26B%20Chart%202025_7_29%2012_30_39.png)

Also on additional datasets --- AMC and AIME:

![dataset](https://raw.githubusercontent.com/billhhh/KRPO_LLMs_RL/refs/heads/main/logs/W%26B%20Chart%202025_7_29%2012_33_12.png)


## Acknowledgement

If you got a chance to use our code, you can cite us!

Enjoy!!
