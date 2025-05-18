# Kalman Filter Enhanced Group Relative Policy Optimization for Language Model Reasoning

The code repository for paper "Kalman Filter Enhanced Group Relative Policy Optimization for Language Model Reasoning"

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

## Acknowledgement

If you got a chance to use our code, you can cite us!

Enjoy!!
