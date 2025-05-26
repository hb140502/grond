# Towards Backdoor Stealthiness in Model Parameter Space

## Environment settings

```
conda create --name grond python=3.9
conda activate grond
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=12.1 -c pytorch -c nvidia
conda install --yes -c conda-forge --file requirements.txt 
```

## Train a parameter space backdoor model

```
python train_backdoor.py --pr 0.1
```
`pr` is the poisoning rate of the target class.



## Evaluate using CLP
```
python defense_lipschitzness_pruning.py
```

## Generate UPGD 
```
python generate_upgd.py --model_path ${clean_model_weights}
```
