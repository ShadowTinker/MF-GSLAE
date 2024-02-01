
# Implement of TOIS submission

Our model file is in `recbole_cdr/model/cross_domain_recommender/mfgslae.py`

## Requirements

```
recbole==1.0.1
torch>=1.7.0
python>=3.7.0
```

## Reproduce reported results 
### Epinions

1. Change config in `recbole_cdr/properties/model/MFGSLAE.yaml` like this:
```yaml
dropout_prob: 0.7
tau: 0.5
factor: 4
epsilon: 5
alpha: 0.1
ratio: 0.99
ratio_threshold: 0.5
l1_rate: 1e-06
learning_rate: 0.001
weight_decay: 0.01
latent_dimension: 64
use_user_loader: true
```
2. Run following 
```bash
python run_recbole_cdr.py --model=MFGSLAE --config_files=./config/epinions.yaml --gpu_id=1

```

### Douban
1. Change config in `recbole_cdr/properties/model/MFGSLAE.yaml` like this:
```yaml
dropout_prob: 0.5
tau: 2
factor: 4
epsilon: 0.5
alpha: 0.01
ratio: 0.99
ratio_threshold: 0.5
l1_rate: 1e-06
learning_rate: 0.001
weight_decay: 0.01
latent_dimension: 64
use_user_loader: true
```
2. Run following 
```bash
python run_recbole_cdr.py --model=MFGSLAE --config_files=./config/douban_bmovie.yaml --gpu_id=1

```

### Amazon
1. Change config in `recbole_cdr/properties/model/MFGSLAE.yaml` like this:
```yaml
dropout_prob: 0.9
tau: 1
factor: 8
epsilon: 1
alpha: 0.001
ratio: 0.99
ratio_threshold: 0.5
l1_rate: 1e-06
learning_rate: 0.001
weight_decay: 0.0001
latent_dimension: 64
use_user_loader: true
```
2. Run following 
```bash
python run_recbole_cdr.py --model=MFGSLAE --config_files=./config/ama-elecmov.yaml --gpu_id=1

```


