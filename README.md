# Instance Adaptive Adversarial Training

Code accompanying out paper: "Instance adaptive adversarial training: Improved accuracy tradeoffs in neural nets"

## Requirements

* Pytorch
* Foolbox
* Adversarial robustness toolbox

## Training commands

Please update config json files to modily parameters/ specify data paths.

Training a clean model (ERM) - 

```
python main.py --cfg-path configs/train.json --data_root [specify path] --alg clean 
```

Training adversarial training - 

```
python main.py --cfg-path configs/train.json --data_root [specify path] --alg adv_training 
```

Training instance adaptive algorithm (our approach)

```
python main.py --cfg-path configs/train_adaptive.json --data_root [specify path]
```

To evaluate a model, run

```
python main.py --cfg-path configs/eval.json --data_root [specify path] --save_path [path to log results] --attack_steps [number of attack steps] --restore [path to restore saved model from]
```
