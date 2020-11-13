### Training the Unsupervised Representation Learning Model

The procedure follows that in [https://github.com/facebookresearch/moco](https://github.com/facebookresearch/moco) with some modifications for CIFAR-10 (see default hyperparameters in the following commands)

Training CPC:
```
python main_moco_cifar.py -a resnet50 --lr 0.3 --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --moco-t 0.07 --world-size 1 --rank 0 --mlp -j 4 --loss cpc --epochs 200 --batch-size 256 --moco-dim 2048 --aug-plus --cos --save-dir [ save location ] --data [ cifar10 | cifar100 ]
```

Training ML-CPC with curriculum:
```
python main_moco_cifar.py -a resnet50 --lr 0.3 --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --moco-t 0.07 --world-size 1 --rank 0 --mlp -j 4 --loss ml_cpc --ml-cpc-alpha 10.0 --ml-cpc-alpha-low 0.1 --ml-cpc-alpha-geo --epochs 200 --batch-size 256 --moco-dim 2048 --aug-plus --cos --save-dir . --data [ cifar10 | cifar100 ]
```


Training linear classifier:
```
python main_lincls_cifar.py -a resnet50 --lr 3 --dist-url 'tcp://localhost:10002' --multiprocessing-distributed --world-size 1 --rank 0 -j 4 --pretrained=[ checkpoint location ] --save-dir [ save location ] --data [ cifar10 | cifar100 ]
```