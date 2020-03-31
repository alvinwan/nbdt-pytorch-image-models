# Sample [Neural-Backed Decision Trees](https://github.com/alvinwan/neural-backed-decision-trees) Integration with [pytorch-image-models](https://github.com/rwightman/pytorch-image-models)

[Project Page](http://nbdt.alvinwan.com) &nbsp;//&nbsp; [Paper](http://nbdt.alvinwan.com/paper/) &nbsp;//&nbsp; [No-code Web Demo](http://nbdt.alvinwan.com/demo/) &nbsp;//&nbsp; [Colab Notebook](https://colab.research.google.com/github/alvinwan/neural-backed-decision-trees/blob/master/examples/load_pretrained_nbdts.ipynb)

Wondering what neural-backed decision trees are? See the [Neural-Backed Decision Trees](https://github.com/alvinwan/neural-backed-decision-trees) repository.

**Table of Contents**

- [Explanation](#explanation)
- [Training and Evaluation](#training-and-evaluation)
- [Results](#results)

## Explanation

The full diff between the original repository `pytorch-image-models` and the integrated version is [here, using Github's compare view](https://github.com/alvinwan/nbdt-pytorch-image-models/compare/nbdt). There are a total of 9 lines added:

1. Generate hierarchy (0 lines): Start by generating an induced hierarchy. We use a hierarchy induced from EfficientNet-B7.

```bash
nbdt-hierarchy --dataset=Imagenet1000 --arch=efficientnet_b7b
```

2. Wrap loss during training (3 lines): In `train.py`, we add the custom loss function. This is a wrapper around the existing loss functions.

```python
from nbdt.loss import SoftTreeSupLoss
train_loss_fn = SoftTreeSupLoss(criterion=train_loss_fn, dataset='Imagenet1000', tree_supervision_weight=10, hierarchy='induced-efficientnet_b7b')
validate_loss_fn = SoftTreeSupLoss(criterion=validate_loss_fn, dataset='Imagenet1000', tree_supervision_weight=10, hierarchy='induced-efficientnet_b7b')
```

3. Wrap model during inference (6 lines): In `validate.py`, we add NBDT inference. This is a wrapper around the existing model. We actually spend 4 lines adding and processing a custom `--nbdt` argument, so the actual logic for adding NBDT inference is only 2 lines.

```python
parser.add_argument('--nbdt', choices=('none', 'soft', 'hard'), default='none',
                    help='Type of NBDT inference to run')
...
from nbdt.model import SoftNBDT, HardNBDT
if args.nbdt != 'none':
    cls = SoftNBDT if args.nbdt == 'soft' else HardNBDT
    model = cls(model=model, dataset='Imagenet1000', hierarchy='induced-efficientnet_b7b')
```

## Training and Evaluation

To reproduce our results, **make sure to checkout the `nbdt` branch**.

```bash
# 1. git clone the repository
git clone git@github.com:alvinwan/nbdt-pytorch-image-models.git  # or http addr if you don't have private-public github key setup
cd nbdt-pytorch-image-models

# 2. install requirements
pip install -r requirements.txt

# 3. checkout branch with nbdt integration
git checkout nbdt
```

**Training**: For our ImageNet results, we use the hyperparameter settings reported for ImageNet-EdgeTPU-Small found in the original README: [EfficientNet-ES (EdgeTPU-Small)](https://github.com/rwightman/pytorch-image-models#efficientnet-es-edgetpu-small-with-randaugment---78066-top-1-93926-top-5). Note the accuracy reported at this link is the average of 8 checkpoints. However, we use only 1 checkpoint, so we compare against the best single-checkpoint 77.23% result for EfficientNet-ES reported in the official [EfficientNet-EdgeTPU](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet/edgetpu) repository. The hyperparameter settings reported in the first link are reproduced below:

```bash
./distributed_train.sh 8 /data/imagenetwhole/ilsvrc2012/ --model efficientnet_es -b 128 --sched step --epochs 450 --decay-epochs 2.4 --decay-rate .97 --opt rmsproptf --opt-eps .001 -j 8 --warmup-lr 1e-6 --weight-decay 1e-5 --drop 0.2 --drop-connect 0.2 --aa rand-m9-mstd0.5 --remode pixel --reprob 0.2 --amp --lr .064
```

**Validation**: To run inference, we use the following command. The majority of this command is typical for this repository. We simply add an extra `--nbdt` flag at the end for the type of NBDT we wish to run.

```bash
python validate.py /data/imagenetwhole/ilsvrc2012/val/ --model efficientnet_es --checkpoint=./output/train/20200319-185245-efficientnet_es-224/model_best.pth.tar --nbdt=soft
```

## Results

NofE, shown below, was the strongest competing decision-tree-based method. Note that our NBDT-S outperforms NofE by ~14%. The acccuracy of the original neural network, EfficientNet-ES, is also shown. Our decision tree's accuracy is within 2% of the original neural network's accuracy.

|                | NBDT-S (Ours) | NBDT-H (Ours) | NofE   | EfficientNet-ES |
|----------------|---------------|---------------|--------|-----------------|
| ImageNet Top-1 | 75.30%        | 74.79%        | 61.29% | 77.23%          |

See the original Neural-Backed Decision Trees [results](https://github.com/alvinwan/neural-backed-decision-trees#results) for a full list of all baselines. You can download our pretrained model and all associated logs at [v1.0](https://github.com/alvinwan/nbdt-pytorch-image-models/releases/tag/1.0).

**For more information, return to the original [Neural-Backed Decision Trees](https://github.com/alvinwan/neural-backed-decision-trees) repository.**
