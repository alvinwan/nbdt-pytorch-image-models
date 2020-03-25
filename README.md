# Sample NBDT Integration with [`pytorch-image-models`](https://github.com/rwightman/pytorch-image-models)

## Explanation

The full diff between the original repository `pytorch-image-models` and the integrated version is [here, using Github's compare view](https://github.com/alvinwan/nbdt-pytorch-image-models/compare/nbdt). Ignoring changes to this README, there are a total of 9 lines added:

1. (3 lines) In `train.py`, we add the custom loss function. This is a wrapper around the existing loss functions.

```
from nbdt.loss import SoftTreeSupLoss
train_loss_fn = SoftTreeSupLoss(criterion=train_loss_fn, dataset='Imagenet1000', tree_supervision_weight=10, hierarchy='induced-efficientnet_b7b')
validate_loss_fn = SoftTreeSupLoss(criterion=validate_loss_fn, dataset='Imagenet1000', tree_supervision_weight=10, hierarchy='induced-efficientnet_b7b')
```

2. (6 lines) In `validate.py`, we add NBDT inference. This is a wrapper around the existing model. We actually spend 4 lines adding and processing a custom `--nbdt` argument, so the actual logic for adding NBDT inference is only 2 lines.

```
parser.add_argument('--nbdt', choices=('none', 'soft', 'hard'), default='none',
                    help='Type of NBDT inference to run')
...
from nbdt.model import SoftNBDT, HardNBDT
if args.nbdt != 'none':
    cls = SoftNBDT if args.nbdt == 'soft' else HardNBDT
    model = cls(model=model, dataset='Imagenet1000', hierarchy='induced-efficientnet_b7b')
```

## Training and Evaluation

To reproduce our results, **make sure to checkout the `nbdt/` branch**.

```
# 1. git clone the repository

# 2. install requirements
cd nbdt-pytorch-image-models && pip install -r requirements.txt

# 3. checkout `nbdt`
git checkout nbdt
```

**Training**: For our ImageNet results, we use the hyperparameter settings reported for ImageNet-EdgeTPU-Small found in the original README: [EfficientNet-ES (EdgeTPU-Small) with RandAugment - 78.066 top-1, 93.926 top-5](https://github.com/rwightman/pytorch-image-models#efficientnet-es-edgetpu-small-with-randaugment---78066-top-1-93926-top-5). Note the reported 78.066% is the average of 8 checkpoints, whereas we only use one checkpoint to match the original paper. These settings are reproduced below:

```
./distributed_train.sh 8 /data/imagenetwhole/ilsvrc2012/ --model efficientnet_es -b 128 --sched step --epochs 450 --decay-epochs 2.4 --decay-rate .97 --opt rmsproptf --opt-eps .001 -j 8 --warmup-lr 1e-6 --weight-decay 1e-5 --drop 0.2 --drop-connect 0.2 --aa rand-m9-mstd0.5 --remode pixel --reprob 0.2 --amp --lr .064
```

**Validation**: To run inference, we use the following command. The majority of this command is typical for this repository. We simply add an extra `--nbdt` flag at the end for the type of NBDT we wish to run.

```
python validate.py /data/imagenetwhole/ilsvrc2012/val/ --model efficientnet_es --checkpoint=./output/train/20200319-185245-efficientnet_es-224/model_best.pth.tar --nbdt=soft
```

For more information, return to the original [Neural-Backed Decision Trees](https://github.com/alvinwan/neural-backed-decision-trees) repository.
