# Training Commands

Reproduction commands for the reported USNet experiments on KITTI Road and
Cityscapes.

## KITTI Road

```bash
python train.py --num_epochs 250 --data <path>/data/KITTI
```

Add `--non_pretrained` for the non-pretrained variant.

## Cityscapes

```bash
python train.py --num_epochs 50 --data <path>/data --dataset Cityscapes
```

Add `--non_pretrained` for the non-pretrained variant.
