# kitti
python train.py --data data --num_epochs 251 --batch_size 32 --num_workers 32
python test.py --checkpoint_path <path_to_training>/models/usnet_best.pth --data data/KITTI

# cityscapes
python train.py --data data --dataset Cityscapes --num_epochs 251 --batch_size 32 --num_workers 32

# xlstm
python train.py --data data --dataset Cityscapes --num_epochs 251 --batch_size 64 --num_workers 32 --backbone_name xlstm
python test.py --checkpoint_path runs/xlstm___lr_0.0010___bs_64___epochs_1001___2025-04-01T17\:36\:37/models/usnet_best.pth --dataset Cityscape --backbone_name xlstm
