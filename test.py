import argparse
import cv2
import numpy as np
import os
import torch

from dataset.cityscapes import Cityscapes_Dataset
from dataset.kitti import Kitti_Dataset
from model.usnet import USNet
from torch.utils.data import DataLoader
from utils import fast_hist, getScores


def test(args, model, dataloader):
    # {{{

    print('start test!')
    with torch.no_grad():
        model.eval()
        hist = np.zeros((args.num_classes, args.num_classes))
        for i, sample in enumerate(dataloader):
            image, depth, label = sample['image'], sample['depth'], sample['label']
            oriHeight, oriWidth = sample['oriHeight'], sample['oriWidth']
            oriWidth = oriWidth.cpu().numpy()[0]
            oriHeight = oriHeight.cpu().numpy()[0]

            if torch.cuda.is_available() and args.use_gpu:
                image = image.cuda()
                depth = depth.cuda()
                label = label.cuda()

            # get predict image
            evidence, evidence_a, alpha, alpha_a = model(image, depth)

            s = torch.sum(alpha_a, dim=1, keepdim=True)
            p = alpha_a / (s.expand(alpha_a.shape))

            pred = p[:,1]
            pred = pred.view(args.crop_height, args.crop_width)
            pred = pred.detach().cpu().numpy()
            pred = np.array(pred)

            # save predict image
            visualize = cv2.resize(pred, (oriWidth, oriHeight))
            visualize = np.floor(255*(visualize - visualize.min()) / (visualize.max()-visualize.min()))
            img_path = sample['img_path'][0]
            img_name = img_path.split('/')[-1]
            save_name = img_name.split('_')[0]+'_road_'+img_name.split('_', 1)[1]
            cv2.imwrite(os.path.join(args.save_path, save_name), np.uint8(visualize))

            pred = np.uint8(pred > 0.5)
            pred = cv2.resize(pred, (oriWidth, oriHeight), interpolation=cv2.INTER_NEAREST)

            # get label image
            label = label.squeeze()
            label = label.cpu().numpy()
            label = np.array(label)
            label = cv2.resize(np.uint8(label), (oriWidth, oriHeight), interpolation=cv2.INTER_NEAREST)

            hist += fast_hist(label.flatten(), pred.flatten(), args.num_classes)
        F_score, pre, recall, fpr, fnr = getScores(hist)
        test_result  = f"Model {str(args.checkpoint_path)} tested against validation set in {str(args.data)}\n"
        test_result += 'F_\{1\} %.3f\n' % F_score
        test_result += 'P       %.3f\n' % pre
        test_result += 'R       %.3f\n' % recall
        test_result += 'FPR     %.3f\n' % fpr
        test_result += 'FNR     %.3f\n' % fpr
        file = open('test_result.txt', mode='a+')
        file.write(test_result)
        file.close()
        print(test_result)
    return F_score, pre, recall, fpr, fnr

    # }}}


def main():
    # {{{

    # basic parameters
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--checkpoint_path',
        type=str,
        default= 'models/usnet_best.pth',
        help='The path to the pretrained weights of model'
    )
    parser.add_argument(
        '--crop_height',
        type=int,
        default=384,
        help='Height of cropped/resized input image to network'
    )
    parser.add_argument(
        '--crop_width',
        type=int,
        default=1248,
        help='Width of cropped/resized input image to network'
    )
    parser.add_argument(
        '--data',
        type=str,
        default='./data/KITTI',
        help='Path of testing data'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default="Kitti",
        help='Dataset you are using.'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=1,
        help='Number of images in each batch'
    )
    parser.add_argument(
        '--backbone_name',
        type=str,
        default="resnet18",
        help='The backbone model you are using.'
    )
    parser.add_argument(
        '--cuda',
        type=str,
        default='0',
        help='GPU ids used for training'
    )
    parser.add_argument(
        '--use_gpu',
        type=bool,
        default=True,
        help='Whether to user gpu for training'
    )
    parser.add_argument(
        '--num_classes',
        type=int,
        default=2,
        help='num of object classes (with void)'
    )
    parser.add_argument(
        '--save_path',
        type=str,
        default= 'test',
        help='Path to save predict image'
    )
    # args = parser.parse_args(params)
    args = parser.parse_args()

    if args.dataset == 'Kitti':
        dataset = Kitti_Dataset(args, root=args.data, split='validating')
    elif args.dataset == 'Cityscape':
        dataset = Cityscapes_Dataset(args, root=args.data, split='val')
    else:
        print(f"unknown dataset {str(args.dataset)}, exiting...")
        exit(1)

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=1,
    )

    # build model
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
    model = USNet(args.num_classes, args.backbone_name, True)
    if torch.cuda.is_available() and args.use_gpu:
        model = torch.nn.DataParallel(model).cuda()
    else:
        print(f"no gpu found, exiting…")
        exit(1)

    # load trained model
    print('load model from %s ...' % args.checkpoint_path)
    model.module.load_state_dict(torch.load(args.checkpoint_path))
    print('Done!')

    # make save folder
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    test(args, model, dataloader)

    # }}}


if __name__ == '__main__':
    main()
