from __future__ import print_function
import argparse
import os
import random
import torch
import torch.backends.cudnn as cudnn
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision.utils as vutils

from metric import distance, knn, wasserstein, mmd, inception_score, mode_score, fid
from metric import ConvNetFeatureSaver
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--real', required=True, help='path to real dataset dir')
    parser.add_argument('--fake', required=True, help='path to fake dataset dir')
    parser.add_argument('--batchSize', type=int, default=16, help='input batch size for Dataloader and Covnet')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
    parser.add_argument('--cuda', action='store_true', help='enables cuda')
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    parser.add_argument('--outf', default='results', help='folder to output scores .npz')
    parser.add_argument('--imageSize', type=int, default=256, help='the height / width of the input image to network')

    opt = parser.parse_args()
    print(opt)

    if opt.manualSeed is None:
        opt.manualSeed = random.randint(1, 10000)
    print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)

    cudnn.benchmark = True

    if not os.path.exists(opt.outf):
        os.mkdir(opt.outf)

    if torch.cuda.is_available() and not opt.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    #########################
    #### Dataset prepare ####
    #########################
    # dataset_real = ImageFolder(root=opt.real,
    #                       transform=transforms.Compose([
    #                           transforms.Resize(opt.imageSize),
    #                           transforms.CenterCrop(opt.imageSize),
    #                           transforms.ToTensor(),
    #                           transforms.Normalize(
    #                               (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    #                       ]))
    # dataloader_real = DataLoader(dataset_real, batch_size=opt.batchSize,
    #                                          shuffle=True, num_workers=int(opt.workers))
    # dataset_fake = ImageFolder(root=opt.fake,
    #                       transform=transforms.Compose([
    #                           transforms.Resize(opt.imageSize),
    #                           transforms.CenterCrop(opt.imageSize),
    #                           transforms.ToTensor(),
    #                           transforms.Normalize(
    #                               (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    #                       ]))
    # dataloader_fake = DataLoader(dataset_real, batch_size=opt.batchSize,
    #                                          shuffle=True, num_workers=int(opt.workers))

    convnet_feature_saver = ConvNetFeatureSaver(model='inception_v3',
                                                batchSize=opt.batchSize, workers=int(opt.workers))
    feature_r = convnet_feature_saver.save(opt.real)
    feature_f = convnet_feature_saver.save(opt.fake)
    # 4 feature spaces and 7 scores + incep + modescore + fid
    score = np.zeros(4 * 7 + 3)
    for i in range(0, 4):
        print('compute score in space: ' + str(i))
        Mxx = distance(feature_r[i], feature_r[i], False)
        Mxy = distance(feature_r[i], feature_f[i], False)
        Myy = distance(feature_f[i], feature_f[i], False)

        score[i * 7] = wasserstein(Mxy, True)
        score[i * 7 + 1] = mmd(Mxx, Mxy, Myy, 1)
        tmp = knn(Mxx, Mxy, Myy, 1, False)
        score[(i * 7 + 2):(i * 7 + 7)] = \
            tmp.acc, tmp.acc_t, tmp.acc_f, tmp.precision, tmp.recall

    score[28] = inception_score(feature_f[3])
    score[29] = mode_score(feature_r[3], feature_f[3])
    score[30] = fid(feature_r[3], feature_f[3])

    # [emd-mmd-knn(knn,real,fake,precision,recall)]*4 - IS - mode_score - FID
    res = ''
    res += 'feature_pixl:<\temd\tmmd\tknn(knn,real,fake,precision,recall)>\n{}\n'.format(score[:7])
    res += 'feature_conv:<\temd\tmmd\tknn(knn,real,fake,precision,recall)>\n{}\n'.format(score[7:14])
    res += 'feature_logit:<\temd\tmmd\tknn(knn,real,fake,precision,recall)>\n{}\n'.format(score[14:21])
    res += 'feature_smax:<\temd\tmmd\tknn(knn,real,fake,precision,recall)>\n{}\n'.format(score[21:28])
    res += 'IS:\t{}\n'.format(score[28])
    res += 'MODE_score:\t{}\n'.format(score[29])
    res += 'FID:\t{}\n'.format(score[30])
    print(res)

    if opt.real[-1] == '\\' or opt.real[-1] == '/':
        opt.real = opt.real[:-1]

    if opt.fake[-1] == '\\' or opt.fake[-1] == '/':
        opt.fake = opt.fake[:-1]

    # save scores to .txt file
    path = os.path.join(opt.outf,
                        'score_{}_{}.txt'.format(str(opt.real).split('/')[-1].split('\\')[-1], str(opt.fake).split('/')[-1].split('\\')[-1]))
    with open(path, 'w', encoding='utf-8') as f:
        f.write(res)

    # save final metric scores to npy
    path = os.path.join(opt.outf,
                        'score_{}_{}.npy'.format(str(opt.real).split('/')[-1].split('\\')[-1], str(opt.fake).split('/')[-1].split('\\')[-1]))
    np.save(path, score)
    print('##### training completed :) #####')
    print('### metric scores output is scored at {} ###'.format(path))
