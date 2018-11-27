import os
import torch
import scipy.io
import argparse
import pandas as pd
import numpy as np


def compute_mAP(index, good_index, junk_index):
    ap = 0
    cmc = torch.IntTensor(len(index)).zero_()
    if good_index.size == 0:
        cmc[0] = -1
        return ap, cmc

    # remove junk_index
    mask = np.in1d(index, junk_index, invert=True)
    index = index[mask]

    # find good_index index
    ngood = len(good_index)
    mask = np.in1d(index, good_index)
    rows_good = np.argwhere(mask == True)
    rows_good = rows_good.flatten()

    cmc[rows_good[0]:] = 1
    for i in range(ngood):
        d_recall = 1.0 / ngood
        precision = (i + 1) * 1.0 / (rows_good[i] + 1)
        if rows_good[i] != 0:
            old_precision = i * 1.0 / rows_good[i]
        else:
            old_precision = 1.0
        ap = ap + d_recall * (old_precision + precision) / 2

    return ap, cmc[:20]


def evaluate(score, ql, qc, gl, gc):
    # predict index sort from small to large
    index = np.argsort(score)[::-1]

    # good index
    query_index = np.argwhere(gl == ql)
    camera_index = np.argwhere(gc == qc)

    good_index = np.setdiff1d(query_index, camera_index, assume_unique=True)

    junk_index1 = np.argwhere(gl == -1)
    junk_index2 = np.intersect1d(query_index, camera_index)
    junk_index = np.append(junk_index2, junk_index1)

    return compute_mAP(index, good_index, junk_index)


def main():
    # settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--features', type=str, default='./log_dir')
    parser.add_argument('--labels', type=str, default='./log_dir')
    parser.add_argument('--gpu', type=int, default=1)
    args = parser.parse_args()

    #print(torch.cuda.device_count())
    #print(torch.cuda.device(1))
    torch.cuda.set_device(args.gpu)
    logPath = args.features
    labelPath = args.labels

    logFile = {subset: scipy.io.loadmat(os.path.join(logPath, 'feature_val_%s.mat' % subset))
               for subset in ['query', 'gallery']}

    labelDict = {subset: pd.read_csv(os.path.join(labelPath, subset + 'Info.txt'), header=None,
                                     delimiter='\t').set_index(0)[1].to_dict() for subset in ['query', 'gallery']}

    names = {subset: logFile[subset]['names'] for subset in ['query', 'gallery']}
    labels = {subset: np.array([labelDict[subset][name] for name in names[subset]]) for subset in ['query', 'gallery']}
    features = {subset: torch.FloatTensor(logFile[subset]['features']).cuda() for subset in ['query', 'gallery']}

    CMC = torch.IntTensor(20).zero_()
    ap = 0.0

    for i in range(len(labels['query'])):
        score = torch.mm(features['gallery'], features['query'][i].view(-1, 1))
        score = score.squeeze(1).cpu().numpy()

        # validation set and test set no need consider qc==gc cases
        ap_tmp, CMC_tmp = evaluate(score, labels['query'][i], [], labels['gallery'], [])
        if CMC_tmp[0] == -1:
            continue

        CMC += CMC_tmp
        ap += ap_tmp

    CMC = CMC.float()
    CMC /= len(labels['query'])
    ap /= len(labels['query'])

    print('top1: %.4f, top5: %.4f, top10: %.4f, mAP: %.4f' % (CMC[0], CMC[4], CMC[9], ap))


if __name__ == '__main__':
    main()
