import argparse
import pickle
import os

import IPython
import numpy as np
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',
                        required=True,
                        choices={'ntu/xsub', 'ntu/xview', 'ntu120/xsub', 'ntu120/xset', 'NW-UCLA'},
                        help='the work folder for storing results')
    parser.add_argument('--alpha',
                        default=None,
                        help='weighted summation',
                        type=float)
    parser.add_argument('--joint-dir1', help='Directory containing "epoch1_test_score.pkl" for Koopman joint eval results')
    parser.add_argument('--bone-dir1', help='Directory containing "epoch1_test_score.pkl" for Koopman bone eval results')
    parser.add_argument('--joint-motion-dir1', help='Directory containing "epoch1_test_score.pkl" for Koopman joint motion eval results')
    parser.add_argument('--bone-motion-dir1', help='Directory containing "epoch1_test_score.pkl" for Koopman bone motion eval results')

    parser.add_argument('--joint-dir2', help='Directory containing "epoch1_test_score.pkl" for original joint eval results')
    parser.add_argument('--bone-dir2', help='Directory containing "epoch1_test_score.pkl" for original bone eval results')
    parser.add_argument('--joint-motion-dir2', help='Directory containing "epoch1_test_score.pkl" for original joint motion eval results')
    parser.add_argument('--bone-motion-dir2', help='Directory containing "epoch1_test_score.pkl" for original bone motion eval results')

    arg = parser.parse_args()

    dataset = arg.dataset
    if 'UCLA' in arg.dataset:
        label = []
        with open('./data/' + 'NW-UCLA/' + '/val_label.pkl', 'rb') as f:
            data_info = pickle.load(f)
            for index in range(len(data_info)):
                info = data_info[index]
                label.append(int(info['label']) - 1)
    elif 'ntu120' in arg.dataset:
        if 'xsub' in arg.dataset:
            npz_data = np.load('./data/' + 'ntu120/' + 'NTU120_CSub.npz')
            label = np.where(npz_data['y_test'] > 0)[1]
        elif 'xset' in arg.dataset:
            npz_data = np.load('./data/' + 'ntu120/' + 'NTU120_CSet.npz')
            label = np.where(npz_data['y_test'] > 0)[1]
    elif 'ntu' in arg.dataset:
        if 'xsub' in arg.dataset:
            npz_data = np.load('./data/' + 'ntu/' + 'NTU60_CS.npz')
            label = np.where(npz_data['y_test'] > 0)[1]
        elif 'xview' in arg.dataset:
            npz_data = np.load('./data/' + 'ntu/' + 'NTU60_CV.npz')
            label = np.where(npz_data['y_test'] > 0)[1]
    else:
        raise NotImplementedError

    with open(os.path.join(arg.joint_dir1, 'epoch1_test_score.pkl'), 'rb') as r1:
        r1 = list(pickle.load(r1).items())

    with open(os.path.join(arg.bone_dir1, 'epoch1_test_score.pkl'), 'rb') as r2:
        r2 = list(pickle.load(r2).items())

    if arg.joint_motion_dir1 is not None:
        with open(os.path.join(arg.joint_motion_dir1, 'epoch1_test_score.pkl'), 'rb') as r3:
            r3 = list(pickle.load(r3).items())
    if arg.bone_motion_dir1 is not None:
        with open(os.path.join(arg.bone_motion_dir1, 'epoch1_test_score.pkl'), 'rb') as r4:
            r4 = list(pickle.load(r4).items())

    if arg.joint_dir2 is not None:
        with open(os.path.join(arg.joint_dir2, 'epoch1_test_score.pkl'), 'rb') as r5:
            r5 = list(pickle.load(r5).items())
    if arg.bone_dir2 is not None:
        with open(os.path.join(arg.bone_dir2, 'epoch1_test_score.pkl'), 'rb') as r6:
            r6 = list(pickle.load(r6).items())
    if arg.joint_motion_dir2 is not None:
        with open(os.path.join(arg.joint_motion_dir2, 'epoch1_test_score.pkl'), 'rb') as r7:
            r7 = list(pickle.load(r7).items())
    if arg.bone_motion_dir2 is not None:
        with open(os.path.join(arg.bone_motion_dir2, 'epoch1_test_score.pkl'), 'rb') as r8:
            r8 = list(pickle.load(r8).items())

    score_1 = [r1[i][1] for i in range(len(r1))]
    score_2 = [r2[i][1] for i in range(len(r2))]
    score_3 = [r3[i][1] for i in range(len(r3))]
    score_4 = [r4[i][1] for i in range(len(r4))]
    score_5 = [r5[i][1] for i in range(len(r5))]
    score_6 = [r6[i][1] for i in range(len(r6))]
    score_7 = [r7[i][1] for i in range(len(r7))]
    score_8 = [r8[i][1] for i in range(len(r8))]

    score_1 = np.array(score_1)
    score_2 = np.array(score_2)
    score_3 = np.array(score_3)
    score_4 = np.array(score_4)
    score_5 = np.array(score_5)
    score_6 = np.array(score_6)
    score_7 = np.array(score_7)
    score_8 = np.array(score_8)


    arg.alpha = [0.3, 0.4, 0.1, 0.1, 0.4, 0.5, 0.2, 0.0]     # ntu120 cross-subject
    score = score_1 * arg.alpha[0] + score_2 * arg.alpha[1] + score_3 * arg.alpha[2] + score_4 * arg.alpha[3] + \
                    score_5 * arg.alpha[4] + score_6 * arg.alpha[5] + score_7 * arg.alpha[6] + score_8 * arg.alpha[7]

    pred = np.argmax(score, axis=-1)
    cmp = (pred == label)
    acc = (np.sum(cmp != 0) / len(cmp))

    print('Top1 Acc: {:.4f}%'.format(acc * 100))

