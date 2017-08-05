# -*- coding: utf-8 -*-
from __future__ import print_function

import os, sys, h5py, gc, argparse, codecs, shutil
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

from torch_model import *

train = pd.read_csv('../../input/data_train_image.txt', header = None, sep = ' ', names = ['img', 'label', 'url'])
val = pd.read_csv('../../input/val.txt', header = None, sep = ' ', names = ['img', 'label', 'url'])

# train['img'] = '../input/train/' + train['img'] + '.jpg'
# val['img'] = '../input/test1/' + val['img'] + '.jpg'

# 删除标签不一致的情况
train = train[~train['img'].duplicated(keep = False)]
train_val = pd.concat([train, val], axis = 0, ignore_index = True)

lbl = LabelEncoder()
train_val['label'] = lbl.fit_transform(train_val['label'].values)

# train_val = train_val.iloc[: 1000]
test = os.listdir('../../input/image/')
test = [x[: -4] for x in test]


train_feat, test_feat = [], []
feature_file = [
    # './feature/googlenet_pet_breed.h5',
    
    # './feature_yolo/resnet18.h5',
    # './feature_yolo/densenet161.h5',
    # './feature_yolo/densenet169.h5',
    # './feature_yolo/densenet201.h5',
    # './feature_yolo/densenet121.h5',
    # './feature_yolo/densenet201.h5',
    
    'dpn92.h5'
    
    # './feature/vgg11.h5',
    # './feature/vgg13.h5',
    # './feature/vgg16.h5',
    # './feature/vgg19.h5',
    
    # './feature/resnet18.h5',
    # './feature/resnet34.h5',
    # './feature/resnet50.h5',
    # './feature/resnet101.h5',
    # './feature/resnet152.h5',

    # './feature/densenet121.h5',
    # './feature/densenet161.h5',
    # './feature/densenet169.h5',
    # './feature/densenet201.h5',

    # './feature/inception.h5',
]
for ffile in feature_file:
    with h5py.File(ffile, "r") as f:
        train_feat.append(f['train_feature'][:])
        test_feat.append(f['test_feature'][:])

train_feat = np.concatenate(train_feat, 1)
test_feat = np.concatenate(test_feat, 1)
print('Feature:', train_feat.shape)
print(feature_file)

class modelnn(nn.Module):
    def __init__(self):
        super(modelnn, self).__init__()
        self.model = nn.Sequential(
            # nn.Dropout(0.05),
            nn.Linear(train_feat.shape[1], 100),
            # nn.ReLU(True),
            
            # nn.Dropout(0.2),
            # nn.Linear(256, 100),
        )

    def forward(self, x):
        x = x.view(x.size(0), train_feat.shape[1])
        out = self.model(x)
        return out

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = 0.05 * (0.95 ** (epoch // 5))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


skf = StratifiedKFold(n_splits = 6)
train_preds, test_preds = np.zeros(train_feat.shape[0]), []
train_logs = [[], [], [], []]
for train_index, test_index in skf.split(train_feat, train_val['label']):
    X_train, X_test = train_feat[train_index, :], train_feat[test_index, :]
    y_train, y_test = train_val['label'].values[train_index], train_val['label'].values[test_index]

    # from imblearn.over_sampling import SMOTE
    # sm = SMOTE()
    # X_train, y_train = sm.fit_sample(X_train, y_train)

    train_set = Arrayloader(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size = 128, shuffle=True, num_workers=4)

    val_set = Arrayloader(X_test, y_test)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size = 128, shuffle=True, num_workers=4)

    model = modelnn()
    model = model.cuda()

    class_weight = np.array([ 1.65876777,  1.4       ,  1.45228216,  1.47058824,  1.4  ,
        1.89189189,  1.60550459,  4.72972973,  1.75879397,  4.86111111,
        4.16666667,  7.        ,  1.40562249,  6.48148148,  3.5       ,
        6.03448276,  1.75      ,  1.75      ,  1.40562249,  4.48717949,
        1.41129032,  1.96629213,  1.77664975,  1.4       ,  1.69082126,
        1.4       ,  5.2238806 ,  1.4       ,  1.41129032,  1.4       ,
        2.09580838,  3.27102804,  1.42276423,  1.4       ,  3.72340426,
        1.40562249,  1.4       ,  1.40562249,  1.4       ,  3.39805825,
        5.14705882,  1.44032922,  3.36538462,  5.73770492,  2.77777778,
        2.09580838,  1.40562249,  1.41129032,  1.4       ,  1.40562249,
        1.40562249,  2.51798561,  1.4       ,  1.97740113,  1.41700405,
        1.54867257,  1.79487179,  1.66666667,  2.71317829,  1.89189189,
        1.40562249,  1.41129032,  1.4       ,  1.41129032,  1.40562249,
        3.24074074,  1.60550459,  2.13414634,  2.65151515,  3.80434783,
        2.1875    ,  2.04678363,  1.40562249,  2.86885246,  3.72340426,
        1.66666667,  1.40562249,  2.71317829,  4.32098765,  3.64583333,
        3.18181818,  5.73770492,  1.63551402,  1.2195122 ,  6.60377358,
        1.52173913,  5.07246377,  1.40562249,  1.88172043,  1.41700405,
        1.40562249,  1.40562249,  1.4       ,  1.4       ,  1.54867257,
        1.        ,  2.69230769,  1.2962963 ,  1.4       ,  6.25      ], dtype = np.float32)
    class_weight = torch.from_numpy(class_weight)
    # criterion = nn.CrossEntropyLoss(weight = class_weight.float()).cuda()
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer_ft = torch.optim.SGD(model.parameters(), lr = 0.0005, momentum = 0.75, weight_decay = 1e-4)

    epochsize = 80
    for epoch in range(epochsize):
        adjust_learning_rate(optimizer_ft, epoch)
        # Traing batch
        running_corrects = 0.0
        running_loss = 0.0
        for data in train_loader:
            dta_x, dta_y = data
            dta_x, dta_y = Variable(dta_x.cuda()), Variable(dta_y.cuda().view(dta_y.size(0)))
            optimizer_ft.zero_grad()
            outputs = model(dta_x)
            _, preds = torch.max(outputs.data, 1)
            loss = criterion(outputs, dta_y)
            loss.backward()
            optimizer_ft.step()

            running_loss += loss.data[0]
            running_corrects += torch.sum(preds == dta_y.data)

        train_loss = running_loss / len(train_set)
        train_acc = running_corrects / len(train_set)

        # Val batch
        running_corrects = 0.0
        running_loss = 0.0
        for data in val_loader:
            dta_x, dta_y = data
            dta_x, dta_y = Variable(dta_x.cuda()), Variable(dta_y.cuda().view(dta_y.size(0)))
            outputs = model(dta_x)
            _, preds = torch.max(outputs.data, 1)
            loss = criterion(outputs, dta_y)

            running_loss += loss.data[0]
            running_corrects += torch.sum(preds == dta_y.data)

        val_loss = running_loss / len(val_set)
        val_acc = running_corrects / len(val_set)

        epoch_log = '[%d/%d] Loss %.6f/%.6f Acc %.6f/%.6f' % (epoch, epochsize, train_loss, val_loss, train_acc, val_acc)
        print(epoch_log)
        # loging(epoch_log, './log/resnet50.log', epoch % 5 == 0)
    
    val_set = Arrayloader(X_test, y_test)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size = 1, shuffle = False, num_workers=4)
    val_pred = []
    for data in val_loader:
        dta_x, _ = data
        dta_x = Variable(dta_x.cuda())

        outputs = model(dta_x)
        _, preds = torch.max(outputs.data, 1)
        val_pred.append(lbl.inverse_transform(preds.cpu().numpy()[0][0]))
    train_preds[test_index] = val_pred
    
    train_logs[0].append(train_loss); train_logs[1].append(val_loss)
    train_logs[2].append(train_acc); train_logs[3].append(val_acc)
    print('Val:', sum(train_preds[test_index] == lbl.inverse_transform(y_test)) * 1.0 / len(y_test) )
    
    test_set = Arrayloader(test_feat, np.zeros_like(test_feat))
    test_loader = torch.utils.data.DataLoader(test_set, batch_size = 1, shuffle = False, num_workers=4)
    test_pred = []
    for data in test_loader:
        dta_x, _ = data
        dta_x = Variable(dta_x.cuda())
        outputs = model(dta_x)
        _, preds = torch.max(outputs.data, 1)
        test_pred.append(lbl.inverse_transform(preds.cpu().numpy()[0][0]))
    test_preds.append(test_pred)
    
    with codecs.open('test_.txt' + str(train_logs[3][-1]), 'w') as f:
        for i in range(len(test)):
            f.write(str(test_pred[i])  + '\t' + test[i] + '\n')

print(feature_file)
print('+++ Loss %.6f/%.6f Acc %.6f/%.6f' % (np.mean(train_logs[0]), np.mean(train_logs[1]), np.mean(train_logs[2]), np.mean(train_logs[3])))

train_val = train_val.drop('url', axis = 1)
train_val['label'] = train_preds
with codecs.open('0710_train.txt', 'w') as f:
    for i in range(train_preds.shape[0]):
        f.write(str(int(train_preds[i]))  + '\t' + train_val['img'].iloc[i] + '\n')

from scipy.stats import mode
test_preds = np.array(test_preds)
with codecs.open('0710_test4.txt', 'w') as f:
    for i in range(test_preds.shape[1]):
        f.write(str(mode(test_preds[:, i])[0][0])  + '\t' + test[i] + '\n')
