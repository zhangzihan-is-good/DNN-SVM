import copy

import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn import svm

print('Loading data ...')

train = np.load('train_11.npy')
train_label = np.load('train_label_11.npy')

print('Size of all data: {}'.format(train.shape))

class TIMITDataset(Dataset):
    def __init__(self, X, y=None):
        self.data = torch.from_numpy(X).float()
        if y is not None:
            y = y.astype(int)
            self.label = torch.LongTensor(y)
        else:
            self.label = None

    def __getitem__(self, idx):
        if self.label is not None:
            return self.data[idx], self.label[idx]
        else:
            return self.data[idx]

    def __len__(self):
        return len(self.data)

VAL_RATIO = 0.2
#使用百分比的分割方法，通过隔断截取来简化数据集
percent = int(train.shape[0] * (1 - VAL_RATIO))
step = 1000
train_x, train_y, val_x, val_y = train[:percent:step], train_label[:percent:step], train[percent::step], train_label[percent::step]
print('Size of training set: {}'.format(train_x.shape))
print('Size of validation set: {}'.format(val_x.shape))
letest = len(val_y)
rsave = np.zeros((letest, 1))
#训练和测试环节
for i in range(0,39):
    train_yi = copy.copy(train_y)
    for m,ele in enumerate(train_yi):
        if int(ele) == i:
            train_yi[m] = 0
        else:
            train_yi[m] = 1
    linear_svc = svm.SVC(probability=True)
    linear_svc.fit(train_x, train_yi)
    rst = linear_svc.predict(val_x)
    rstp = linear_svc.predict_proba(val_x)
    rsave = np.hstack((rsave,rstp[:,0:1]))
#因为初始化的时候加了一行0，所以这里要去掉
idx = np.argmax(rsave, axis=1)-1
sumtest = 0
for i,m in enumerate(idx):
    if int(m) == int(val_y[i]):
        sumtest += 1
print(sumtest / letest)
