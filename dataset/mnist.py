import numpy as np
import paddle
from paddle.io import Dataset
# from paddle.vision.transforms import Compose, ColorJitter, Resize

class MNISTDataset(Dataset):
    def __init__(self, 
                 data_path='data/mnist.npz',
                 train=True,
                 reduce_dataset=False,
                 ssp=False):
        super(MNISTDataset, self).__init__()
        self.num_classes = 10
        self.train = train
        self.ssp = ssp

        # if self.ssp:
        #     assert self.train, \
        #         f'In ssp mode, self.train should be `True`, but get {self.train}.'
        #     assert reduce_dataset, \
        #         f'In ssp mode, reduce_dataset should be `True`, but get {reduce_dataset}.'

        with np.load(data_path, allow_pickle=True) as f:
            if self.train:
                x, y = f['x_train'].astype(np.float32), f['y_train']
            else:
                x, y = f['x_test'].astype(np.float32), f['y_test']

        if self.train and reduce_dataset:
            x, y = self._reduce_dataset(x, y)

        self._items = ((x / 255.0) - 0.1307) / 0.3081
        self._labels = y

    def __getitem__(self, idx):
        img = self._items[idx]
        label = self._labels[idx]

        if self.ssp:
            img, label = self._ssp_generate(img, label)
        else:
            label = self._onehot(label)

        img = img[None, :, :]
        
        return paddle.to_tensor(img), paddle.to_tensor(label)

    def __len__(self):
        return len(self._items)

    def _onehot(self, class_idx):
        onehot = np.zeros(self.num_classes, dtype=np.float32)
        onehot[class_idx] = 1

        return onehot

    def _reduce_dataset(self, x, y):
        print('Reducing training set. Images of 0,1,2,3,4 will be reduced to 10%.')
        mask = np.zeros(y.shape, dtype=np.bool_)
        for i in range(10):
            if i >= 5:
                mask[np.nonzero(y == i)[0]] = True
                continue
            eq2i_index = np.nonzero(y == i)[0]
            delete_num = int(len(eq2i_index)  * 0.9)
            # delete_index = np.random.choice(eq2i_index, delete_num, replace=False)
            reserve_index = eq2i_index[::10]
            mask[reserve_index] = True

        x = x[mask]
        y = y[mask]

        return x, y

    def _ssp_generate(self, ori_img, gt):
        assert len(ori_img.shape) == 2, \
            f'Input image should be 2-dim, but get shape of {ori_img.shape}.'

        label = np.zeros(4, dtype=np.float32)
        rot_type = np.random.randint(4)
        if gt == 6 or gt == 9:
            label[rot_type] = 0.5
            label[(rot_type + 2) % 4] = 0.5
        else:
            label[rot_type] = 1.0

        img = np.rot90(ori_img, rot_type)

        return img, label