import numpy as np
from keras.utils import Sequence, to_categorical
from scipy.ndimage.interpolation import shift
import os

class DataGenerator(Sequence):
    def __init__(self, labels, partition, mode="train", oversample=False, flip=False, shift=False,
                 classes=6, batch_size=5, timesteps=1, channels=1, cropped=True):
        """Initialization"""
        self.mode = mode
        self.folder = "single_" + self.mode if timesteps == 1 else "time_" + self.mode
        self.IDs = partition[mode]
        self.labels = labels
        self.classes = classes # 1 if regression
        self.timesteps = timesteps
        self.batch_size = batch_size
        self.class_samples_per_batch = self.batch_size // 3  # 3 class buckets used for oversampling
        self.channels = channels
        self.dims = (50, 146, 118) if cropped else (142, 322, 262)
        self.shuffle = True if mode == "train" else False
        self.oversample = oversample
        self.flip = flip
        self.shift = shift
        self.on_epoch_end() # set self.ID_queue on init


    def __len__(self):
        """Get number of batches per epoch"""
        if self.mode == "train" and self.oversample:
            batches_per_epoch = int(len(self.IDs_1) // self.class_samples_per_batch)
        else:
            batches_per_epoch = int(len(self.IDs) // self.batch_size)

        return batches_per_epoch

    def __getitem__(self, idx):
        """Get loaded batch of IDs"""
        batch_IDs = self.ID_queue[idx*self.batch_size : (idx+1)*self.batch_size]
        X, y = self.__load_batch(batch_IDs)

        return X, y

    def __oversample_IDs(self):
        """Get sequence of batches with minority class oversampling"""
        seq = []

        # get class labels
        c_vals = sorted(np.unique(list(self.labels.values())))
        c1, c2, c3to5 = c_vals[0], c_vals[1], tuple(c_vals[2:])

        # get IDs for each class label
        self.IDs_1 = np.array([ID for ID in self.IDs if self.labels[ID] == c1])
        self.IDs_2 = np.array([ID for ID in self.IDs if self.labels[ID] == c2])
        self.IDs_3to5 = np.array([ID for ID in self.IDs if self.labels[ID] in c3to5])

        for i in range(self.__len__()):
            batch = []

            idx_slice = np.arange(i * self.class_samples_per_batch, (i + 1) * self.class_samples_per_batch)

            sample_1 = self.IDs_1.take(idx_slice, mode='wrap')
            sample_2 = self.IDs_2.take(idx_slice, mode='wrap')
            sample_3to5 = self.IDs_3to5.take(idx_slice, mode='wrap')

            batch += list(sample_1)
            batch += list(sample_2)
            batch += list(sample_3to5)

            np.random.shuffle(batch)
            seq += batch

        return seq

    def on_epoch_end(self):
        """Update ID queue on epoch end"""
        self.ID_queue = []

        if self.shuffle:
            np.random.shuffle(self.IDs)

        if self.mode == "train" and self.oversample:
            # prepare sequence with oversampled minority classes
            self.ID_queue += self.__oversample_IDs()
        else:
            # prepare sequence with every ID occurring exactly once
            self.ID_queue += self.IDs

    def __random_flip(self, array):
        flip_x, flip_y, flip_z = np.random.binomial(1, 0.5, size=3).astype(bool)

        if flip_x:
            array = np.flip(array, axis=2)
        if flip_y:
            array = np.flip(array, axis=3)
        if flip_z:
            array = np.flip(array, axis=4)

        return array

    def __random_shift(self, array):
        shift_x, shift_y, shift_z = np.random.randint(-11, 11, 3)

        # shift along img dims (0=time dim, 1=color dim)
        array = shift(array, (0, 0, shift_z, shift_y, shift_z), cval=-800) # cval = -800 (background val)

        return array

    def __load_batch(self, batch_IDs):
        """Load batch data"""
        X = np.empty((self.batch_size, self.timesteps, self.channels, *self.dims))
        y = np.empty((self.batch_size, 1), dtype=float)

        for i, ID in enumerate(batch_IDs):
            prefix = './data/full/' if self.dims == (142, 322, 262) else './data/crop/'
            path = prefix + self.folder + "/vol_" + str(ID) + ".npy"
            img = np.load(path)

            # preprocess image
            if self.flip:
                img = self.__random_flip(img)
            if self.shift:
                img = self.__random_shift(img)

            X[i, :, :, :, :, :] = img
            y[i, 0] = self.labels[ID]

        if self.classes > 1:
            return X, to_categorical(y, num_classes=self.classes)

        return X, y