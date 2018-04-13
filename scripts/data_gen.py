import numpy as np
from keras.utils import Sequence, to_categorical

class DataGenerator(Sequence):
    def __init__(self, labels, partition, mode="train", oversample=False,
                 classes=6, batch_size=5, timesteps=1, channels=1, dims=(142, 322, 262)):
        """Initialization"""
        self.mode = mode
        self.folder = "single_" + self.mode if timesteps == 1 else "time_" + self.mode
        self.IDs = partition[mode]
        self.labels = labels
        self.classes = classes
        self.timesteps = timesteps
        self.batch_size = batch_size
        self.class_samples_per_batch = self.batch_size // 3  # 3 class buckets used for oversampling
        self.channels = channels
        self.dims = dims
        self.shuffle = True if mode == "train" else False
        self.oversample = oversample
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

    def __load_batch(self, batch_IDs):
        """Load batch data"""
        X = np.empty((self.batch_size, self.timesteps, self.channels, *self.dims))
        y = np.empty((self.batch_size, 1), dtype=float)

        for i, ID in enumerate(batch_IDs):
            path = './data/' + self.folder + "/vol_" + str(ID) + ".npy"
            X[i, :, :, :, :, :] = np.load(path)
            y[i, 0] = self.labels[ID]

        if self.classes > 1:
            return X, to_categorical(y, num_classes=self.classes)

        return X, y