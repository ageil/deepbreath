# Adapted from https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly.html
import numpy as np

class DataGenerator(object):
    def __init__(self, folder, batch_size=16, timesteps=1, channels=1, 
                 dim_x=142, dim_y=322, dim_z=262, shuffle=True):
        'Initialization'
        self.folder = folder
        self.timesteps = timesteps
        self.channels = channels
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.dim_z = dim_z
        self.batch_size = batch_size
        self.shuffle = shuffle
        
    def __get_order(self, IDs):
        'Get order of indices'
        
        indices = np.arange(len(IDs))
        if self.shuffle == True:
            np.random.shuffle(indices)
        
        return indices
    
    def __onehot(self, y, labels):
        'Encode labels to onehot format'
        n_classes = len(np.unique(list(labels.values())))
        onehot = np.array([[1 if y[i] == j else 0 for j in range(n_classes)]
                           for i in range(y.shape[0])])
        return onehot
    
    def __gen_batch(self, labels, list_IDs_temp):
        'Generate sample of size batch_size'
        X = np.empty((self.batch_size, self.channels, self.timesteps, 
                      self.dim_x, self.dim_y, self.dim_z))
        y = np.empty((self.batch_size, 1, 1, 1, 1, 1), dtype=int)
        
        for i, ID in enumerate(list_IDs_temp):
            # store volume
            path = '/Users/anders1991/deepbreath/data/'+self.folder+"/vol_"+str(ID) +".npy"
            X[i, :, :, :, :, :] = np.load(path)
            y[i, 0, 0, 0, 0, 0] = labels[ID]
        
#        return X, self.__onehot(y, labels)
        return X, y
    
    def generate(self, labels, IDs):
        'Generate batches indefinitely'
        while True:
            indices = self.__get_order(IDs)
            
            imax = int(len(indices)/self.batch_size)
            for i in range(imax):
                # get list of IDs
                list_IDs_temp = [IDs[k] for k 
                                 in indices[i*self.batch_size : (i+1)*self.batch_size]]
                
                # generate data
                X, y = self.__gen_batch(labels, list_IDs_temp)
                
                yield X, y
