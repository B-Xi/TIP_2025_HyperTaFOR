import math
import torch
import scipy.io as sio
import numpy as np
from sklearn import preprocessing
import torch.utils.data as data
from torch.utils.data import Dataset

def sanity_check(all_set):
    nclass = 0
    nsamples = 0
    all_good = {}
    for class_ in all_set:
        if len(all_set[class_]) >= 200:
            all_good[class_] = all_set[class_][:200]
            nclass += 1
            nsamples += len(all_good[class_])
    return all_good


def load_data(image_file, label_file):
    image_data = sio.loadmat(image_file)
    label_data = sio.loadmat(label_file)

    data_key = image_file.split('/')[-1].split('.')[0]
    label_key = label_file.split('/')[-1].split('.')[0]
    data_all = image_data[data_key].astype(np.float32)  # dic-> narray , KSC:ndarray(512,217,204)
    GroundTruth = label_data[label_key].astype(np.int64)

    data = data_all.reshape(np.prod(data_all.shape[:2]), np.prod(data_all.shape[2:]))  # (111104,204)
    data_scaler = preprocessing.scale(data)  # (X-X_mean)/X_std,
    Data_Band_Scaler = data_scaler.reshape(data_all.shape[0], data_all.shape[1], data_all.shape[2])

    return Data_Band_Scaler, GroundTruth  # image:(512,217,3),label:(512,217)


def flip(data):
    y_4 = np.zeros_like(data)
    y_1 = y_4
    y_2 = y_4
    first = np.concatenate((y_1, y_2, y_1), axis=1)
    second = np.concatenate((y_4, data, y_4), axis=1)
    third = first
    Data = np.concatenate((first, second, third), axis=0)
    return Data


class matcifar(data.Dataset):
    """`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.
    Args:
        root (string): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """

    def __init__(self, imdb, train, d, medicinal):

        self.train = train  # training set or test set
        self.imdb = imdb
        self.d = d

        self.x1 = np.argwhere(self.imdb['set'] == 1)
        self.x2 = np.argwhere(self.imdb['set'] == 3)
        self.x1 = self.x1.flatten()
        self.x2 = self.x2.flatten()
        #        if medicinal==4 and d==2:
        #            self.train_data=self.imdb['data'][self.x1,:]
        #            self.train_labels=self.imdb['Labels'][self.x1]
        #            self.test_data=self.imdb['data'][self.x2,:]
        #            self.test_labels=self.imdb['Labels'][self.x2]

        if medicinal == 1:
            self.train_data = self.imdb['data'][self.x1, :, :, :]
            self.train_labels = self.imdb['Labels'][self.x1]
            self.test_data = self.imdb['data'][self.x2, :, :, :]
            self.test_labels = self.imdb['Labels'][self.x2]

        else:
            self.train_data = self.imdb['data'][:, :, :, self.x1]
            self.train_labels = self.imdb['Labels'][self.x1]
            self.test_data = self.imdb['data'][:, :, :, self.x2]
            self.test_labels = self.imdb['Labels'][self.x2]
            if self.d == 3:
                self.train_data = self.train_data.transpose((3, 2, 0, 1))  ##(17, 17, 200, 10249)
                self.test_data = self.test_data.transpose((3, 2, 0, 1))
            else:
                self.train_data = self.train_data.transpose((3, 0, 2, 1))
                self.test_data = self.test_data.transpose((3, 0, 2, 1))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:

            img, target = self.train_data[index], self.train_labels[index]
        else:

            img, target = self.test_data[index], self.test_labels[index]

        return img, target

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)


def radiation_noise(data, alpha_range=(0.9, 1.1), beta=1 / 25):
    alpha = np.random.uniform(*alpha_range)
    noise = np.random.normal(loc=0., scale=1.0, size=data.shape)
    return alpha * data + beta * noise


def get_train_test_loader(Data_Band_Scaler, GroundTruth, class_num, shot_num_per_class, args):
    [nRow, nColumn, nBand] = Data_Band_Scaler.shape #166*600
    # label start
    data_band_scaler = flip(Data_Band_Scaler)
    groundtruth = flip(GroundTruth) #498*1800
    HalfWidth = int((args.patch - 1) / 2)
    G = groundtruth[nRow - HalfWidth:2 * nRow + HalfWidth, nColumn - HalfWidth:2 * nColumn + HalfWidth] #174*608
    data = data_band_scaler[nRow - HalfWidth:2 * nRow + HalfWidth, nColumn - HalfWidth:2 * nColumn + HalfWidth, :]
    [Row, Column] = np.nonzero(G)  # (10249,) (10249,)
    # Sampling samples
    train = {}
    test = {}
    da_train = {}  # Data Augmentation
    m = int(np.max(G))  # 9
    nlabeled = shot_num_per_class
    for i in range(m):
        if i + 1 in args.known_classes:
            indices = [j for j, x in enumerate(Row.ravel().tolist()) if G[Row[j], Column[j]] == i + 1]
            np.random.shuffle(indices)
            nb_val = shot_num_per_class
            train[i] = indices[:nb_val]
            da_train[i] = []
            for j in range(math.ceil((200 - nlabeled) / nlabeled) + 1):
                da_train[i] += indices[:nb_val]
            test[i] = indices[nb_val:]
        else:
            indices = [j for j, x in enumerate(Row.ravel().tolist()) if G[Row[j], Column[j]] == i + 1]
            np.random.shuffle(indices)
            test[i] = indices

    train_indices = []
    test_indices = []
    da_train_indices = []
    for i in range(m):
        if i + 1 in args.known_classes:
            train_indices += train[i]
            test_indices += test[i]
            da_train_indices += da_train[i]
        else:
            test_indices += test[i]
    np.random.shuffle(test_indices)
    x_test = [Row[i] - HalfWidth for i in test_indices]
    y_test = [Column[i] - HalfWidth for i in test_indices]
    x_train = [Row[i] - HalfWidth for i in train_indices]
    y_train = [Column[i] - HalfWidth for i in train_indices]
    indices=[x_test, y_test, x_train, y_train]

    nTrain = len(train_indices)
    nTest = len(test_indices)
    da_nTrain = len(da_train_indices)
    imdb = {}
    imdb['data'] = np.zeros([2 * HalfWidth + 1, 2 * HalfWidth + 1, nBand, nTrain + nTest],
                            dtype=np.float32)  # (9,9,100,n)
    imdb['Labels'] = np.zeros([nTrain + nTest], dtype=np.int64)
    imdb['set'] = np.zeros([nTrain + nTest], dtype=np.int64)
    RandPerm = train_indices + test_indices
    RandPerm = np.array(RandPerm)
    for iSample in range(nTrain + nTest):
        imdb['data'][:, :, :, iSample] = data[
                                         Row[RandPerm[iSample]] - HalfWidth:  Row[RandPerm[iSample]] + HalfWidth + 1,
                                         Column[RandPerm[iSample]] - HalfWidth: Column[
                                                                                    RandPerm[iSample]] + HalfWidth + 1,
                                         :]
        imdb['Labels'][iSample] = G[Row[RandPerm[iSample]], Column[RandPerm[iSample]]].astype(np.int64)
    imdb['Labels'] = imdb['Labels'] - 1  # 1-16 0-15
    imdb['set'] = np.hstack((np.ones([nTrain]), 3 * np.ones([nTest]))).astype(np.int64)
    train_dataset = matcifar(imdb, train=True, d=3, medicinal=0)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=class_num * shot_num_per_class, shuffle=False,
                                               num_workers=0)
    test_dataset = matcifar(imdb, train=False, d=3, medicinal=0)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=0)

    # Data Augmentation for target domain for training
    imdb_da_train = {}
    imdb_da_train['data'] = np.zeros([2 * HalfWidth + 1, 2 * HalfWidth + 1, nBand, da_nTrain],
                                     dtype=np.float32)  # (9,9,100,n)
    imdb_da_train['Labels'] = np.zeros([da_nTrain], dtype=np.int64)
    imdb_da_train['set'] = np.zeros([da_nTrain], dtype=np.int64)

    da_RandPerm = np.array(da_train_indices)
    for iSample in range(da_nTrain):  # radiation_noise，flip_augmentation
        imdb_da_train['data'][:, :, :, iSample] = radiation_noise(
            data[Row[da_RandPerm[iSample]] - HalfWidth:  Row[da_RandPerm[iSample]] + HalfWidth + 1,
            Column[da_RandPerm[iSample]] - HalfWidth: Column[da_RandPerm[iSample]] + HalfWidth + 1, :])
        imdb_da_train['Labels'][iSample] = G[Row[da_RandPerm[iSample]], Column[da_RandPerm[iSample]]].astype(np.int64)

    imdb_da_train['Labels'] = imdb_da_train['Labels'] + 60  # 1-16 0-15
    imdb_da_train['set'] = np.ones([da_nTrain]).astype(np.int64)

    return train_loader, test_loader, imdb_da_train, indices


def get_target_dataset(Data_Band_Scaler, GroundTruth, class_num, shot_num_per_class, args):
    train_loader, test_loader, imdb_da_train, indices = get_train_test_loader(
        Data_Band_Scaler=Data_Band_Scaler, GroundTruth=GroundTruth, class_num=class_num,
        shot_num_per_class=shot_num_per_class, args=args)  # 9 classes and 5 labeled samples per class
    del Data_Band_Scaler
    del GroundTruth
    # target data with data augmentation
    target_da_datas = np.transpose(imdb_da_train['data'], (3, 2, 0, 1))  # (9,9,100, 1800)->(1800, 100, 9, 9)
    target_da_labels = imdb_da_train['Labels']  # (1800,)
    del imdb_da_train
    # metatrain data for few-shot classification
    target_da_train_set = {}
    for class_, path in zip(target_da_labels, target_da_datas):
        if class_ not in target_da_train_set:
            target_da_train_set[class_] = []
        target_da_train_set[class_].append(path)
    del target_da_labels
    del target_da_datas
    return train_loader, test_loader, target_da_train_set, indices


class Traindata(Dataset):
    def __init__(self, data, args):
        super(Traindata, self).__init__()
        self.data = data
        self.n_ways = args.n_ways
        self.n_open_ways = args.n_open_ways
        self.n_shots = args.n_shots
        self.n_queries = args.n_queries
        self.episodes = args.episodes
        self.classes = list(self.data.keys())

    def __getitem__(self, item):
        return self.get_episode(item)

    def get_episode(self, item):
        cls_sampled = np.random.choice(self.classes, self.n_ways, False)
        support_xs = []
        support_ys = []
        suppopen_xs = []
        suppopen_ys = []
        query_xs = []
        query_ys = []
        openset_xs = []
        openset_ys = []

        # Close set preparation
        for idx, the_cls in enumerate(cls_sampled):
            imgs = self.data[the_cls]
            support_xs_ids_sampled = np.random.choice(range(len(imgs)), self.n_shots, False)
            support_xs.extend([imgs[the_id] for the_id in support_xs_ids_sampled])
            support_ys.extend([idx] * self.n_shots)
            query_xs_ids = np.setxor1d(np.arange(len(imgs)), support_xs_ids_sampled)
            query_xs_ids = np.random.choice(query_xs_ids, self.n_queries, False)
            query_xs.extend([imgs[the_id] for the_id in query_xs_ids])
            query_ys.extend([idx] * self.n_queries)

        # Open set preparation
        cls_open_ids = np.setxor1d(self.classes, cls_sampled)
        cls_open_ids = np.random.choice(cls_open_ids, self.n_open_ways, False)
        for idx, the_cls in enumerate(cls_open_ids):
            imgs = self.data[the_cls]
            suppopen_xs_ids_sampled = np.random.choice(range(len(imgs)), self.n_shots, False)
            suppopen_xs.extend([imgs[the_id] for the_id in suppopen_xs_ids_sampled])
            suppopen_ys.extend([idx] * self.n_shots)
            openset_xs_ids = np.setxor1d(np.arange(len(imgs)), suppopen_xs_ids_sampled)
            openset_xs_ids_sampled = np.random.choice(openset_xs_ids, self.n_queries, False)
            openset_xs.extend([imgs[the_id] for the_id in openset_xs_ids_sampled])
            openset_ys.extend([the_cls] * self.n_queries)
        support_xs, query_xs, suppopen_xs, openset_xs = torch.tensor(np.asarray(support_xs)), torch.tensor(
            np.asarray(query_xs)), torch.tensor(np.asarray(suppopen_xs)), torch.tensor(np.asarray(openset_xs))
        support_ys, query_ys, openset_ys, suppopen_ys = np.array(support_ys), np.array(query_ys), np.array(
            openset_ys), np.array(suppopen_ys)

        return support_xs, support_ys, query_xs, query_ys, suppopen_xs, suppopen_ys, openset_xs, openset_ys,

    def __len__(self):
        return self.episodes
