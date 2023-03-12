# Adapted from longhai

import random
from sklearn.neighbors import NearestNeighbors
import numpy as np
from collections import Counter
from sklearn.cluster import k_means
import copy
from scipy import stats
import time
import math
from core.distMix import distmix
import pandas as pd


# Calculate how many majority samples are in the k nearest neighbors of the minority samples
def number_maj(imbalanced_featured_data, minor_feature_data, minor_label, imbalanced_label_data, method, weigths_boolean = True, nbins=3, idnum = list(range(11,24)), idbin = [], idcat = list(range(1,11))):
    #gower_1 = gower.gower_matrix(imbalanced_featured_data)
    #gower_2 = gower.gower_matrix(minor_feature_data, imbalanced_featured_data)
    gower_1 = distmix(imbalanced_featured_data, method = method, weigths_boolean = weigths_boolean, nbins=nbins, idnum = idnum, idbin = [], idcat = idcat )
    gower_2 = distmix(minor_feature_data, imbalanced_featured_data, method = method, weigths_boolean = weigths_boolean, nbins=nbins, idnum = idnum, idbin = [], idcat = idcat )
    print(imbalanced_featured_data.shape)
    print(minor_feature_data.shape)
    print(gower_1)
    print(gower_2)
    nnm_x = NearestNeighbors(n_neighbors=6, metric="precomputed").fit(gower_1).kneighbors(gower_2,return_distance=False)[:, 1:]
    nn_label = (imbalanced_label_data[nnm_x] != minor_label).astype(int)
    n_maj = np.sum(nn_label, axis=1)
    return n_maj


class RSMOTENC:
    """
    class RSMOTE usage is as follows:
    "
    clf = LogisticRegression()
    data_Rsmote = RSMOTE.RSmote(data, ir=1, k=5).over_sampling()
    "
    """

    def __init__(self, cat_vars, ir=1, k=5, random_state=None, method = "ahmad", weigths_boolean = True, nbins=3):
        """
        :param data: array for all data with label in 0th col.
        :param ir: imbalanced ratio of synthetic data.
        :param k: Number of nearest neighbors.
        """
    
        self.cat_vars = cat_vars
        
        #self.num_idx = [x for x in list(range(1,data.shape[1]-1)) if x not in cat_idx]
        self.IR = ir
        self.k = k
        self.new_index = 0
        self.new_points = 0
        self.random_state = random_state
        self.N = 0
        self.synthetic = None
        self.method = method
        self.weigths_boolean = weigths_boolean
        self.nbins = nbins
        #print(self.cat_idx)
        #print(self.num_idx)

    def _div_data(self, data):
        """
        divide the dataset.
        :return: None
        """
        
        self.data = data
        
        count = Counter(self.data[:, 0])
        a, b = set(count.keys())
        self.tp_less, self.tp_more = (a, b) if count[a] < count[b] else (b, a)

        data_less = self.data[self.data[:, 0] == self.tp_less]
        data_more = self.data[self.data[:, 0] == self.tp_more]

        self.train_less = data_less
        self.train_more = data_more

        self.train = np.vstack((self.train_more, self.train_less))
    
        self.n_train_less, self.n_attrs = self.train_less.shape
        
        self.cat_idx = []
        for i in self.cat_vars:
            self.cat_idx.append(self.data.columns[1:].get_loc(i))
            
        self.num_idx = [x for x in list(range(self.data.shape[1]-1)) if x not in self.cat_idx]
        

    def over_sampling(self, data):
        
        self._div_data(data)
        
        if self.k + 1 > self.n_train_less:
            print('Expected n_neighbors <= n_samples,  but n_samples = {}, n_neighbors = {}, '
                  'has changed the n_neighbors to {}'.format(self.n_train_less, self.k + 1, self.n_train_less))
            self.k = self.n_train_less - 1
        data_less_filter = []
        num_maj_filter = []
        length_less = len(self.train_less)
        num_maj = number_maj(self.train[:, 1:], self.train_less[:, 1:], self.tp_less, self.train[:, 0], self.method, self.weigths_boolean, self.nbins, self.num_idx, [], self.cat_idx)
        print(num_maj)
        for m in range(len(num_maj)):
            if num_maj[m] < self.k:
                data_less_filter.append(self.train_less[m])
                num_maj_filter.append(num_maj[m])

        if len(data_less_filter) >= 3:        
            self.train_less = np.array(data_less_filter)
        else:
            num_maj_filter = []
            for m in range(len(num_maj)):
                if num_maj[m] < self.k:
                    num_maj_filter.append(num_maj[m])
                else:
                    num_maj_filter.append(num_maj[m]-1)

        print(self.train_less)
        print(self.train_less.shape)

        chk = time.time()
        #gower_more = gower.gower_matrix(self.train_more[:, 1:])
        gower_more = distmix(self.train_more[:,1:], method = self.method, weigths_boolean = self.weigths_boolean, nbins = self.nbins, idnum = self.num_idx, idbin = [], idcat = self.cat_idx)
        print("Gower1", time.time() - chk)

        chk = time.time()
        #gower_less = gower.gower_matrix(self.train_less[:, 1:])
        gower_less = distmix(self.train_less[:,1:], method = self.method, weigths_boolean = self.weigths_boolean, nbins = self.nbins, idnum = self.num_idx, idbin = [], idcat = self.cat_idx )
        print("Gower2",time.time() - chk)

        chk = time.time()
        #gower_less_more = gower.gower_matrix(self.train_less[:, 1:], self.train_more[:, 1:])
        gower_less_more = distmix(self.train_less[:, 1:], self.train_more[:, 1:], method = self.method, weigths_boolean = self.weigths_boolean, nbins = self.nbins, idnum = self.num_idx, idbin = [], idcat = self.cat_idx )
        print("Gower3",time.time() - chk)

        chk = time.time()
        if len(gower_less_more) < (self.k + 1):
            n_neigh = math.ceil(len(gower_less_more)/2) + 1
        else:
            n_neigh = self.k + 1
        distance_more, nn_array_more = NearestNeighbors(n_neighbors=n_neigh, metric="precomputed").fit(gower_more).kneighbors(gower_less_more, return_distance=True)
        print("NN1",time.time() - chk)

        chk = time.time()
        print(len(gower_less))
        print(gower_less.shape)
        if len(gower_less) < (self.k + 1):
            n_neigh = math.ceil(len(gower_less)/2) + 1
        else:
            n_neigh = self.k + 1
        distance_less, nn_array = NearestNeighbors(n_neighbors=n_neigh, metric="precomputed").fit(gower_less).kneighbors(gower_less, return_distance=True)
        print("NN2",time.time() - chk)

        # distance = 0 if n_neigh=1 ERROR!!!
        distance_less = distance_less.sum(axis=1)
        distance_more = distance_more.sum(axis=1)
        distance = distance_less / distance_more
        # print(distance)
        density = 1 / distance  # calculate density

        density = list(map(lambda x: min(100, x), density))  # Control the maximum density range at 100

        # The density is sorted below, and the minority samples are also sorted in order of density.
        density_sorted = sorted(range(len(density)), key=lambda a: density[a], reverse=True)  # sorted
        data_resorted = []
        density_sorted_data = []
        num_sorted = []
        for i in range(len(self.train_less)):
            data_resorted.append(self.train_less[density_sorted[i]])
            density_sorted_data.append(density[density_sorted[i]])
            num_sorted.append(num_maj_filter[density_sorted[i]])

        density = np.array(density_sorted_data)
        cluster_big_density = []
        cluster_small_density = []
        cluster_big_data = []
        cluster_small_data = []
        cluster_big_num = []
        cluster_small_num = []
        cluster = k_means(X=density.reshape((len(density), 1)), n_clusters=2)
        for i in range(cluster[1].shape[0]-1): #THE -1 WAS ADDED
            if cluster[1][i] != cluster[1][i + 1]:  # Partition cluster
                cluster_big_density = density[:i + 1]
                cluster_big_data = np.array(data_resorted)[:i + 1, :]
                cluster_big_num = num_sorted[:i + 1]
                cluster_small_density = density[i + 1:]
                cluster_small_data = np.array(data_resorted)[i + 1:, :]
                cluster_small_num = num_sorted[i + 1:]
                break

        # If there is only one point in a cluster, do not divide the cluster
        if len(cluster_big_data) < 2 or len(cluster_small_data) < 2:
            cluster_big_data = np.array(data_resorted)
            cluster_big_density = density
            cluster_big_num = num_sorted
            flag = 1  # if flag==1 only run big cluster once
        else:
            flag = 2
        sum_0 = 0
        sum_1 = 0
        # Calculate weight
        for p in range(len(cluster_big_num)):
            sum_0 += (5 - cluster_big_num[p]) / self.k + 1
        for p in range(len(cluster_small_num)):
            sum_1 += (5 - cluster_small_num[p]) / self.k + 1

        ratio = []  # save the every cluster's totol weight
        ratio.append(sum_0)
        ratio.append(sum_1)
        wight = [5 / 6, 4 / 6, 3 / 6, 2 / 6, 1 / 6]
        kk = self.k
        diff = len(self.train_more) - length_less  # the number of samples need to synthesize
        totol_less = len(self.train_less)

        print("Flag ", flag)

        for i in range(flag):
            if i == 0:  # big cluster
                density = cluster_big_density
                self.n_train_less = len(cluster_big_data)
                self.train_less = cluster_big_data
                maj_num_ab = cluster_big_num
            else:  # small cluster
                density = cluster_small_density
                self.n_train_less = len(cluster_small_data)
                self.train_less = cluster_small_data
                maj_num_ab = cluster_small_num

            self.k = min(len(self.train_less) - 1, kk)  # if len(self.train_less)<k,set k =len(self.train_less)

            # The number of sample points that need to be inserted at each point
            if flag == 1:
                number_synthetic = int(len(self.train_more) / self.IR - len(self.train_less))
            else:
                if i == 0:
                    number_synthetic = int((len(self.train_less) / totol_less) * diff)
                    len_big = number_synthetic
                else:
                    number_synthetic = diff - len_big

            print(ratio)
            # Calculate how many points should be inserted for each sample
            N = list(map(lambda x: int((x / ratio[i]) * number_synthetic), wight))
            print("N ", N)
            print("N Sum ", np.sum(N))
            print("number_synthetic ", number_synthetic)
            self.reminder = number_synthetic - sum(N)
            self.num = 0

            #gower_less = gower.gower_matrix(self.train_less[:, 1:])
            gower_less = distmix(self.train_less[:,1:], method = self.method, weigths_boolean = self.weigths_boolean, nbins = self.nbins, idnum = self.num_idx, idbin = [], idcat = self.cat_idx )
            if len(gower_less) < (self.k + 1):
                n_neigh = math.ceil(len(gower_less)/2) + 1
            else:
                n_neigh = self.k + 1
            neighbors = NearestNeighbors(n_neighbors=self.k + 1, metric="precomputed").fit(gower_less)
            nn_array = neighbors.kneighbors(gower_less, return_distance=False)

            self.synthetic = np.zeros((number_synthetic, self.n_attrs - 1))
            print("len: ", self.train_less.shape[0])
            for p in range(self.train_less.shape[0]):
                self._populate(p, nn_array[p][1:], number_synthetic, N, maj_num_ab)

            print("self.num: ", self.num)
            print("self.reminder: ", self.reminder)

            label_synthetic = np.array([self.tp_less] * number_synthetic).reshape((number_synthetic, 1))
            np.random.seed(self.random_state)
            synthetic_dl = copy.deepcopy(self.synthetic)
            synthetic_dl = np.hstack((label_synthetic, synthetic_dl))  # class column

            data_res = synthetic_dl[:self.new_index,:]
            if i == 0:
                return_data = np.vstack((copy.deepcopy(self.train), data_res))
                if flag == 1:
                    return return_data
                self.new_index = 0
            else:
                return_data = np.vstack((copy.deepcopy(return_data), data_res))

                return return_data

    # for each minority class samples, generate N synthetic samples.
    def _populate(self, index, nnarray, number_synthetic, N, maj_num_ab):

        random.seed(self.random_state)
        if self.num < self.reminder:
            turn = N[maj_num_ab[index]] + 1
        else:
            turn = N[maj_num_ab[index]]
        for j in range(turn):
            if self.new_index < number_synthetic:
                if self.k == 1:
                    nn = 0
                else:
                    nn = random.randint(0, self.k - 1)
                #Numerical Variables
                #print(nnarray[nn])
                #print(self.train_less.shape)
                #print(self.train_less[nnarray[nn], self.num_idx])
                #print(self.train_less[index, self.num_idx])
                dif = self.train_less[:,1:][nnarray[nn], self.num_idx] - self.train_less[:,1:][index, self.num_idx]
                gap = random.random()
                self.synthetic[self.new_index, self.num_idx] = self.train_less[:,1:][index, self.num_idx] + gap * dif
                #Categorical Variables
                self.synthetic[self.new_index, self.cat_idx] = stats.mode(self.train_less[:,1:][:, self.cat_idx][nnarray,:])[0]
                #self.synthetic[self.new_index, self.cat_idx] = self.train_less[:,1:][nnarray[nn], self.cat_idx][0]
                #self.synthetic[self.new_index, self.cat_idx] = stats.mode(self.train_less[:,1:][nnarray[nn], self.cat_idx])[0]
                #print(self.synthetic[self.new_index, :])
                ###
                self.new_index += 1
            else:
                break
        self.num += 1

    def fit_resample(self, X, y):
        
        self.data = copy.deepcopy(self.X)
        self.data.insert(0, "anomaly", self.y)
        self.data = self.data.to_numpy()

        data_Rsmote = self.over_sampling(self.data)

        new_X = pd.DataFrame(data_Rsmote[:,1:], columns = X.columns)
        new_y = pd.DataFrame(data_Rsmote[:,0])

        new_X[self.cat_vars] = new_X[self.cat_vars].astype("category")

        #Change new_X and new_y column names, according to X and y, respectively
        
        return new_X, new_y