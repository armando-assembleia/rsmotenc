#from sys import breakpointhook
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats.contingency import crosstab
from sklearn.preprocessing import KBinsDiscretizer
import time
import copy
#from threading import Thread
from multiprocessing import Process

THREADS = 10

def weigths_num_var(data, nbins=0):
    #suppose it is a matrix

    # IF matrix and dataframe
    rn = data.shape[0]
    # BINS
    if nbins == 0:
        data = np.floor(data)
    else:
        try:
            kbins = KBinsDiscretizer(n_bins=nbins, encode='ordinal', strategy='uniform')
            data = kbins.fit_transform(data)
        except ValueError:
            raise ValueError("NBINS must be an non-negative integer number, except 1")
    
    col = data.shape[1]

    if col!=1:

        newdata = [0]*col
        
        for i in range(col):
            newdata[i] = _newdist(data, col, i)
    
        weigths = [0]*col
        for k in range(col):
            S = newdata[k].shape[0]
            #print("S calculated")
            newdata[k] = np.triu(newdata[k], k=1)
            #print("tri calculates")
            weigths[k] = np.sum(newdata[k]) / (S * (S + 1) / 2)
            #print("one weigth calculated")
    
    return(np.array(weigths))

def _newdist(data, col, colnum):
    
    nvar = range(col)
    n = len(np.unique(data[:,colnum]))
    var = [0]*(col-1)
    prob = np.zeros((1,n)) #np.array([[0]*n])
    # VER MAP
    for i in range(col-1):
        var[i] = crosstab(data[:, np.delete(nvar,colnum)[i]],data[:,colnum])[1]
        prob = np.concatenate( ( prob, var[i]/np.sum(var[i],axis=0) ) )
    
    prob = prob[1:,:]
    #matnew = np.array([0.0]*n*n).reshape(n,n)
    matnew = np.zeros((n,n))
    for i in range(n):
        for j in range(i,n):
            #matnew[j,i] = matnew[i,j] = ( np.sum(np.max(prob[:,[i,j]], axis=1)) - (col - 1) ) / (col - 1)
            matnew[i,j] = ( np.sum(np.max(prob[:,[i,j]], axis=1)) - (col - 1) ) / (col - 1)
    
    matnew = np.triu(matnew,1).T + matnew

    return(matnew)

def cooccur(data1, data2=None):
    #THESIS COMMENT THAT IS MORE EFFICIENT WHEN DATA2=NONE

    #suppose it is a matrix

    # IF matrix and dataframe

    initial_data2 = copy.copy(data2)

    if data2 is None:
        #copy
        #simplificar neste caso para matrix simetrica
        data1_2 = data2 = data1
    else:
        data1_2 = np.concatenate((data1,data2))

    rn1 = data1.shape[0]
    rn2 = data2.shape[0]
    # BINS
    col = data1_2.shape[1]

    if col!=1:

        newdata = [0]*col
        #newdata = np.zeros((col,dim,dim))

        for i in range(col):
            newdata[i] = _newdist(data1_2, col, i)

        distmat = np.zeros((rn1,rn2)) #np.array([0]*rn*rn).reshape(rn,rn)
        #distmat = np.zeros((col,rn1,rn2))
        #print((rn1,rn2,col))
        
        if initial_data2 is None:
            start = time.time()
            for i in range(rn1):
                distsum = np.zeros((rn2-i,col))
                data_i = data1[i, :col]
                newdata_aux = [newdata[k][data_i[k],:] for k in range(col)]
                for j in range(i,rn2):
                    data_j = data2[j, :col]
                    distsum[j-i] = [newdata_aux[k][data_j[k]] for k in range(col)]
                distmat[i:,i] = distmat[i,i:] = np.sum(np.array(distsum)**2, axis=1)
                #print(i)
            print("Same DataFrame")
        else:
            start = time.time()
            distmat = [None] * rn1
            threads = [Process(target = create_distance_row, args=(i, distmat, data1, data2, rn2, col, newdata)) for i in range(rn1)]
            [thread.start() for thread in threads] 
            [thread.join() for thread in threads]    
            # pool = multiprocessing.Pool(processes=THREADS)
            # distmat = pool.map(create_distance_row(data1, data2, rn2, col, newdata), range(rn1))
            
            print("Different DataFrame")
            print(time.time()-start)
                 
    else:
        distmat = cdist(data1, data2, 'hamming')
        print("Due to only 1 variable, simple matching distance is calculated instead! To produce coocurrence distance, it requires at least 2 variables.")

    return(distmat)

def create_distance_row(nrow, distmat, data1, data2, rn2, col, newdata):
    distsum = np.zeros((rn2,col))
    data_i = data1[nrow, :col]
    newdata_aux = [newdata[k][data_i[k],:] for k in range(col)]
    for j in range(rn2):
        data_j = data2[j, :col]
        distsum[j] = [newdata_aux[k][data_j[k]] for k in range(col)]
    distmat[nrow] =  np.sum(np.array(distsum)**2, axis=1)
    return

def mahalanobis_array(data, y, cov_inv):

    x_y = data - y

    #left = np.dot(x_y, cov_inv)
    #mahal = np.dot(left, x_y.T)
    #return np.sqrt(mahal.diagonal())

    left = np.dot(x_y, cov_inv)
    mahal = np.einsum('ij,ji->i', left, x_y.T)
    return np.sqrt(mahal)

def mahalanobis_matrix2(data1, data2=None, cov_method="standard"):

    from scipy.spatial.distance import mahalanobis
    from sklearn.covariance import MinCovDet

    dim1 = data1.shape[0]

    if data2 is None:
        data1_2 = data1
        dim2 = dim1
        distance_matrix = np.zeros((dim1,dim1))
    else:
        dim2 = data2.shape[0]
        distance_matrix = np.zeros((dim1,dim1+dim2)) #np.array([0]*dim*dim).reshape(dim,dim)

        data1_2 = np.concatenate((data1,data2))

    start = time.time()
    if cov_method=="standard":
        cov = np.cov(data1_2.T)
        cov_inv = np.linalg.inv(cov)
    elif cov_method == "mcd":
        cov = MinCovDet(random_state=0).fit(data1_2)
        cov_inv = np.linalg.inv(cov.covariance_)

    print("Inv cov time: ", time.time()-start)

    start = time.time()
    for i in range(dim1):
        distance_matrix[i,:] = mahalanobis_array(data1_2, data1_2[i,:],cov_inv)
    print("Mahalanobis arrays time: ", time.time()-start)

    if data2 is not None:
        distance_matrix = distance_matrix[:,dim1:(dim1+dim2)]

    return distance_matrix

def mrv(data1, data2):
    range_by_column = np.ptp(np.concatenate((data1,data2)), axis=0)
    #Normalize data by the squared range
    x1 = data1 / range_by_column
    x2 = data2 / range_by_column

    return cdist(x1, x2, 'cityblock')

def ser2(data1, data2):
    range_by_column = np.ptp(np.concatenate((data1,data2)), axis=0)
    #Normalize data by the squared range
    x1 = data1 / range_by_column**2
    x2 = data2 / range_by_column**2

    return cdist(x1, x2, 'sqeuclidean')

def _check_dataset(data, idnum, idbin, idcat):

    if type(data) is not np.ndarray:
        if type(data) is pd.DataFrame:
            data = data.to_numpy()
        else:
            raise ValueError("Variable data should be a Numpy Array or a DataFrame")

    from sklearn.preprocessing import OrdinalEncoder
    encoder = OrdinalEncoder()
    #IMPORTANTE ALTERAR
    print(idnum)
    print(idbin)
    print(idcat)
    data[:, idbin + idcat] = encoder.fit_transform(data[:, idbin + idcat])

    x_num = data[:,idnum]
    x_bin = data[:,idbin]
    x_cat = data[:,idcat]
    x_bin_cat = data[:,idbin + idcat]


    #List of IFS
    try:
        x_num = x_num.astype(float)
    except ValueError:
        raise ValueError("Numerical variables can't be transformed into float")
    ###
    try:
        x_bin = x_bin.astype(int)
    except ValueError:
        raise ValueError("Binary variables can't be transformed into integers")
    ###
    try:
        x_cat = x_cat.astype(int)
    except ValueError:
        raise ValueError("Categorical variables can't be transformed into integers. Check whether they are strings!")
    try:
        x_bin_cat = x_bin_cat.astype(int)
    except ValueError:
        raise ValueError("Categorical variables can't be transformed into integers. Check whether they are strings!")

    return data, x_num, x_bin, x_cat, x_bin_cat

def distmix(data1, data2=None, method = "gower", weigths_boolean = True, nbins=0, idnum = [], idbin = [], idcat = []):

    data1, x_num1, x_bin1, x_cat1, x_bin_cat1 = _check_dataset(data1, idnum, idbin, idcat)
    rows_data1 = len(data1)
    initial_data2 = copy.copy(data2)

    if data2 is None:
        data1_2  = data2  = data1
        x_num1_2 = x_num2  = x_num1
        x_bin1_2 = x_bin2 = x_bin1
        x_cat1_2 = x_cat2 = x_cat1
        x_bin_cat1_2 = x_bin_cat2 = x_bin_cat1
        rows_data2 = rows_data1
    else:
        data2, x_num2, x_bin2, x_cat2, x_bin_cat2 = _check_dataset(data2, idnum, idbin, idcat)
    
        data1_2 = np.concatenate((data1,data2))
        data1_2, x_num1_2, x_bin1_2, x_cat1_2, x_bin_cat1_2 = _check_dataset(data1_2, idnum, idbin, idcat)

        rows_data2 = len(data2)

    
    ########################
    # 1. Numerical Component
    #########################
    print("---------------------------")

    if len(idnum) == 0:
        num = 0
        msd = 0
        dist_numeric = 0
    else:
        x1 = x_num1
        x2 = x_num2
        x1_2 = x_num1_2
        num = len(idnum)
        msd = np.mean(np.std(x1_2, axis=0))
        
        ### Ahmad Distance - Multiplying the weigths of the numerical variables
        if (weigths_boolean) & (x1_2.shape[1]>1):
            weigths = weigths_num_var(x1_2, nbins)
            print("Weigths calculated")
            x1 = x1 * weigths
            x2 = x2 * weigths
            #print("Weigths multiplied")
            #print(weigths)
        

        if method == "gower":
            dist_numeric = mrv(x1,x2)
        elif method == "wishart":
            dist_numeric = cdist(x1, x2, 'seuclidean', V=None)**2
        elif method == "podani":
            dist_numeric = ser2(x1,x2)
        elif method == "huang":
            dist_numeric = cdist(x1, x2, 'sqeuclidean')
        elif method == "harikumar" or method == "ahmad_l1":
            dist_numeric = cdist(x1, x2, 'cityblock')
        elif method == "ahmad":
            dist_numeric = cdist(x1, x2, 'sqeuclidean')
        elif method == "ahmad_mahalanobis":
            if initial_data2 is None:
                dist_numeric = mahalanobis_matrix2(x1, None, cov_method="standard")
                print(np.any(np.isnan(dist_numeric)))
                print(np.all(np.isfinite(dist_numeric)))
                print(dist_numeric)
            else:
                dist_numeric = mahalanobis_matrix2(x1, x2, cov_method="standard")
        print("Distance matrix calculated") 

    ######################
    # 2. Binary Component
    ######################
    print("---------------------------")

    if len(idbin) == 0:
        bin = 0
        dist_binary = 0
    else:
        bin = len(idbin)
        x1 = x_bin1
        x2 = x_bin2
        x1_2 = x_bin1_2

        dist_binary = cdist(x1, x2, 'hamming')

        if method == "huang" or method == "harikumar":
            dist_binary = dist_binary * bin

    print("Binary finished")

    ##########################
    # 3. Categorical Component
    ##########################
    print("---------------------------")

    if method == "ahmad" or method == "ahmad_mahalanobis" or method == "ahmad_l1":
        bin = len(idbin)
        cat = len(idcat)
        print("Calculating coocccur...")
        start = time.time()
        if initial_data2 is None:
            dist_cat = cooccur(x_bin_cat1, None)
        else:
            dist_cat = cooccur(x_bin_cat1, x_bin_cat2)
        print("Cooccur time: ", time.time()-start)
    else: 
        if len(idcat) == 0:
            cat = 0
            dist_cat = 0
        else:
            cat = len(idcat)
            x1 = x_cat1
            x2 = x_cat2
            x1_2 = x_cat1_2
            if method == "harikumar":
                if initial_data2 is None:
                    dist_cat = cooccur(x1, None)
                else:
                    dist_cat = cooccur(x1, x2)
            else:
                dist_cat = cdist(x1, x2, 'hamming')
                if method == "huang":
                    dist_cat = dist_cat * cat

    print("Cat finished")
    #####################################
    # LAST STEP: Sum different components
    #####################################
    print("---------------------------")

    nvar = num + bin + cat

    #print(dist_cat)

    if method == "gower":
        dist_mix = dist_numeric * 1/nvar + dist_binary * bin/nvar + dist_cat * cat/nvar
    elif method == "wishart":
        dist_mix = (dist_numeric *  1/nvar + dist_binary * bin/nvar + dist_cat * cat/nvar)**0.5
    elif method == "podani":
        dist_mix = (dist_numeric + dist_binary * bin + dist_cat * cat)**0.5
    elif method == "huang":
        dist_mix = dist_numeric + dist_binary * msd + dist_cat * msd
    elif method == "harikumar":
        dist_mix = dist_numeric + dist_binary + dist_cat
    elif method == "ahmad" or method == "ahmad_mahalanobis" or method == "ahmad_l1":
        dist_mix = dist_numeric + dist_cat
  
    return(dist_mix)




'''
from import_datasets import df_domain_ordinal

df_domain_ordinal =df_domain_ordinal.drop("lgd_3",axis=1)

a = distmix(df_domain_ordinal.to_numpy(), method = "gower", weigths_boolean = False, nbins=0, idnum = list(range(11,24)), idbin = [], idcat = list(range(1,11)) )
b = distmix(df_domain_ordinal.to_numpy(), method = "huang", weigths_boolean = False, nbins=0, idnum = list(range(11,24)), idbin = [], idcat = list(range(1,11)) )
start = time.time()
c = distmix(df_domain_ordinal.to_numpy(), data2 = df_domain_ordinal.to_numpy(),  method = "ahmad_mahalanobis", weigths_boolean = True, nbins=3, idnum = list(range(10,24)), idbin = [], idcat = list(range(10)) )
print(time.time()-start) 147 295

c = distmix(a, method = "ahmad", weigths_boolean = True, nbins=3, idnum = list(range(3)), idbin = [], idcat = [3,4] )


np.cov(df_domain_ordinal.to_numpy()[list(range(11,24))])

start = time.time()
mahalanobis_matrix(b,b, cov_method="standard")
print(time.time()-start)

start = time.time()
mahalanobis_matrix2(df_domain_ordinal.to_numpy(), cov_method="standard")
print(time.time()-start)

a = np.array([[1,3,7],[0,6,1],[0,0,4]])
b = np.array([[1,1,1],[0,7,7]])
a = np.array([[1,3,7,4,9],[0,6,1,8,3],[0,0,4,1,1],[0,0,0,1,5],[0,0,0,0,3]])
b = np.array([[3,3,1,0,3],[2,2,0,1,1]])

distmix(a,b,method = "ahmad_mahalanobis", weigths_boolean = False, nbins=0, idnum = [0,1,2], idbin = [], idcat = [3,4] )
distmix(np.concatenate((a,b)), method = "ahmad_mahalanobis", weigths_boolean = True, nbins=3, idnum = [0,1,2], idbin = [], idcat = [3,4] )[:len(a),len(a):(len(a)+len(b))]

# "gower" "wishart" "podani" "huang" "harikumar" "ahmad" "ahmad_mahalanobis"

from sklearn.preprocessing import OrdinalEncoder

encoder = OrdinalEncoder()
encoder.fit_transform(a)

a = np.array(range(12)).reshape(3,2,2)
a[1,:,:].shape

a = np.array(range(10000)).reshape(100,100)
b = np.array(range(10000)).reshape(100,100)

start = time.time()
for i in range(100):
    for j in range(100):
        for k in range(10):
            a[i,k]
            b[i,k]
print(time.time()-start)

start = time.time()
for i in range(100):
    a_i = a[i,range(10)]
    for j in range(100):
        b_j = b[j, range(10)]
        for k in range(10):
            a_i[k]
            b_j[k]
print(time.time()-start)

from joblib import Parallel, delayed
import time, math

def my_fun(i,other):
    time.sleep(1)
    return [6,5]

num = 10

start = time.time()
for i in range(num):
    my_fun(i)
end = time.time()
print(end-start)

#start = time.time()
a = Parallel(n_jobs=2)(delayed(my_fun)(i,5) for i in range(num))
#end = time.time()
#print(end-start)
'''
