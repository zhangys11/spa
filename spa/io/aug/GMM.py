from sklearn.mixture import GaussianMixture
import numpy as np
import pandas as pd

def expand_dataset(X, y, nobs, verbose = True):
    '''
    Generate equal number of samples for each class.

    nobs : samples to generate per class.
    '''

    # X = pd.DataFrame(X)

    final_data=pd.DataFrame()
    for label in set(y):
        # for i in range(X.shape[1]):
        gmm = GaussianMixture(n_components= max(len(set(y)), 3))  # 设定混合成分数
        gmm.fit(X)  # 对数据进行拟合
        if verbose:
            print('GMM weights for label', label, ":", 
                    # str(np.round(gmm.means_,2)).replace('\r','').replace('\n',''), 
                    # str(np.round(gmm.covariances_,2)).replace('\r','').replace('\n',''), 
                    str(np.round(gmm.weights_, 2)).replace('\r','').replace('\n',''))
        X_new, _ = gmm.sample(nobs)
        batch=pd.DataFrame(X_new)
        batch['label']=[label]*len(batch)
        final_data=pd.concat([final_data,batch],axis=0)

    final_data.reset_index(drop=True, inplace=True)
    return final_data.iloc[:,:-1],final_data.iloc[:,-1]