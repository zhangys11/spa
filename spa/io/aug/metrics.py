# TODO: need test

import scipy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, r2_score, f1_score, precision_score, recall_score, accuracy_score
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
from ...cla import run_multiclass_clfs

# from . import SMOTE, Gaussian, GMM, KDE, VAE, TVAE, MDN, GAN, DCGAN, ctGAN

# def JS_divergence(p,q):
#     M=(p+q)/2
#     return 0.5*scipy.stats.entropy(p, M, base=2)+0.5*scipy.stats.entropy(q, M, base=2)

def calculate_pairwise_metrics(px, py):

    KL = scipy.stats.entropy(px, py)
    JS = scipy.spatial.distance.jensenshannon(px, py)  # JS_divergence(px, py)
    pearson = px.corr(py, method="pearson")
    spearman = px.corr(py, method="spearman")
    kendall = px.corr(py, method="kendall")
    similarity = 1 - scipy.spatial.distance.cosine(px, py)
    r2 = r2_score(px, py)
    mse = mean_squared_error(px, py)

    ### SSIM and PSNR are orginally used on images. We tile/stack the 1D signal to 2D array and then apply the two metrics.
    n = 20  # len(px)
    SSIM = structural_similarity(np.tile(px, (n, 1)).astype(int), np.tile(py, (n, 1)).astype(int), data_range=max(px))
    # PSNR=peak_signal_noise_ratio(np.tile(px, (n,1)),np.tile(py, (n,1)))

    return KL, JS, pearson, spearman, kendall, similarity, r2, mse, SSIM  # , PSNR


def calculate_similarity_metrics(dataset_raw, dataset_syn):

    KL = []
    JS = []
    pearson = []
    spearman = []
    kendall = []
    similarity = []
    r2 = []
    mse = []
    SSIM = []
    PSNR = []

    for i in range(dataset_raw.shape[0]):
        for j in range(dataset_syn.shape[0]):
            px = dataset_raw.iloc[i, :]
            py = dataset_syn.iloc[j, :]
            result = calculate_pairwise_metrics(px, py)

            KL.append(result[0])
            JS.append(result[1])
            pearson.append(result[2])
            spearman.append(result[3])
            kendall.append(result[4])
            similarity.append(result[5])
            r2.append(result[6])
            mse.append(result[7])
            SSIM.append(result[8])
            # PSNR.append(result[9])

    KL_mean = np.mean(KL)
    JS_mean = np.mean(JS)
    pearson_mean = np.mean(pearson)
    spearman_mean = np.mean(spearman)
    kendall_mean = np.mean(kendall)
    similarity_mean = np.mean(similarity)
    r2_mean = np.mean(r2)
    mse_mean = np.mean(mse)
    SSIM_mean = np.mean(SSIM)
    # PSNR_mean = np.mean(PSNR)

    return [KL_mean, JS_mean, pearson_mean, spearman_mean, kendall_mean, similarity_mean, r2_mean, mse_mean,
            SSIM_mean]  # , PSNR_mean]

def evaluate_fidelity(X_synth, y_synth, model_names, baseline, verbose=True):  # df_test[df_test['Label']==0]
    '''
    Evaluate how each synthed dataset resembles the original data

    Parameters
    ----------
    X_synth : list of synth datasets
    y_synth : list of corresponding y values
    model_names : synth / data augmentation methods
    baseline : a pandas dataframe of the original dataset
    '''
    
    data_org = baseline
    data_org.drop('label', axis=1, inplace=True)
    data_org[data_org < 0.0001] = 1
    data_org = data_org.reset_index(drop=True)

    if verbose:
        plt.figure(figsize=(12, 4))
        plt.plot(data_org.columns, data_org.mean(), c='r', label='original data')
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.legend(prop={'size': 14})
        plt.show()

    result = pd.DataFrame()
    for (i, j, k) in zip(X_synth, y_synth, model_names):
        x_sam = pd.DataFrame(i)
        y_sam = pd.DataFrame(j)
        y_sam.columns = ['label']
        data_synth = pd.concat([x_sam, y_sam], axis=1)
        data_synth = data_synth.iloc[:, :-1]
        data_synth.columns = data_org.columns

        if verbose:        
            plt.figure(figsize=(12, 4))
            plt.plot(data_synth.columns, data_synth.mean(), c='r', label=k)
            plt.xticks(fontsize=16)
            plt.yticks(fontsize=16)
            plt.legend(prop={'size': 14})
            plt.show()

        data_synth = data_synth.reset_index(drop=True)
        data_synth[data_synth < 0.0001] = 1

        ### 评估生成数据
        result[k] = calculate_similarity_metrics(data_org, data_synth)
    
    result.index = ['KL', 'JS', 'pearson', 'spearman', 'kendall', 'similarity', 'r2', 'mse', 'SSIM']  # ,'PSNR']
    return result


def evaluate_classification(X_synth, y_synth, model_names, verbose=False):

    dic = {}
    for (i, j, k) in zip(X_synth, y_synth, model_names):
        result,_ = run_multiclass_clfs(i, j, show=verbose)
        df = pd.DataFrame()
        for w in range(len(result)):
            df_tem = pd.DataFrame([result[w]])
            df_tem.columns = [col.split('(')[0] for col in df_tem.columns]
            df = pd.concat([df, df_tem], axis=0)
        df.index = ['train_accs', 'test_accs', 'train_precisions', 'test_precisions', \
                    'train_recalls', 'test_recalls', 'train_f1s', 'test_f1s']
        dic[k] = df
    return dic

