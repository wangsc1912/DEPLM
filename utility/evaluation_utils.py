import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
from sklearn.metrics.pairwise import pairwise_distances

import numpy as np
import torch
import pandas as pd
from datetime import datetime


'''Performance Evaluation'''
def similarity(x: np.array, y: np.array, metric='cosine'):
    if metric == 'cosine':
        # return torch.nn.functional.cosine_similarity(x, y)
        sim = np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))
        return 1 - sim
    elif metric == 'euclidean':
        # return torch.norm(x - y)
        return np.linalg.norm(x - y, axis=1)


def intra_cls_sim(embeddings: np.array, cls_center: np.array, metric: str):
    return np.mean(similarity(embeddings, cls_center, metric))


def cal_similarity(embeddings: np.array, labels: np.array, metric='cosine'):
    class_idx = np.unique(labels)

    # divide embedding according to label
    embedding_cls = []
    for i in class_idx:
        embedding_cls.append(embeddings[labels == i])

    # class cente r
    cls_centers = [np.mean(e, 0) for e in embedding_cls]
    # intra-class similarity
    intra_sim = [intra_cls_sim(e, c, metric) for e, c in zip(embedding_cls, cls_centers)]
    # intra_avg_dist = [1 - s for s in intra_sim]
    # diam = [torch.norm(e - c) for e, c in zip(embedding_cls, cls_centers)]

    # inter_sim = [similarity(c1, c2) for c1 in cls_centers for c2 in cls_centers if c1 != c2]
    # similarity_matrix = torch.stack(intra_sim + inter_sim).reshape(len(class_idx), len(class_idx) - 1)
    similarity_matrix = pairwise_distances(cls_centers, cls_centers, metric=metric)
    for i in range(len(class_idx)):
        similarity_matrix[i, i] = intra_sim[i]
    return similarity_matrix, intra_sim


def instance_similarity(embeddings, labels, metric='cosine'):
    '''a large matrix showing instance-wise similarity'''
    class_idx = np.unique(labels)
    # sort 
    embeddings, labels = zip(*sorted(zip(embeddings, labels), key=lambda x: x[1]))
    # similairity matrix
    try:
        similarity_matrix = pairwise_distances(embeddings, embeddings, metric=metric)
    except AttributeError:
        embeddings = np.stack(embeddings, axis=0)
        similarity_matrix = pairwise_distances(embeddings, embeddings, metric=metric)
    except ValueError:
        print(f'ValueError: pairwise_distances with {metric} metric got an empty sequence')
        similarity_matrix = np.zeros((len(labels), len(labels)))
    return similarity_matrix


'''Plotting'''
def plot_similarity(simi_mat, annot=False, path='', name='similarity_matrix', format='pdf'):
    plt.figure()
    sns.heatmap(simi_mat, annot=annot, cmap='Blues')
    # plt.colorbar()
    # plt.savefig(os.path.join(path, f'{name}.{format}'), dpi=100)
    plt.savefig(os.path.join(path, f'{name}.{format}'), format=format)
    plt.close()


def plot_confusion_matrix(y_true, y_pred, path='', normalize=False, save_data=True):
    conf_mat = confusion_matrix(y_true, y_pred)
    fmt, norm_str = 'd', ''
    if normalize:
        conf_mat = conf_mat.astype('float') / conf_mat.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
        norm_str = '_normalized'
    conf_mat_df = pd.DataFrame(conf_mat, index=range(10), columns=range(10))
    if save_data:
        conf_mat_df.to_excel(os.path.join(path, f'conf_mat{norm_str}.xlsx'))
    plt.figure()
    sns.heatmap(conf_mat_df, annot=True, cmap='Blues', fmt=fmt )
    # plt.colorbar()
    plt.savefig(os.path.join(path, f'confusion_matrix{norm_str}.pdf'))
    plt.close()


def plot_pca(X, y, n_comp=2, path='', save_data=True):
    '''
    :param X: (n_samples, n_features)
    y: (n_samples, )
    '''
    pca = PCA(n_components=n_comp)
    X_pca = pca.fit_transform(X)
    if n_comp == 2:
        plt.figure()
        plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y)
        # plt.savefig(os.path.join(path, 'pca.pdf'))
    elif n_comp == 3:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=y)
        # plt.savefig(os.path.join(path, 'pca.pdf'))
    plt.savefig(os.path.join(path, f'pca_{n_comp}d.pdf'))
    plt.close()
    if save_data:
        np.savez(f'pca_{n_comp}d_data.npz', X_pca=X_pca, y=y)


def plot_tsne(X, y, n_comp=2, path='', save_data=True):
    '''
    :param X: (n_samples, n_features)
    y: (n_samples, )
    '''
    color = []
    tsne = TSNE(n_components=n_comp)
    X_tsne = tsne.fit_transform(X)
    if n_comp == 2:
        plt.figure()
        plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y)
    elif n_comp == 3:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(X_tsne[:, 0], X_tsne[:, 1], X_tsne[:, 2], c=y)
    plt.savefig(os.path.join(path, f'tsne_{n_comp}d.pdf'))
    plt.show()
    plt.close()
    if save_data:
        np.savez(f'tsne_{n_comp}d_data.npz', X_tsne=X_tsne, y=y)
