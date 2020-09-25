import pandas as pd
import seaborn as sb
from matplotlib import rcParams
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from scipy.spatial.distance import euclidean
import numpy as np
from matplotlib.lines import Line2D
import matplotlib.animation as animation
#%%
class k_mean():
    def __init__(self, data, K, features):
        self.data = data
        self.feature_names = features
        self.K = K
        self.scaled_names = [feature_name + '_scaled' for feature_name in self.feature_names]
        self.scaled_df = pd.DataFrame(scale(self.data[self.feature_names]), columns=self.scaled_names) 
        self.pca_names = ['pc' + str(i+1) for i in range(len(self.scaled_names))]
        self.pca = PCA()
        self.pca.fit(self.scaled_df)
        self.pca_df = pd.DataFrame(self.pca.transform(self.scaled_df), columns=self.pca_names)
        data_dfs = [self.data, self.scaled_df, self.pca_df]
        self.data_df = pd.concat(data_dfs, axis=1)
        self.initial_centroids = self.data_df.loc[np.random.randint(0,len(self.data_df), self.K)].copy()
        self.centroids = self.initial_centroids.copy()
        self.cent_df = self.data_df[self.scaled_names].apply(euclidean_dis,
                        result_type = 'expand',
                        axis=1,
                        cents=self.centroids[self.scaled_names])
        self.cent_names = ['cluster' + str(i+1) for i in range(K)] + ['min','argmin']
        self.cent_df.columns = self.cent_names
        self.data_cent_df = pd.concat([self.data_df, self.cent_df], axis=1)
        self.columns = self.data_cent_df.columns
    def apply(self):
        self.max_num_iter = 100
        self.num_iter = 0
        self.cluster_changes = []
        self.SSD = []
        for i in range(self.max_num_iter):
            self.num_iter += 1
            self.centroids = self.data_cent_df.groupby('argmin').mean()
            self.data_cent_df[self.cent_names] = self.data_cent_df[self.scaled_names].apply(euclidean_dis,
                                    result_type = 'expand',
                                    axis=1,
                                    cents=self.centroids[self.scaled_names]).values
            self.cluster_changes.append(self.data_cent_df['argmin'].copy())
            self.SSD.append((self.data_cent_df['min']**2).sum())
            if i > 1:
                if (self.SSD[-2] - self.SSD[-1]) < (self.SSD[-1]*.0001):
                    break
        self.plot_data = {'cluster_change': self.cluster_changes,
             'SSD':self.SSD,
             'num_iter':self.num_iter,
             'K':self.K,
             'data':self.data_cent_df,
             'initial_cent': self.initial_centroids,
             'cent_df': self.cent_df}
        return self.plot_data
    
    
def euclidean_dis(row, cents):
    dist_and_min_index = [euclidean(row, cents.loc[i]) for i in cents.index]
    dist_and_min_index.append(np.array(dist_and_min_index).min() + 1)
    dist_and_min_index.append(np.array(dist_and_min_index).argmin())
    return dist_and_min_index
def scatterplot(data, x, y, col):
    sb.set(style = "whitegrid")
    plt.figure(figsize = (8, 6))
    plt.scatter(data[x], data[y], alpha = 1,
                c = data[col], s= 150, cmap = 'Spectral', edgecolors = 'grey')
    rcParams['axes.titlepad'] = 25 
    plt.title(x + ' vs. ' + y, fontsize = 25, fontweight = 'bold')
    plt.xlabel('$\it{' + x.replace(' ', '\ ') + '}$', fontsize = 20)
    plt.ylabel('$\it{' + y.replace(' ', '\ ') + '}$', fontsize = 20)
    centroids = data.groupby('argmin').mean()
    plt.scatter(centroids[x], centroids[y],
                    marker='*', s=800, c='gold', edgecolors = 'black')
        
class SubplotAnimation(animation.TimedAnimation):
    def __init__(self, data):
        self.data = data
        self.colors = ['black','red','blue','green','grey','purple','brown','gold','yellow']
        fig = plt.figure(figsize=(8,9))
        ax1 = fig.add_subplot(2, 1, 1)
        ax2 = fig.add_subplot(2, 1, 2)

        ax1.set_xlabel('pc1')
        ax1.set_ylabel('pc2')
        self.lines = []
        for k in range(data['K']):
            self.lines.append(Line2D(
                [], [], color=self.colors[k], marker='o', ms = 10, markeredgecolor='white', lw = 0, alpha = .8))
            self.lines.append(Line2D(
                [], [], color=self.colors[k], marker='*', mew = 1, ms = 20, markeredgecolor='yellow', lw = 0))
            ax1.add_line(self.lines[-2])
            ax1.add_line(self.lines[-1])
        ax1.set_xlim(self.data['data']['pc1'].min() -1, self.data['data']['pc1'].max()+1)
        ax1.set_ylim(self.data['data']['pc2'].min() -1, self.data['data']['pc2'].max()+1)
        
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('SSD')
        self.lines.append(Line2D([], [], color='black'))
        ax2.add_line(self.lines[-1])
        ax2.set_xlim(0, self.data['num_iter'])
        ax2.set_ylim(min(self.data['SSD'])*0.99, max(self.data['SSD'])*1.001)

        animation.TimedAnimation.__init__(self, fig, interval=500, blit=True)

    def _draw_frame(self, framedata):
        i = framedata
        for k in range(self.data['K']):
            self.lines[2*k].set_data(self.data['data']['pc1'][self.data['cluster_change'][i] == k],
                                   self.data['data']['pc2'][self.data['cluster_change'][i] == k])
            self.lines[2*k+1].set_data(self.data['data']['pc1'][self.data['cluster_change'][i] == k].mean(),
                                       self.data['data']['pc2'][self.data['cluster_change'][i] == k].mean())
        self.lines[-1].set_data(range(i+1), self.data['SSD'][:i+1])

        self._drawn_artists = self.lines

    def new_frame_seq(self):
        return iter(range(self.data['num_iter']))

    def _init_draw(self):
        for k in range(self.data['K']):
            self.lines[2*k].set_data(self.data['data']['pc1'][self.data['cent_df']['argmin'] == k],
                                     self.data['data']['pc2'][self.data['cent_df']['argmin'] == k])
            self.lines[2*k+1].set_data(self.data['data']['pc1'][self.data['cent_df'].index[k]],
                                       self.data['data']['pc2'][self.data['cent_df'].index[k]])
        self.lines[-1].set_data([], [])      