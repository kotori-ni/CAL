from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import json
from sklearn.cluster import DBSCAN
import numpy as np
from .config import default_eps, default_min_examples
import os

def readjsonl(filename):
    with open(filename, encoding='utf8') as f:
        datas = f.readlines()
    datas_tmp=[]
    for data in datas:
        data = json.loads(data)
        datas_tmp.append(data)
    return datas_tmp

class ClusterEngine:
    def __init__(self, dataset_name, model_name):
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.dir = 'data/' + dataset_name + '/' + model_name+ '/'
        self.orig_datas = readjsonl(self.dir + 'filtered_examples_relative.jsonl')

    def cluster(self, eps=None, min_examples=None, ratio=0.04, write_file=True, n_components=2, alpha = 0.08):
        if eps is None:
            if self.model_name in default_eps and self.dataset_name in default_eps[self.model_name]:
                eps = default_eps[self.model_name][self.dataset_name]
            else:
                raise('no default eps for {} {}'.format(self.model_name,self.dataset_name))

        if min_examples is None:
            if self.model_name in default_min_examples and self.dataset_name in default_min_examples[self.model_name]:
                min_examples = default_min_examples[self.model_name][self.dataset_name]
            else:
                raise('no default min_examples for {} {}'.format(self.model_name,self.dataset_name))

        datas, negas, posis = [], [], []
        num = 0.0
        for i in range(len( self.orig_datas)):
            posi_data = self.orig_datas[i][0]
            nega_data = self.orig_datas[i][1]
            result = []
            for j in range(len(posi_data["representations"])):
                if (posi_data["representations"][j] + nega_data["representations"][j]) == 0 or abs(posi_data["representations"][j] - nega_data["representations"][j]) / (
                    posi_data["representations"][j] + nega_data["representations"][j]) < ratio:
                    result.append((posi_data["representations"][j] + nega_data["representations"][j]) / 2)
                    num += 1
                else:
                    result.append(0)
            datas.append(result)
            posis.append((posi_data['content'], posi_data['gold'], posi_data['pred']))
            negas.append((nega_data['content'], nega_data['gold'], nega_data['pred']))

        datas = np.array(datas)
        pca_tool = PCA()
        pca = pca_tool.fit_transform(datas)[:,:n_components]
        del pca_tool
        del datas
        y_pred = DBSCAN(eps=eps, min_samples=min_examples).fit_predict(pca)
        colors = ['tomato', 'cyan', 'royalblue', 'yellow', 'purple', 'orange', 'darkgrey', 'lightpink', 'maroon', 'green']
        notinanycluster_num = 0
        d = []
        results, results_neg = {}, {}
        for i in range(10000):
            if i in y_pred:
                d.append(i)
                results[i] = []
                results_neg[i] = set()
            else:
                break

        pca1, pca2, y_pred1 = [], [], []
        for i in range(len(y_pred)):
            if y_pred[i] != -1:
                results[y_pred[i]].append((posis[i], negas[i]))
                results_neg[y_pred[i]].add(negas[i])
                pca1.append(pca[i][0])
                pca2.append(pca[i][1])
                y_pred1.append(y_pred[i])
            else:
                notinanycluster_num += 1

        for key in results_neg:
            results_neg[key] = list(results_neg[key])
        print('number of cluster categories:', len(d))

        if write_file is True:
            fig, axes = plt.subplots(1, 1, figsize=(12, 6))
            axes.scatter(pca[:, 0], pca[:, 1], edgecolor='none')
            if not os.path.exists(self.dir + 'PCA_relative'):
                os.makedirs(self.dir + 'PCA_relative')
            plt.savefig(self.dir + 'PCA_relative/origin.jpg')
            fig, axes = plt.subplots(1, 1, figsize=(12, 6))
            if len(d) + 1 > len(colors):
                axes.scatter(pca1, pca2, c=y_pred1, cmap='viridis', alpha=alpha, edgecolor='none')
            else:
                colors1 = [colors[y_pred[i]] if y_pred[i] != -1 else 'white' for i in range(len(y_pred))]
                axes.scatter(pca[:, 0], pca[:, 1], c=colors1, alpha=alpha, edgecolor='none')
            axes.set_title('PCA')
            plt.savefig(self.dir + 'PCA_relative/' + str(eps) + '_' + str(min_examples) + '.jpg')
            dir = self.dir + 'bias_clustered_relative.jsonl'
            dir1 = self.dir + 'bias_clustered_relative_neg.jsonl'
            with open(dir1, 'w', encoding='utf-8') as f:
                json.dump(results_neg, f, ensure_ascii=False)
                f.write('\n')
            with open(dir, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False)
                f.write('\n')
