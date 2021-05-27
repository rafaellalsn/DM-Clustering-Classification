#!/usr/bin/env python
# coding: utf-8

# ## ANÁLISE ESTATÍSTICA DOS DADOS - MINERAÇÃO DE DADOS 2017.1

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df = pd.read_csv('data/transformed.csv', index_col=0)


# In[3]:


df.head()


# In[4]:


len(df.columns)


# ## DEALING WITH UNBALANCED DATA

# In[5]:


df['ALVO_FINAL'].unique()


# In[6]:


len(df[df['ALVO_FINAL'] == 0])


# In[7]:


len(df[df['ALVO_FINAL'] == 1])


# In[8]:


n_ones = len(df[df['ALVO_FINAL'] == 1])
df_ones = df[df['ALVO_FINAL'] == 1]


# In[9]:


# CHOOSE n_ones RANDOM FROM TARGET == 0
instances = np.random.permutation(n_ones)


# In[10]:


df_zeros = df[df['ALVO_FINAL'] == 0].iloc[instances, :]


# In[11]:


df_balanced = pd.concat(objs=[df_ones, df_zeros], axis=0)
df_balanced = df_balanced.reset_index()
df_balanced.drop(labels='index', axis=1, inplace=True)


# In[12]:


df_balanced.head()


# In[13]:


len(df_balanced[df_balanced['ALVO_FINAL'] == 0])


# In[14]:


len(df_balanced[df_balanced['ALVO_FINAL'] == 1])


# ## MEASURING FEATURE IMPORTANCE WITH RANDO FOREST

# In[15]:


X = df_balanced.drop(labels=['ALVO_FINAL'], axis=1).values
y = df_balanced['ALVO_FINAL'].values


# In[16]:


from sklearn.ensemble import RandomForestClassifier

forest = RandomForestClassifier(n_estimators=1000, n_jobs=-1, verbose=100)
forest.fit(X, y)
importances = forest.feature_importances_


# In[17]:


features = list(df.columns)
features.remove('ALVO_FINAL')


# In[52]:


indices = np.argsort(importances)[::-1]
sorted_features = []
for f in range(X.shape[1] - 1):
    sorted_features.append(features[indices[f]])
    sorted_importances.append(importances[indices[f]])
    print("%2d) %-*s %f" % (f + 1, 30, features[indices[f]], importances[indices[f]]))
    
sorted_features = np.array(sorted_features)


# In[56]:


sns.reset_orig()
plt.figure(figsize=(14,7))
plt.title('Feature Importances')
plt.bar(range(X.shape[1]), importances[indices], color='lightblue', align='center')
plt.xticks(range(X.shape[1]), [string.decode("latin-1") for string in sorted_features], rotation=90)
plt.xlim([-1, X.shape[1]])
sns.despine(bottom=True, left=True)
plt.tight_layout()


# In[59]:


sorted_features


# In[72]:


indexes = np.arange(0, 10, 1)
Att = []
for idx in indexes:
    Att.append(sorted_features[idx])
    print sorted_features[idx], importances[indices[idx]]


# In[73]:


Att


# In[74]:


df_balanced[Att].head()


# In[75]:


df_balanced[Att].to_csv('balanced.csv', index=False)


# In[76]:


pd.read_csv('balanced.csv').head()


# In[77]:


from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier


# In[78]:


X = df_balanced[Att].values
y = df_balanced['ALVO_FINAL'].values


# In[79]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


# In[80]:


std = StandardScaler()
X_train = std.fit_transform(X_train)
X_test = std.transform(X_test)


# ## NEURAL NET REVISITED

# In[81]:


dim = X.shape[1]
clf = MLPClassifier(activation='logistic', solver='adam', alpha=1e-5, hidden_layer_sizes=(dim, dim/2, 2*dim),
                    max_iter=500)
clf.fit(X_train, y_train)
print 'SCORE: {}'.format(clf.score(X_test, y_test))

preds = clf.predict(X_test)
cfm = confusion_matrix(y_test, preds)
sns.heatmap(cfm , annot=True, cbar=False, fmt='g', cmap='Blues')
plt.title('REDE NEURAL - MATRIX DE CONFUSAO')

print classification_report(y_test, preds)


# ## RANDOM FORESTS REVISITED

# In[82]:


clf = RandomForestClassifier(n_estimators=100, max_depth=None, min_samples_split=2)
clf.fit(X_train, y_train)
print 'SCORE: {}'.format(clf.score(X_test, y_test))

preds = clf.predict(X_test)
cfm = confusion_matrix(y_test, preds)
sns.heatmap(cfm , annot=True, cbar=False, fmt='g', cmap='Blues')
plt.title('RANDOM FOREST - MATRIX DE CONFUSAO')

print classification_report(y_test, preds)


# ## SUPPORT VECTOR MACHINES

# In[83]:


clf = SVC(kernel='rbf', C=10, gamma=0.001)
clf.fit(X_train, y_train)
print 'SCORE: {}'.format(clf.score(X_test, y_test))

preds = clf.predict(X_test)
cfm = confusion_matrix(y_test, preds)
sns.heatmap(cfm , annot=True, cbar=False, fmt='g', cmap='Blues')
plt.title('SUPPORT VECTOR MACHINE - MATRIX DE CONFUSAO')

print classification_report(y_test, preds)


# ## CLUSTERING DATA WITH MEAN SHIFT

# In[124]:


from sklearn.cluster import MeanShift
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import SpectralClustering
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import AffinityPropagation


# In[104]:


df_balanced[Att][df_balanced['ALVO_FINAL'] == 1].head()


# In[105]:


to_cluster = df_balanced[Att][df_balanced['ALVO_FINAL'] == 1]


# In[106]:


to_cluster.head()


# In[107]:


data = to_cluster.values[:1000, :]


# In[108]:


data


# In[109]:


clf = MiniBatchKMeans(n_clusters=2)


# In[110]:


clf.fit(data)


# In[111]:


clf.cluster_centers_


# In[112]:


def intra_cluster_statistic(data, centroids):
       clusters = {}
       for k in centroids:
           clusters[k] = []

       for xi in data:
           dist = [np.linalg.norm(xi - centroids[c]) for c in centroids]
           class_ = dist.index(min(dist))
           clusters[class_].append(xi)

       inter_cluster_sum = 0.0
       non_empty_clusters = 0
       for c in centroids:
           intra_sum = 0.0
           if len(clusters[c]) > 0:
               for point in clusters[c]:
                   intra_sum += np.linalg.norm(point - centroids[c])
               intra_sum = intra_sum / len(clusters[c])
               non_empty_clusters += 1
           inter_cluster_sum += intra_sum
       inter_cluster_sum = inter_cluster_sum / non_empty_clusters
       return inter_cluster_sum


# In[113]:


len(clf.cluster_centers_)


# In[114]:


def to_dict_centroids(centers):
    centroids = {}
    for i in range(len(centers)):
        centroids[i] = centers[i]
    return centroids


# In[115]:


centroids = to_dict_centroids(clf.cluster_centers_)


# In[116]:


intra_cluster_statistic(data, centroids)


# In[117]:


avg = []
std = []
for n_clusters in range(2, 10):
    intra_statistic = []
    for i in range(30):
        clf = MiniBatchKMeans(n_clusters=n_clusters)
        clf.fit(data)
        centroids = to_dict_centroids(clf.cluster_centers_)
        intra_statistic.append(intra_cluster_statistic(data, centroids))
    avg.append(np.mean(intra_statistic))
    std.append(np.std(intra_statistic))


# In[40]:


plt.plot(range(2, 10), intra_statistic)


# ## K-MEANS

# In[ ]:


avg = []
std = []
for n_clusters in range(2, 10):
    intra_statistic = []
    for i in range(30):
        clf = MiniBatchKMeans(n_clusters=n_clusters)
        clf.fit(data)
        centroids = to_dict_centroids(clf.cluster_centers_)
        intra_statistic.append(intra_cluster_statistic(data, centroids))
    avg.append(np.mean(intra_statistic))
    std.append(np.std(intra_statistic))


# In[119]:


sns.set_style('darkgrid')
plt.errorbar(x=range(2, 10), y=avg, yerr=std, lw=1)


# ## FUZZ-C Means

# In[126]:


import numpy as np

class FCMeans(object):
    def __init__(self, n_clusters=3, n_iter=300, fuzzy_c=2, tolerance=0.001):
        self.n_clusters = n_clusters
        self.n_iter = n_iter
        self.fuzzy_c = fuzzy_c
        self.tolerance = tolerance
        self.run = False

    def fit(self, x):
        self.run = True
        self.centroids = {}

        if len(x.shape) < 1:
            raise Exception("DataException: Dataset must contain more examples" +
                            "than the required number of clusters!")

        for k in range(self.n_clusters):
            self.centroids[k] = np.random.random(x.shape[1])

        self.degree_of_membership = np.zeros((x.shape[0], self.n_clusters))
        for idx_ in self.centroids:
            for idx, xi in enumerate(x):
                updated_degree_of_membership = 0.0
                norm = np.linalg.norm(xi - self.centroids[idx_])
                all_norms = [norm / np.linalg.norm(xi - self.centroids[c]) for c in self.centroids]
                all_norms = np.power(all_norms, 2 / (self.fuzzy_c - 1))
                updated_degree_of_membership = 1 / sum(all_norms)
                self.degree_of_membership[idx][idx_] = updated_degree_of_membership

        for iteration in range(self.n_iter):
            powers = np.power(self.degree_of_membership, self.fuzzy_c)
            for idx_ in self.centroids:
                centroid = []
                sum_membeship = 0
                for idx, xi in enumerate(x):
                    centroid.append(powers[idx][idx_] * np.array(xi))
                    sum_membeship += powers[idx][idx_]
                centroid = np.sum(centroid, axis=0)
                centroid = centroid / sum_membeship
                self.centroids[idx_] = centroid

            max_episilon = 0.0
            for idx_ in self.centroids:
                for idx, xi in enumerate(x):
                    updated_degree_of_membership = 0.0
                    norm = np.linalg.norm(xi - self.centroids[idx_])
                    all_norms = [norm / np.linalg.norm(xi - self.centroids[c]) for c in self.centroids]
                    all_norms = np.power(all_norms, 2 / (self.fuzzy_c - 1))
                    updated_degree_of_membership = 1 / sum(all_norms)
                    diff = updated_degree_of_membership - self.degree_of_membership[idx][idx_]
                    self.degree_of_membership[idx][idx_] = updated_degree_of_membership

                    if diff > max_episilon:
                        max_episilon = diff
            if max_episilon <= self.tolerance:
                break

    def predict(self, x):
        if self.run:
            if len(x.shape) > 1:
                class_ = []
                for c in self.centroids:
                    class_.append(np.sum((x - self.centroids[c]) ** 2, axis=1))
                return np.argmin(np.array(class_).T, axis=1)
            else:
                dist = [np.linalg.norm(x - self.centroids[c]) for c in self.centroids]
                class_ = dist.index(min(dist))
                return class_
        else:
            raise Exception("NonTrainedModelException: You must fit data first!")


# In[128]:


avg = []
std = []
for n_clusters in range(2, 10):
    intra_statistic = []
    for i in range(30):
        clf = FCMeans(n_clusters=n_clusters)
        clf.fit(data)
        centroids = to_dict_centroids(clf.centroids)
        intra_statistic.append(intra_cluster_statistic(data, centroids))
    avg.append(np.mean(intra_statistic))
    std.append(np.std(intra_statistic))


# In[129]:


sns.set_style('darkgrid')
plt.errorbar(x=range(2, 10), y=avg, yerr=std, lw=1)


# ## Majority Vote Classifier - Ensemble Method

# In[84]:


import numpy as np
import operator

from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.base import clone
from sklearn.externals import six
from sklearn.pipeline import _name_estimators
from sklearn.preprocessing import LabelEncoder

class MajorityVoteClassifier(BaseEstimator, ClassifierMixin):
    
    def __init__(self, classifiers, votes='classlabel', weights=None):
        self.classifiers = classifiers
        self.named_classifiers = {key:value for key, value in _name_estimators(classifiers)}
        self.votes = votes
        self.weights = weights
        
    def fit(self, X, y):
        if self.votes not in ('probability', 'classlabel'):
            raise ValueError("vote must be 'probability' or 'classlabel'; got (vote=%r)" % self.vote)
        
        if self.weights and len(self.weights) != len(self.classifiers):
            raise ValueError('Number of classifiers and weights must be equal ; got %d weights, %d classifiers'
                             % (len(self.weights), len(self.classifiers)))
            
        self.lablenc_ = LabelEncoder()
        self.lablenc_.fit(y)
        self.classes_ = self.lablenc_.classes_
        self.classifiers_ = []
        for clf in self.classifiers:
            fitted_clf = clone(clf).fit(X, self.lablenc_.transform(y))
            self.classifiers_.append(fitted_clf)
        return self
    
    def predict(self, X):
        if self.votes == 'probability':
            maj_vote = np.argmax(self.predict_proba(X), axis=1)
        else:
            preds = np.asarray([clf.predict(X) for clf in self.classifiers_]).T
            maj_vote = np.apply_along_axis(lambda x : np.argmax(np.bincount(x, weights=self.weights)),
                                           axis=1, arr=preds)
        maj_vote = self.lablenc_.inverse_transform(maj_vote)
        return maj_vote
    
    def predict_proba(self, X):
        probas = np.asarray([clf.predict_proba(X) for clf in self.classifiers_])
        avg_proba = np.average(probas, axis=0, weights=self.weights)
        return avg_proba
    
    def get_params(self, deep=True):
        if not deep:
            return super(MajorityVoteClassifier, self).get_params(deep=False)
        else:
            out = self.named_classifiers.copy()
            for name, step in six.iteritems(self.named_classifiers):
                for key, value in six.iteritems(step.get_params(deep=True)):
                    out['%s__%s' % (name, key)] = value
            return out


# In[88]:


clf = MajorityVoteClassifier(classifiers=[clf1, clf2, clf3])
clf.fit(X_train, y_train)
print 'SCORE: {}'.format(clf.score(X_test, y_test))

preds = clf.predict(X_test)
cfm = confusion_matrix(y_test, preds)
sns.heatmap(cfm , annot=True, cbar=False, fmt='g', cmap='Blues')
plt.title('ENSEMBLE CLASSIFIER - MATRIX DE CONFUSAO')

print classification_report(y_test, preds)


# ## Combinando as diferentes tecnicas para Ensemble Classification

# In[85]:


clf1 = MLPClassifier(activation='logistic', solver='adam', alpha=1e-5, hidden_layer_sizes=(dim, dim/2, 2*dim),
                    max_iter=500)
clf2 = SVC(kernel='rbf', C=10, gamma=0.001, probability=True)
clf3 = RandomForestClassifier(n_estimators=100, max_depth=None, min_samples_split=2)
clfM = MajorityVoteClassifier(classifiers=[clf1, clf2, clf3])
clf_labels = ['MLP', 'SVM', 'Random Forests', 'Majority Voting']
clfs = [clf1, clf2, clf3, clfM]


# In[42]:


from sklearn.metrics import auc
from sklearn.metrics import roc_curve

colors = ['black', 'orange', 'blue', 'green']
linestyles = [':', '--', '-.', '-']

plt.figure(figsize=(12,8))
for clf, label, clr, ls in zip(clfs, clf_labels, colors, linestyles):
    preds = clf.fit(X_train, y_train).predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_true=y_test, y_score=preds)
    roc_auc = auc(x=fpr, y=tpr)
    plt.plot(fpr, tpr, color=clr, linestyle=ls, label='%s (auc = %0.2f)' % (label, roc_auc))
plt.legend(loc='lower right')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', linewidth=2)
plt.xlim([-0.1, 1.1])
plt.ylim([-0.1, 1.1])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')


# In[ ]:




