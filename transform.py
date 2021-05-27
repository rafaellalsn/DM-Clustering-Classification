import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.random_projection import GaussianRandomProjection

def main():
    comp = pd.read_csv('data/transformed.csv')

    X = comp.drop(labels=['ALVO_FINAL'], axis=1)
    y = comp['ALVO_FINAL']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    print X_train.shape
    print X_test.shape

    tf = GaussianRandomProjection(n_components=10)
    X_train = tf.fit_transform(X_train)
    X_test = tf.transform(X_test)

    print X_train.shape
    print X_test.shape

    x_train_pca = pd.DataFrame(X_train)
    x_train_pca.to_csv('data/xtrain.csv')

    x_test_pca = pd.DataFrame(X_test)
    x_test_pca.to_csv('data/xtest.csv')

if __name__ == '__main__':
    main()
