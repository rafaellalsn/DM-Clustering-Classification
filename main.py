import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier

def preprocess_data(file_name, separator):
    BASE_DM_TARGET = pd.read_csv(file_name, sep=separator)
    to_transform = ['DRR', 'MICRO_REGIAO', 'REGIAO_DESENVOLVIMENTO', 'CONTADOR_UF',
                    'ARREC_ULTIMOS_12M', 'ARREC_ULTIMOS_6M', 'MDB']

    for col in to_transform:
        values = np.unique(np.array(BASE_DM_TARGET[col]))
        le = LabelEncoder()
        le.fit(values)
        BASE_DM_TARGET[col] = le.transform(BASE_DM_TARGET[col])

    to_eliminate = ['PESSOA_CD', 'MICRO_REGIAO', 'REGIAO_DESENVOLVIMENTO',
                    'CEP1', 'CEP2', 'CEP3', 'CEP4']
    BASE_DM_TARGET.drop(to_eliminate, 1, inplace=True)


    X = np.array(BASE_DM_TARGET.drop(['ALVO_FINAL'], 1))
    y = np.array(BASE_DM_TARGET['ALVO_FINAL'])
    return X, y


def main():
    X, y = preprocess_data(file_name='data/BASE_DM_TARGET.txt', separator='	')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    dim = X.shape[1]
    clf = MLPClassifier(activation='logistic', solver='adam', alpha=1e-5, hidden_layer_sizes=(dim / 2, dim, 2 * dim),
                        random_state=1)
    clf.fit(X_train, y_train)
    print clf.score(X_test, y_test)
    y_pred = clf.predict(X_test)

    cnf_m = confusion_matrix(y_test, y_pred)
    plt.show()

if __name__ == '__main__':
    main()
