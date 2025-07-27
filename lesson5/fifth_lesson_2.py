import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA


def main():
    
    predict = lr.predict(X_test)
    A = np.sum((y_test - predict) ** 2)
    B = np.sum((y_test - y_test.mean()) ** 2)
    #print(1 - A / B)

    pca = PCA(n_components=2)
    pca.fit(X)
    X_pca = pca.transform(X)

    ax = plt.figure()
    plt.scatter(X_pca[:, 0], X_pca[:1], y, c = y)
    plt.show()


if __name__ == '__main__':
    main()

