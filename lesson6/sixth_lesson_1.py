# import random
#
# import matplotlib.pyplot as plt
# import numpy as np
# import random
# from sklearn.decomposition import PCA
# from sklearn.datasets import make_blobs
#
#
#
# def generate_blobs(n_samples, centers):
#     data = []
#     for i in range(centers):
#         x = random.random
#         y = random.random
#         for j in range(int(n_samples/centers)):
#             data.append([random.gauss(x), random.gauss(y), centers])
#     return np.array(data)
#
# def main():
#     data = make_blobs(n_samples=100, n_features=5, centers=4)
#     print(data)
#
#
# if __name__ == '__main__':
#     main()

import random

import matplotlib.pyplot as plt
import numpy as np
import random
from sklearn.decomposition import PCA
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


def generate_blobs(n_samples, centers):
    data = []
    for i in range(centers):
        x = random.random
        y = random.random
        for j in range(int(n_samples/centers)):
            data.append([random.gauss(x), random.gauss(y), centers])
    return np.array(data)

def main():
    data, y = make_blobs(n_samples=100, n_features=5, centers=4)
    pca = PCA(n_components=2)
    pca.fit(data)
    data = pca.transform(data)

    X_train, X_test, y_train, y_test = train_test_split(data, y)



    knn = KNeighborsClassifier(n_neighbors=10)
    knn.fit(X_train, y_train)
    print(knn.score(X_test, y_train))


    # print(data)


if __name__ == '__main__':
    main()