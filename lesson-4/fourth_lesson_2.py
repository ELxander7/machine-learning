import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn import datasets

def main():
    data = []
    for i in range(3):
        centerX = random.random()*5
        centerY = random.random()*5
        for j in range(30):
            data.append([random.gauss(centerX, 0.5), random.gauss(centerY, 0.5)])
    data = np.array(data)

    # plt.scatter(data[:,0], data[:,1])
    # plt.show()

    #dbscan = DBSCAN(eps = 0.5, min_samples=3)
    dbscan = DBSCAN(eps=5, min_samples=5)
    dbscan.fit(data)
    predict = dbscan.labels_
    # plt.scatter(data[:,0], data[:,1], c = predict)
    # plt.show()
    print(predict)

if __name__ == '__main__':
    main()