import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN


def main():
    data = []
    for i in range(3):
        centreX = random.random()*5
        centreY = random.random()*5
        for j in range(30):
            data.append([random.gauss(centreX,0.5),random.gauss(centreY,0.5)])
    data = np.array(data)

    db = DBSCAN(eps=1, min_samples=3).fit(data)
    
    labels = db.labels_

    # plt.scatter(data[:,0],data[:,1],c=labels)
    # plt.show()
    print(labels)

if __name__ == "__main__":
    main()