import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from fcmeans import FCM 


def main():
    irises = load_iris()
    data = irises.data
    target = irises.target
    # plt.scatter(data[:,0],data[:,1],c = target)
    # plt.show()
    

    pca = PCA(n_components=3)
    pca.fit(data)
    data_pca = pca.transform(data)

    # plt.scatter(data_pca[:,0],data_pca[:,1],c=target)
    # plt.show()



    kmeans = KMeans(n_clusters=3)
    kmeans.fit(data)
    predict = kmeans.predict(data)
    plt.scatter(data[:,0],data[:,1],c=predict)
    plt.show()

    fcm = FCM(n_clusters=3)
    fcm.fit(data)
    predict = fcm.predict(data)
    plt.scatter(data[:,0],data[:,1],c=predict)
    plt.show()

    





if __name__ == "__main__":
    main()