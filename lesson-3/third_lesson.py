from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from fcmeans import FCM

def main():
    irises = load_iris()
    data = irises.data
    target = irises.target
    #plt.scatter(data[:,0], data[:,1], c = target)
    #plt.show()

    pca = PCA(n_components=1)
    pca.fit(data)
    data_pca = pca.transform(data)
    print(data_pca)

    plt.scatter(data_pca[:,0], data_pca[:,1],c = target)
    plt.show

    kmeans = KMeans(n,clusters=3)
    kmeans.fit(data_pca)
    predict = kmeans.predict(data_pca)
    plt.scatter(data_pca[:,0], data_pca[:,1], c=predict)
    plt.show()


    fcm = FCM(n_clusters=3)
    fcm.fit(data)
    data_fcm = fcm.predict(data)
    plt.scatter(data_pca[:,0],data_pca[:,1], c = data_fcm)
    plt.show()

if __name__ == '__main__':
    main()


import numpy as np
from PIL import Image

def main():
    path = "3uk4rhu3bkujr4buk34.png"
    image = Image.open(path)
    print(image)
    print(np.array(image))

import librosa