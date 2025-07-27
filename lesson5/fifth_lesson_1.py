import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


def main():
    data = pd.read_csv('bikes_rent.csv')
    sns.heatmap(data.corr(), annot=True, cmap="coolwarm")
    plt.show()
    #выкидывает высокую зависимость
    data.drop(["season", "atemp", "windspeed(mph)"], axis = 1)
    X, y = data.drop(["cnt"], axis = 1), data["cnt"]

    #X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.6, test_size=0.6)
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    #print(len(X_train)/len(X))
    #print(len(X_test)/len(X))

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    print(lr.score(X_test, y_test))


    predict = lr.predict(X_test)
    A = np.sum((y_test-predict)**2)
    B = np.sum((y_test-y_test.mean())**2)
    print(1-A/B)
    #print(predict)
    #print(y.loc[2])

if __name__ == '__main__':
    main()

