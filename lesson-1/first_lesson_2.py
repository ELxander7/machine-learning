import pandas as pd

def main():
    dataset = pd.read_csv("D:\D Telegram\train.csv")
    print(dataset[(dataset.Age>18) & (dataset.Survived == 1)])
#корректный вариант выше


#графики
import matplotlib.pyplot as plt

def main():
    dataset = pd.read_csv()
    plt.plot(range(0, dataset.shape[0]), dataset.Age)
    plt.scatter(range(0, dataset.shape[0]), dataset.Age)
    plt.bar(range(0, dataset.shape[0]), dataset.Age)
    plt.show


два стобца, количество женщин количество мужчин
def main():
    dataset = pd.read_csv("D:\D Telegram\train.csv")
    dataset[dataset.Sex].value_counts().plot(kind = "bar")
    dataset.groupby(["Sex"]).value_counts().plot(kind="bar")
    dataset.groupby(["Sex"])["Sex"].value_counts().plot(kind="bar")
    dataset.groupby(["Sex"])["Survived"].value_counts().plot(kind="bar")
    dataset.groupby(["Sex"])["Survived"].value_counts().unstack().plot(kind="bar")

    plt.show()

def main():
    dataset = pd.read_csv("train.csv")
    print(dataset[(dataset.Sex == "male") & (dataset.Pclass ==3)]["Survived"].mean())
    mean() среднее арифметическое

    sm = dataset[(dataset.Sex == "male") & (dataset.Pclass ==3) & dataset["Survived"] == 1].shape