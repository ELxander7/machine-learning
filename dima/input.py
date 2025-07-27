import pandas as pd
import matplotlib.pyplot as plt

def main():
    dataset = pd.read_csv('train.csv')
    #print(dataset[(dataset.Age > 18) & (dataset.Survived == 1)])

    #dataset.groupby(["Pclass"])["Survived"].value_counts().unstack().plot(kind="bar")
    print(dataset[(dataset.Sex == "male") & (dataset.Pclass ==3)]["Survived"].mean())

    #plt.plot(range(0,dataset.shape[0]), dataset.Age)
    #plt.show()

    # df = pd.DataFrame()
    # df["Name"] = ["Jack","John","Jeniffer","Jane"]
    # df["Age"] = [20,19,17,15]
    # df.index = [2,3,1,50]
    # print(df.loc[2:1])


if __name__ == "__main__":
    main()