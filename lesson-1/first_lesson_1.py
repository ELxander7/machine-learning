import pandas as pd 

# def main():
#     pass

def main():
    dataset = pd.read_csv("train.csv")
    print(dataset)

    print(dataset.shape) 
    # количество строк

    print(dataset.shape[0])

    print(dataset.Age) столбцы
    print(dataset["Age"]) //универсальнее

    print(dataset.loc[0]) 
    print(dataset.iloc[0])

    print(dataset.loc[0:2])

    print(dataset.loc[dataset.Age>18])

    df = pd.DataFrame()
    df["Name"] = ["Jack", "John", "Jeniffer", "Jane"]
    df["Age"] = [20, 19, 17, 15]
    df.index = [2, 3, 1, 5]
    print(df)
    print(df.iloc[2])
    print(df.loc[2:1]) выведет jake john jeniffer 
    print(df.iloc[2:"q"])

    df.index = [2, range(1,5), "q", 50]
    print(df.loc[2:"q"])

    задание, все кто старше 18
    задание, все кто старше 18 и выжил
    print(dataset[dataset.Age>18][dataset.Survived == 1])

if __name__ == '__main__':
    main()

