# import pandas as pd
# from sklearn.tree import DecisionTreeClassifier, plot_tree, DecisionTreeRegressor
# from sklearn.model_selection import train_test_split
# import matplotlib.pyplot as plt
#
# data = pd.read_csv('AmesHousing.csv')
#
# X = data[["LotArea"]]
# y = data["LotFrontage"]
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
# model = DecisionTreeClassifier(
#     max_depth=3,
#     random_state=42
# )
# model.fit(X_train, y_train)
#
# # Оценка точности
# accuracy = model.score(X_test, y_test)
# print(f"Точность модели: {accuracy:.2f}")
#
# model = DecisionTreeRegressor(max_depth=3, random_state=42)
# model.fit(X_train, y_train)
#
# # Оценка (средняя квадратичная ошибка)
# from sklearn.metrics import mean_squared_error
# y_pred = model.predict(X_test)
# mse = mean_squared_error(y_test, y_pred)
# print(f"MSE: {mse:.2f}")
#
#
# plt.figure(figsize=(12, 8))
# plot_tree(
#     model,
#     feature_names=["feature_column"],
#     class_names=["Class0", "Class1"],
#     filled=True,
#     rounded=True
# )
# plt.show()

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

def main():
    data = pd.read_csv('AmesHousing.csv')
    print(data["Lot Area"].mean())
    data = data.fillna(data.mean())
    X, y = data[["Lot Area", "Lot Frontage"]], data["SalePrice"]
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    #tree = DecisionTreeClassifier
    #tree.fit(X_train, y_train)
    #print(tree.score(X_test, y_test))
    print(X["Lot Area"].mean())

if __name__=='__main__':
    main()
