import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, plot_tree

data = pd.read_csv("AmesHousing.csv")
data = data[['SalePrice', 'Lot Area', 'Lot Frontage']]


X = data[['Lot Area', 'Lot Frontage']]
y = data['SalePrice']
X.fillna(X.median(), inplace=True)
y.fillna(y.median(), inplace=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = DecisionTreeRegressor(max_depth=3)
model.fit(X_train, y_train)
print(model.score(X_test, y_test))
plt.figure(figsize=(20,10))
plot_tree(model, feature_names=X.columns, filled=True)
plt.show()
