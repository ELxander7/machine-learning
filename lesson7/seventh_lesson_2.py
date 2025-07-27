from sklearn.linear_model import LogisticRegression
model = LogisticRegression()

model.fit(x_training_data, y_training_data)
predictions = model.predict(x_test_data)
