import mlflow.sklearn
model = mlflow.sklearn.load_model("models:/Logistic - Elastic Net@production")
print(type(model))