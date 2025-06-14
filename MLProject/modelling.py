import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd
import os

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    # Data
    dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "student_habits_performance_preprocessed.csv")
    df = pd.read_csv(dataset_path)
    X = df.drop("exam_score", axis=1)
    y = df["exam_score"]
    
    reg = RandomForestRegressor(random_state=42)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    input_example = X_train[0:5]
    with mlflow.start_run():
        model = RandomForestRegressor()
        model.fit(X_train, y_train)
    
        y_pred = model.predict(X_test)
    
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
    
        print("MAE:", mae)
        print("MSE:", mse)
        print("R2 Score:", r2)

        mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        input_example=input_example
        )
        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("MSE", mse)
        mlflow.log_metric("R2 Score", r2)
