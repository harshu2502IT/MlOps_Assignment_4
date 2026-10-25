import joblib

if __name__ == "__main__":
    model = joblib.load("models/model.joblib")
    sample = [[5.1, 3.5, 1.4, 0.2]]
    print("Prediction:", model.predict(sample)[0])
