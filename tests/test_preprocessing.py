import joblib


X_train = joblib.load(
    "data/processed/X_train.joblib"
)

print(X_train.shape)