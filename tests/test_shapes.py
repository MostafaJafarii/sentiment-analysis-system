import joblib

X_train = joblib.load(
    "data/processed/X_train.joblib"
)

X_validation = joblib.load(
    "data/processed/X_validation.joblib"
)

X_test = joblib.load(
    "data/processed/X_test.joblib"
)

print("Train Shape:")
print(X_train.shape)

print()

print("Validation Shape:")
print(X_validation.shape)

print()

print("Test Shape:")
print(X_test.shape)