"""
Trains the same Linear Regression model as the original linear_regression.py,
then saves the fitted model and the exact one-hot-encoded column order to
disk so the Vercel API function can reuse it without retraining on every
request (serverless functions should stay fast and stateless).

Run this once locally before deploying:
    python train_model.py
"""

import json
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

df = pd.read_csv("Housing.csv")

# Same encoding as the original script
df_encoded = pd.get_dummies(df, drop_first=True)

X = df_encoded.drop("price", axis=1)
y = df_encoded["price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# Save the model itself
joblib.dump(model, "api/model.pkl")

# Save the exact column order the model expects, so the API can build a
# matching one-hot-encoded row from raw form input at request time.
with open("api/columns.json", "w") as f:
    json.dump(list(X.columns), f)

print("\nSaved api/model.pkl and api/columns.json")
