# train.py
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import os

os.makedirs("model", exist_ok=True)

# ---------- Generate synthetic dataset ----------
np.random.seed(42)
n = 5000

age = np.random.randint(16, 70, size=n)
gender = np.random.binomial(1, 0.5, size=n)  # 1 male, 0 female
height = np.random.normal(170, 10, size=n).clip(140, 200)
weight = np.random.normal(70, 15, size=n).clip(40, 150)
activity_level = np.random.choice([0,1,2,3,4], size=n, p=[0.2,0.3,0.3,0.15,0.05])

# approximate BMR (Mifflin-St Jeor)
bmr = 10*weight + 6.25*height - 5*age + (5*gender - 161*(1 - gender))
act_mult = np.array([1.2, 1.375, 1.55, 1.725, 1.9])
daily_calories = bmr * act_mult[activity_level]
daily_calories = daily_calories * np.random.normal(1.0, 0.06, size=n)  # noise

df = pd.DataFrame({
    "age": age,
    "gender": gender,
    "height_cm": np.round(height,1),
    "weight_kg": np.round(weight,1),
    "activity_level": activity_level,
    "daily_calories": np.round(daily_calories, 0)
})

print("Sample rows:\n", df.head())

# ---------- Train model ----------
features = ["age","gender","height_cm","weight_kg","activity_level"]
X = df[features]
y = df["daily_calories"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

model = RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# evaluate
preds = model.predict(X_test)
mae = mean_absolute_error(y_test, preds)
r2 = r2_score(y_test, preds)
print(f"MAE: {mae:.2f} kcal; R2: {r2:.3f}")

# save model + metadata
joblib.dump({"model": model, "features": features}, "model/calorie_model.joblib")
df.sample(100).to_csv("model/sample_data.csv", index=False)

print("Saved model to model/calorie_model.joblib")