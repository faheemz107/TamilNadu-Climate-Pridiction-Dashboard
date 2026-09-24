import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. LOAD DATA
df = pd.read_csv("TNweather.csv")
print("Original shape:", df.shape)

# 2. CLEAN DATA
df["time"] = pd.to_datetime(df["time"])
df = df[df["time"].dt.year.isin([2024, 2025])].copy()

if "Unnamed: 0" in df.columns:
    df.drop(columns=["Unnamed: 0"], inplace=True)

df["year"] = df["time"].dt.year
df["month"] = df["time"].dt.month

print("2024-2025 shape:", df.shape)
print("Missing values:", df.isnull().sum().sum())
print("Duplicate rows:", df.duplicated().sum())

# 3. BASIC EDA
df.groupby("city")["temperature_2m"].mean().sort_values().plot(
    kind="bar", figsize=(14, 6)
)
plt.title("Average Temperature by District")
plt.xlabel("District")
plt.ylabel("Temperature (°C)")
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig("district_temperature.png")
plt.close()

monthly = df.groupby(["year", "month"])["temperature_2m"].mean().reset_index()

plt.figure(figsize=(12, 6))
for year in [2024, 2025]:
    d = monthly[monthly["year"] == year]
    plt.plot(d["month"], d["temperature_2m"], marker="o", label=str(year))

plt.title("Monthly Temperature Trend")
plt.xlabel("Month")
plt.ylabel("Temperature (°C)")
plt.legend()
plt.tight_layout()
plt.savefig("monthly_temperature.png")
plt.close()

# 4. MACHINE LEARNING
features = [
    "relative_humidity_2m",
    "dew_point_2m",
    "rain",
    "surface_pressure",
    "cloud_cover",
    "cloud_cover_low",
    "wind_speed_10m",
    "wind_direction_10m"
]
target = "temperature_2m"

train = df[df["year"] == 2024]
test = df[df["year"] == 2025]

X_train = train[features]
y_train = train[target]
X_test = test[features]
y_test = test[target]

# Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)

lr_mae = mean_absolute_error(y_test, lr_pred)
lr_rmse = np.sqrt(mean_squared_error(y_test, lr_pred))
lr_r2 = r2_score(y_test, lr_pred)

print("\nLINEAR REGRESSION")
print("MAE :", lr_mae)
print("RMSE:", lr_rmse)
print("R2  :", lr_r2)

# Random Forest (sampled for faster processing)
sample_size = min(100000, len(X_train))
X_sample = X_train.sample(sample_size, random_state=42)
y_sample = y_train.loc[X_sample.index]

rf = RandomForestRegressor(
    n_estimators=50,
    random_state=42,
    n_jobs=1
)

print("\nTraining Random Forest...")
rf.fit(X_sample, y_sample)
rf_pred = rf.predict(X_test)

rf_mae = mean_absolute_error(y_test, rf_pred)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
rf_r2 = r2_score(y_test, rf_pred)

print("\nRANDOM FOREST")
print("MAE :", rf_mae)
print("RMSE:", rf_rmse)
print("R2  :", rf_r2)

# 5. MODEL COMPARISON
comparison = pd.DataFrame({
    "Model": ["Linear Regression", "Random Forest"],
    "MAE": [lr_mae, rf_mae],
    "RMSE": [lr_rmse, rf_rmse],
    "R2": [lr_r2, rf_r2]
})

comparison.to_csv("model_comparison.csv", index=False)
print("\nModel comparison saved.")

# 6. ACTUAL VS PREDICTED
plt.figure(figsize=(12, 6))
plt.plot(y_test.values[:200], label="Actual")
plt.plot(rf_pred[:200], label="Predicted")
plt.title("Actual vs Predicted Temperature")
plt.xlabel("Sample")
plt.ylabel("Temperature (°C)")
plt.legend()
plt.tight_layout()
plt.savefig("actual_vs_predicted.png")
plt.close()

# 7. FINAL CSV FOR POWER BI
result = test[["time", "city", "temperature_2m"]].copy()
result["predicted_temperature"] = rf_pred

result.rename(
    columns={"temperature_2m": "actual_temperature"},
    inplace=True
)

result.to_csv("temperature_predictions.csv", index=False)

print("\n================================")
print("PROJECT COMPLETED SUCCESSFULLY")
print("================================")
print("Power BI file: temperature_predictions.csv")
print("Model comparison: model_comparison.csv")
print("Graphs: district_temperature.png, monthly_temperature.png, actual_vs_predicted.png")
