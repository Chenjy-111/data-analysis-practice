import os
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score


# -----------------------------
# 1. Basic settings
# -----------------------------
LATITUDE = 36.0671
LONGITUDE = 120.3826
START_DATE = "2021-01-01"
END_DATE = "2025-12-31"

DATA_DIR = "data"
FIGURE_DIR = "figures"
REPORT_DIR = "report"

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)


# -----------------------------
# 2. Download weather data
# -----------------------------
def download_weather_data():
    url = "https://archive-api.open-meteo.com/v1/archive"

    daily_variables = [
        "temperature_2m_max",
        "temperature_2m_min",
        "temperature_2m_mean",
        "apparent_temperature_mean",
        "precipitation_sum",
        "rain_sum",
        "wind_speed_10m_max",
        "wind_speed_10m_mean",
        "wind_gusts_10m_max"
    ]

    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "daily": ",".join(daily_variables),
        "timezone": "Asia/Shanghai"
    }

    print("Downloading data from Open-Meteo...")
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()

    data = response.json()["daily"]
    df = pd.DataFrame(data)

    df.rename(columns={"time": "date"}, inplace=True)
    df["date"] = pd.to_datetime(df["date"])

    output_path = os.path.join(DATA_DIR, "qingdao_weather.csv")
    df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"Data saved to {output_path}")
    return df


# -----------------------------
# 3. Clean and enrich data
# -----------------------------
def prepare_data(df):
    df = df.copy()

    # Sort by date
    df = df.sort_values("date").reset_index(drop=True)

    # Basic missing value handling
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].interpolate(method="linear")
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    # Feature engineering
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["dayofyear"] = df["date"].dt.dayofyear
    df["temp_range"] = df["temperature_2m_max"] - df["temperature_2m_min"]
    df["is_rainy_day"] = (df["precipitation_sum"] > 1).astype(int)

    # 7-day rolling average
    df["temp_mean_7d"] = df["temperature_2m_mean"].rolling(window=7).mean()
    df["precipitation_7d"] = df["precipitation_sum"].rolling(window=7).sum()

    df["temp_mean_7d"] = df["temp_mean_7d"].fillna(df["temperature_2m_mean"])
    df["precipitation_7d"] = df["precipitation_7d"].fillna(df["precipitation_sum"])

    return df


# -----------------------------
# 4. Visualization
# -----------------------------
def plot_temperature_trend(df):
    plt.figure(figsize=(12, 5))
    plt.plot(df["date"], df["temperature_2m_mean"], linewidth=0.6, label="Daily mean temperature")
    plt.plot(
        df["date"],
        df["temperature_2m_mean"].rolling(30).mean(),
        linewidth=2,
        label="30-day rolling mean"
    )
    plt.xlabel("Date")
    plt.ylabel("Temperature (°C)")
    plt.title("Qingdao Daily Mean Temperature Trend")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, "temperature_trend.png"), dpi=300)
    plt.close()


def plot_monthly_temperature(df):
    monthly_temp = df.groupby("month")["temperature_2m_mean"].mean()

    plt.figure(figsize=(8, 5))
    plt.bar(monthly_temp.index, monthly_temp.values)
    plt.xlabel("Month")
    plt.ylabel("Mean Temperature (°C)")
    plt.title("Average Monthly Temperature in Qingdao")
    plt.xticks(range(1, 13))
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, "monthly_temperature.png"), dpi=300)
    plt.close()


def plot_monthly_precipitation(df):
    monthly_precipitation = df.groupby("month")["precipitation_sum"].mean()

    plt.figure(figsize=(8, 5))
    plt.bar(monthly_precipitation.index, monthly_precipitation.values)
    plt.xlabel("Month")
    plt.ylabel("Average Daily Precipitation (mm)")
    plt.title("Average Monthly Precipitation in Qingdao")
    plt.xticks(range(1, 13))
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, "monthly_precipitation.png"), dpi=300)
    plt.close()


def plot_correlation_heatmap(df):
    cols = [
        "temperature_2m_max",
        "temperature_2m_min",
        "temperature_2m_mean",
        "apparent_temperature_mean",
        "precipitation_sum",
        "wind_speed_10m_mean",
        "wind_gusts_10m_max",
        "temp_range"
    ]

    corr = df[cols].corr()

    plt.figure(figsize=(9, 7))
    plt.imshow(corr, aspect="auto")
    plt.colorbar(label="Correlation")
    plt.xticks(range(len(cols)), cols, rotation=45, ha="right")
    plt.yticks(range(len(cols)), cols)
    plt.title("Correlation Heatmap of Weather Variables")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, "correlation_heatmap.png"), dpi=300)
    plt.close()


# -----------------------------
# 5. Prediction model
# -----------------------------
def train_prediction_model(df):
    model_df = df.copy()

    # Target: tomorrow's maximum temperature
    model_df["target_next_day_temp_max"] = model_df["temperature_2m_max"].shift(-1)
    model_df = model_df.dropna().reset_index(drop=True)

    feature_cols = [
        "temperature_2m_max",
        "temperature_2m_min",
        "temperature_2m_mean",
        "apparent_temperature_mean",
        "precipitation_sum",
        "rain_sum",
        "wind_speed_10m_max",
        "wind_speed_10m_mean",
        "wind_gusts_10m_max",
        "month",
        "dayofyear",
        "temp_range",
        "is_rainy_day",
        "temp_mean_7d",
        "precipitation_7d"
    ]

    X = model_df[feature_cols]
    y = model_df["target_next_day_temp_max"]

    # Time-based split: first 80% for training, last 20% for testing
    split_index = int(len(model_df) * 0.8)

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]
    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    linear_model = LinearRegression()
    rf_model = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        max_depth=8
    )

    linear_model.fit(X_train, y_train)
    rf_model.fit(X_train, y_train)

    linear_pred = linear_model.predict(X_test)
    rf_pred = rf_model.predict(X_test)

    linear_mae = mean_absolute_error(y_test, linear_pred)
    linear_r2 = r2_score(y_test, linear_pred)

    rf_mae = mean_absolute_error(y_test, rf_pred)
    rf_r2 = r2_score(y_test, rf_pred)

    # Save metrics
    metrics_path = os.path.join(REPORT_DIR, "metrics.txt")
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write("Qingdao Weather Prediction Model Report\n")
        f.write("=" * 45 + "\n\n")
        f.write(f"Data period: {START_DATE} to {END_DATE}\n")
        f.write(f"Number of samples: {len(model_df)}\n")
        f.write("Target: next day's maximum temperature\n\n")

        f.write("Linear Regression:\n")
        f.write(f"MAE: {linear_mae:.3f} °C\n")
        f.write(f"R2 Score: {linear_r2:.3f}\n\n")

        f.write("Random Forest Regressor:\n")
        f.write(f"MAE: {rf_mae:.3f} °C\n")
        f.write(f"R2 Score: {rf_r2:.3f}\n\n")

        f.write("Feature columns:\n")
        for col in feature_cols:
            f.write(f"- {col}\n")

    # Prediction result plot
    test_dates = model_df["date"].iloc[split_index:]

    plt.figure(figsize=(12, 5))
    plt.plot(test_dates, y_test.values, label="Actual", linewidth=2)
    plt.plot(test_dates, rf_pred, label="Random Forest Prediction", linewidth=1.5)
    plt.xlabel("Date")
    plt.ylabel("Maximum Temperature (°C)")
    plt.title("Next-Day Maximum Temperature Prediction")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, "prediction_result.png"), dpi=300)
    plt.close()

    print("Model training completed.")
    print(f"Linear Regression MAE: {linear_mae:.3f} °C, R2: {linear_r2:.3f}")
    print(f"Random Forest MAE: {rf_mae:.3f} °C, R2: {rf_r2:.3f}")


# -----------------------------
# 6. Main function
# -----------------------------
def main():
    df = download_weather_data()
    df = prepare_data(df)

    cleaned_path = os.path.join(DATA_DIR, "qingdao_weather_cleaned.csv")
    df.to_csv(cleaned_path, index=False, encoding="utf-8-sig")

    plot_temperature_trend(df)
    plot_monthly_temperature(df)
    plot_monthly_precipitation(df)
    plot_correlation_heatmap(df)

    train_prediction_model(df)

    print("All tasks completed.")
    print("Please check the data/, figures/, and report/ folders.")


if __name__ == "__main__":
    main()
