import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# LOAD DATASETS

train_df = pd.read_csv("2C_battery-1.csv")
test_df = pd.read_csv("2C_battery-2.csv")

# FEATURES

drop_cols = [
    "runtime_remaining_min",
    "RUL",
    "cycle_life",
    "relative_time_min"
]

X_train = train_df.drop(columns=drop_cols, errors="ignore")
y_train = train_df["runtime_remaining_min"]

X_test = test_df.drop(columns=drop_cols, errors="ignore")
y_test = test_df["runtime_remaining_min"]

# MODEL

model = xgb.XGBRegressor(
    n_estimators=300,
    max_depth=8,
    learning_rate=0.05,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

pred = model.predict(X_test)

# METRICS


print(f"Train Rows : {len(train_df):,}")
print(f"Test Rows  : {len(test_df):,}")

print()

print("MAE :", mean_absolute_error(y_test, pred))
print("RMSE:", mean_squared_error(y_test, pred) ** 0.5)
print("R²  :", r2_score(y_test, pred))




























# from scipy.io import loadmat
# import pandas as pd
# import numpy as np

# file = r'Batch-1\2C_battery-2.mat'
# # Load MAT file
# mat = loadmat(
#     file,
#     squeeze_me=True,
#     struct_as_record=False
# )

# data = mat["data"]
# summary = mat["summary"]

# # ==================================================
# # BUILD SUMMARY DATAFRAME
# # ==================================================

# summary_df = pd.DataFrame({
#     "cycle": np.arange(1, len(summary.charge_capacity_Ah) + 1),

#     "charge_capacity_Ah": summary.charge_capacity_Ah,
#     "discharge_capacity_Ah": summary.discharge_capacity_Ah,

#     "charge_power_Wh": summary.charge_power_Wh,
#     "discharge_power_Wh": summary.discharge_power_Wh,

#     "charge_median_voltage": summary.charge_median_voltage,
#     "discharge_median_voltage": summary.discharge_median_voltage,

#     "charge_mean_voltage": summary.charge_mean_voltage,
#     "discharge_mean_voltage": summary.discharge_mean_voltage,

#     "cycle_life": summary.cycle_life
# })

# # ==================================================
# # EXTRACT RAW TIMESERIES DATA
# # ==================================================

# all_cycles = []

# for cycle_num, cycle in enumerate(data, start=1):

#     n = len(cycle.relative_time_min)

#     temp_df = pd.DataFrame({
#         "cycle": np.full(n, cycle_num),

#         "relative_time_min": cycle.relative_time_min,

#         "voltage_V": cycle.voltage_V,
#         "current_A": cycle.current_A,

#         "capacity_Ah": cycle.capacity_Ah,
#         "power_Wh": cycle.power_Wh,

#         "temperature_C": cycle.temperature_C
#     })

#     all_cycles.append(temp_df)

# raw_df = pd.concat(all_cycles, ignore_index=True)

# # ==================================================
# # MERGE RAW + SUMMARY
# # ==================================================

# df = raw_df.merge(summary_df, on="cycle", how="left")

# # ==================================================
# # FEATURE ENGINEERING
# # ==================================================

# # SOC
# cycle_max_capacity = (
#     df.groupby("cycle")["capacity_Ah"]
#       .transform("max")
# )

# df["SOC"] = (
#     df["capacity_Ah"] /
#     cycle_max_capacity
# )

# # SOH
# initial_capacity = summary_df["discharge_capacity_Ah"].iloc[0]

# df["SOH"] = (
#     df["discharge_capacity_Ah"] /
#     initial_capacity
# )

# # Voltage gradient
# df["dV_dt"] = (
#     df.groupby("cycle")["voltage_V"]
#       .diff()
#       .fillna(0)
# )

# # Current gradient
# df["dCurrent_dt"] = (
#     df.groupby("cycle")["current_A"]
#       .diff()
#       .fillna(0)
# )

# # Temperature gradient
# df["dTemp_dt"] = (
#     df.groupby("cycle")["temperature_C"]
#       .diff()
#       .fillna(0)
# )

# # Capacity fade
# df["capacity_fade"] = (
#     initial_capacity -
#     df["discharge_capacity_Ah"]
# )

# # Energy fade
# initial_energy = summary_df["discharge_power_Wh"].iloc[0]

# df["energy_fade"] = (
#     initial_energy -
#     df["discharge_power_Wh"]
# )

# # Cycle fraction
# df["cycle_fraction"] = (
#     df["cycle"] /
#     df["cycle_life"]
# )

# # ==================================================
# # TARGET 1 : RUNTIME REMAINING
# # ==================================================

# max_cycle_time = (
#     df.groupby("cycle")["relative_time_min"]
#       .transform("max")
# )

# df["runtime_remaining_min"] = (
#     max_cycle_time -
#     df["relative_time_min"]
# )

# # ==================================================
# # TARGET 2 : RUL
# # ==================================================

# df["RUL"] = (
#     df["cycle_life"] -
#     df["cycle"]
# )

# # Avoid negative values
# df["RUL"] = df["RUL"].clip(lower=0)

# # ==================================================
# # SAVE
# # ==================================================

# df.to_csv(
#     "Batch-1/2C_battery-2.csv",
#     index=False
# )

