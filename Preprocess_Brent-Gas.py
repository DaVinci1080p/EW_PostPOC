from datetime import datetime, timedelta

import numpy as np
import pandas as pd

# Load the data
df = pd.read_csv("./Datasets/Gas_SpotPrice.csv", parse_dates=["period"], dayfirst=True)
df["period"] = pd.to_datetime(df["period"]).dt.date

# Define the full date range expected in your dataset
start_date = df["period"].min()
end_date = df["period"].max()
all_dates = pd.date_range(start=start_date, end=end_date, freq="D")

# Convert DatetimeIndex to a list of datetime.date objects
all_dates = [date.date() for date in all_dates]

# Create a new DataFrame with all dates
new_df = pd.DataFrame({"period": all_dates})

# Merge with the original DataFrame
new_df = new_df.merge(df, on="period", how="left")

# Extrapolate missing values using linear interpolation
new_df["value"] = new_df["value"].interpolate(method="linear")

# Optionally, fill any remaining NaN values at the start or end
new_df["value"].fillna(method="bfill", inplace=True)  # Backward fill
new_df["value"].fillna(method="ffill", inplace=True)  # Forward fill

# Round the values to 2 decimal places
new_df["value"] = new_df["value"].round(2)

# Save the new DataFrame to a file
new_df.to_csv("./Datasets/Gas_spotPrice_Extrapolated.csv", index=False)
