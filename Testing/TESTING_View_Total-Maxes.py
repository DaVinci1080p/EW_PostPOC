import pandas as pd

# File path
csv_file_path = "G:/Dataset/Eneco_Electricity_Timeseries000000000000.csv"
# Read the CSV file into a DataFrame
df = pd.read_csv(csv_file_path)

# Convert the 'read_ts' column to datetime and extract the date component
print("converting datetime")
df["read_ts"] = pd.to_datetime(df["read_ts"])
df["date"] = df["read_ts"].dt.date

print("Suming value per day")
# Group by 'date' and 'internal_datasource_id' and sum the 'value'
daily_totals = (
    df.groupby(["date", "internal_datasource_id"])["value"].sum().reset_index()
)

# Find the maximum 'value' for each 'internal_datasource_id'
max_values_per_id = (
    daily_totals.groupby("internal_datasource_id")["value"]
    .max()
    .reset_index(name="max_value")
)

# Count the occurrences of the maximum 'value' for each 'internal_datasource_id'
# This will create a Series with 'internal_datasource_id' as the index
max_counts_per_id = (
    daily_totals.groupby("internal_datasource_id")
    .apply(lambda x: (x["value"] == x["value"].max()).sum())
    .reset_index(name="max_count")
)

# Combine the maximum values and their counts into a single DataFrame
# Ensure that both DataFrames have 'internal_datasource_id' as a column for merging
max_values_and_counts_per_id = pd.merge(
    max_values_per_id, max_counts_per_id, on="internal_datasource_id"
)

print(max_values_and_counts_per_id)
