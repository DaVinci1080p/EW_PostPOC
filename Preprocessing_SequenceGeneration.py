import datetime
import multiprocessing
import os

import numpy as np
import pandas as pd


def load_csv(file_path):
    print(f"Loading CSV file: {file_path}")
    return pd.read_csv(file_path)


def get_unique_ids(df):
    return df["internal_datasource_id"].unique()


def filter_by_ids(df, ids):
    return df[df["internal_datasource_id"].isin(ids)]


def split_ids(ids, train_ratio, test_ratio):
    np.random.shuffle(ids)
    total_ids = len(ids)
    train_end = int(total_ids * train_ratio)
    test_end = train_end + int(total_ids * test_ratio)
    return ids[:train_end], ids[train_end:test_end], ids[test_end:]


def smooth_series(series, window_size=7):
    return series.rolling(window=window_size, min_periods=1, center=True).mean()


def normalize_values(series, lower_percentile=2, upper_percentile=98):
    # Calculate the lower and upper percentiles to clip the data
    lower_bound = series.quantile(lower_percentile / 100.0)
    upper_bound = series.quantile(upper_percentile / 100.0)

    # Clip the data to remove extreme outliers
    clipped_series = series.clip(lower_bound, upper_bound)

    # Apply Min-Max scaling
    min_val = clipped_series.min()
    max_val = clipped_series.max()

    # Check if the range is zero (constant series after clipping)
    if max_val == min_val:
        raise ValueError(
            "Max and min values are equal after clipping. Cannot apply Min-Max scaling to a constant series."
        )

    scaled_series = (clipped_series - min_val) / (max_val - min_val)

    return scaled_series


def preprocess_and_aggregate(df):
    print("Preprocessing and aggregating data...")
    df["read_ts"] = pd.to_datetime(df["read_ts"])
    df["date"] = df["read_ts"].dt.date  # Extract date from timestamp

    # Filter out rows with a 'value' of zero
    df = df[df["value"] != 0]

    # Aggregate values by date and ID
    daily_totals = (
        df.groupby(["date", "internal_datasource_id"])["value"].sum().reset_index()
    )

    # Normalize daily totals per ID
    daily_totals["value_normalized"] = daily_totals.groupby("internal_datasource_id")[
        "value"
    ].transform(normalize_values)

    # Apply smoothing per ID
    daily_totals["value_smoothed"] = daily_totals.groupby("internal_datasource_id")[
        "value_normalized"
    ].transform(lambda x: smooth_series(x))

    # Sort the DataFrame by internal_datasource_id
    daily_totals = daily_totals.sort_values(by="internal_datasource_id")

    return daily_totals


def create_weekly_sequences(df, sequence_length):
    print("Creating weekly sequences...")
    sequences = []
    targets = []
    dates = []  # To keep track of dates for each sequence
    ids = []  # To store the ID for each sequence

    grouped = df.groupby("internal_datasource_id")
    for name, group in grouped:
        group = group.sort_values("date")

        if group["date"].max() - group["date"].min() >= pd.Timedelta(days=365):
            for i in range(len(group) - sequence_length):
                sequence = group.iloc[i : i + sequence_length]["value_smoothed"].values
                target = group.iloc[i + sequence_length]["value_smoothed"]
                sequence_dates = group.iloc[i : i + sequence_length]["date"].values

                sequences.append(sequence)
                targets.append(target)
                dates.append(sequence_dates)
                ids.append(name)  # Append the current ID

    return np.array(sequences), np.array(targets), dates, ids


def save_data(sequences, targets, dates, ids, file_counter, save_path, data_type):
    try:
        print("Converting dates...")
        unix_dates = []
        for date_array in dates:
            # Convert datetime.date objects to Unix timestamps
            unix_dates.append(
                [
                    datetime.datetime.combine(d, datetime.time()).timestamp()
                    for d in date_array
                ]
            )

        # Ensure save directory exists
        os.makedirs(save_path, exist_ok=True)

        file_path = os.path.join(save_path, f"{data_type}_data_{file_counter}.npz")
        np.savez(
            file_path, sequences=sequences, targets=targets, dates=unix_dates, ids=ids
        )
        print(f"Saved {data_type} data to {file_path}")
    except Exception as e:
        print(f"An error occurred in save_data: {e}")


def init_globals(shared_max_ids, shared_lock, shared_stop_flag):
    global max_ids, counter_lock, stop_flag
    max_ids = shared_max_ids
    counter_lock = shared_lock
    stop_flag = shared_stop_flag


def process_file(
    file_path, sequence_length, train_ratio, test_ratio, save_path, file_counter
):
    global max_ids, counter_lock
    if stop_flag.is_set():
        return  # Stop if the flag is set

    print(f"Processing file: {file_path}")
    df = load_csv(file_path)
    df = preprocess_and_aggregate(df)

    file_ids = get_unique_ids(df)
    # Determine split ratios for this file
    train_ids, test_ids, val_ids = split_ids(file_ids, train_ratio, test_ratio)

    # Process each split separately and save
    for split_name, ids_split in zip(
        ["train", "test", "val"], [train_ids, test_ids, val_ids]
    ):
        split_df = filter_by_ids(df, ids_split)
        sequences, targets, dates, ids = create_weekly_sequences(
            split_df, sequence_length
        )
        if len(sequences) > 0:
            save_data(
                sequences, targets, dates, ids, file_counter, save_path, split_name
            )
        else:
            print(f"No sequences to save for {split_name} in file {file_counter}.")

    # Update the global max_ids value based on unique IDs processed
    with counter_lock:
        unique_ids_in_file = len(file_ids)
        max_ids.value -= unique_ids_in_file
        print("ID's left: ", max_ids.value)
        if max_ids.value <= 0:
            stop_flag.set()
            return


def main():
    timeseries_folder_path = "D:\Datasets\Timeseries"
    shared_max_ids = multiprocessing.Value("i", 10000)  # Maximum unique IDs to process
    shared_lock = multiprocessing.Lock()
    stop_flag = multiprocessing.Event()

    sequence_length = 7  # Length of the sequence in days (e.g., 7 days for a week)
    train_ratio = 0.7
    test_ratio = 0.2  # Assuming 20% for testing and the remaining 10% for validation
    save_path = "./Datasets/Sequences"

    file_paths = [
        os.path.join(timeseries_folder_path, f)
        for f in os.listdir(timeseries_folder_path)
        if f.endswith(".csv")
    ]

    # Create a pool of workers to process files in parallel
    with multiprocessing.Pool(
        initializer=init_globals,
        initargs=(shared_max_ids, shared_lock, stop_flag),
        processes=7,  # Adjust the number of processes as needed
    ) as pool:
        for i, path in enumerate(file_paths):
            if stop_flag.is_set():
                print("Stopping further task dispatch.")
                break  # Stop dispatching tasks if the flag is set
            pool.apply_async(
                process_file,
                (path, sequence_length, train_ratio, test_ratio, save_path, i),
            )
        print("Waiting for all workers to finish jobs.")
        pool.close()
        pool.join()
        print("All workers have finished.")


if __name__ == "__main__":
    main()
