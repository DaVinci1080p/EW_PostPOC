import datetime
import os

import numpy as np
import pandas as pd


def load_csv(file_path):
    print(f"Loading CSV file: {file_path}")
    return pd.read_csv(file_path)


def create_weekly_sequences(df, sequence_length):
    print("Creating weekly sequences...")
    sequences = []
    dates = []
    targets = []
    df["period"] = pd.to_datetime(df["period"])
    df["period"] = df["period"].dt.date

    for i in range(len(df) - sequence_length):
        sequence = df.iloc[i : i + sequence_length]["value"]
        sequence_dates = df.iloc[i : i + sequence_length]["period"]
        target = df.iloc[i + sequence_length]["value"]

        sequences.append(sequence)
        dates.append(sequence_dates)
        targets.append(target)

    return np.array(sequences), np.array(targets), np.array(dates)


def save_data(sequences, targets, dates, save_path, save_name):
    unix_dates = []
    for date_array in dates:
        # Convert datetime.date objects to Unix timestamps
        unix_dates.append(
            [
                datetime.datetime.combine(d, datetime.time()).timestamp()
                for d in date_array
            ]
        )

    print("Saving data to file...")

    os.makedirs(save_path, exist_ok=True)
    file_path = os.path.join(save_path, f"{save_name}.npz")
    np.savez(file_path, sequences=sequences, targets=targets, dates=unix_dates)
    print(f"Saved file to {file_path}")


def process_file(file_path, save_path, save_name):
    print(f"Processing file: {file_path}")
    df = load_csv(file_path)
    sequences, targets, dates = create_weekly_sequences(df, 7)

    save_data(sequences, targets, dates, save_path, save_name)


def main():
    tertiary_data_path = "./Datasets/Gas_spotPrice_Extrapolated.csv"
    save_path = "./Datasets/Sequences"
    save_name = "Gas_sequences"
    process_file(tertiary_data_path, save_path, save_name)


if __name__ == "__main__":
    main()
