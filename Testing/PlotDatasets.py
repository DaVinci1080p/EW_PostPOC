import datetime
import os
import random

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np


def load_data_from_random_file(save_path):
    # Create a list of available file numbers
    file_numbers = [
        f.split("_")[2].split(".")[0]
        for f in os.listdir(save_path)
        if f.startswith("val_data_") and f.endswith(".npz")
    ]

    # Randomly select a file number
    selected_file_number = random.choice(file_numbers)

    # Construct the file names using the selected number
    selected_train_file = f"train_data_{selected_file_number}.npz"

    print(f"Loaded train data from file: {selected_train_file}")

    # Load the data from the files
    train_data = np.load(
        os.path.join(save_path, selected_train_file), allow_pickle=True
    )

    return train_data


def filter_data_for_ids(data, num_ids):
    unique_ids = np.unique(data["ids"])
    selected_ids = np.random.choice(unique_ids, num_ids, replace=False)

    filtered_targets = []
    target_dates = []  # Target date for each sequence

    for id_ in selected_ids:
        id_filter = data["ids"] == id_
        sequences = data["sequences"][id_filter]
        targets = data["targets"][id_filter]
        dates = data["dates"][id_filter]

        # Skip the last sequence of each ID as its target date is not available
        if len(sequences) > 1:
            filtered_targets.extend(targets[:-1])
            target_dates.extend(
                dates[1:, -1]
            )  # Target date is the last date of the next sequence
        else:
            print(f"Insufficient data for ID {id_}")

    return (
        np.array(filtered_targets),
        np.array(target_dates),
        selected_ids,
    )


def plot_predictions(id_data):
    plt.figure(figsize=(10, 6))

    # Generate a color for each ID
    colors = plt.cm.get_cmap(
        "tab10", len(id_data)
    )  # 'tab10' is a colormap with 10 distinct colors

    for i, (id_, data) in enumerate(id_data.items()):
        # Convert dates to a plot-friendly format
        dates = [datetime.datetime.fromtimestamp(d) for d in data["dates"]]

        plt.scatter(
            dates,
            data["actuals"],
            label=f"Actual - ID {id_}",
            alpha=0.7,
            color=colors(i),
        )

    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.title("Scaled and Normalized Dataset")
    plt.legend()
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=100))
    plt.gcf().autofmt_xdate()
    plt.show()


def main():
    data_path = "./Datasets/Sequences"

    num_ids = 5
    # Load data from a random train and test file
    data = load_data_from_random_file(data_path)

    # Filter for 5 unique IDs and prepare data for prediction
    targets, target_dates, selected_ids = filter_data_for_ids(data, num_ids)

    # Organize data by ID for plotting
    id_data = {}
    sequence_index = 0  # Index to keep track of sequences
    for id_ in selected_ids:
        id_length = np.sum(data["ids"] == id_) - 1

        id_data[id_] = {
            "dates": target_dates[sequence_index : sequence_index + id_length],
            "actuals": targets[sequence_index : sequence_index + id_length],
        }
        sequence_index += id_length

    # Plotting
    plot_predictions(id_data)


if __name__ == "__main__":
    main()
