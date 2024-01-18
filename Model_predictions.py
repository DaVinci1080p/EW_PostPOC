import datetime
import os
import random

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras import backend as K


def load_model(model_path):
    custom_objects = {
        "root_mean_squared_error": root_mean_squared_error,
        "r_squared": r_squared,
    }
    return tf.keras.models.load_model(model_path, custom_objects=custom_objects)


def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))


def r_squared(y_true, y_pred):
    SS_res = K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return 1 - (SS_res / (SS_tot + K.epsilon()))


def load_data_from_random_file(save_path):
    train_file_list = [
        f
        for f in os.listdir(save_path)
        if f.startswith("train_data_") and f.endswith(".npz")
    ]
    test_file_list = [
        f
        for f in os.listdir(save_path)
        if f.startswith("test_data_") and f.endswith(".npz")
    ]

    selected_train_file = random.choice(train_file_list)
    selected_test_file = random.choice(test_file_list)

    print(f"Loaded train data from file: {selected_train_file}")
    print(f"Loaded test data from file: {selected_test_file}")

    train_data = np.load(
        os.path.join(save_path, selected_train_file), allow_pickle=True
    )
    test_data = np.load(os.path.join(save_path, selected_test_file), allow_pickle=True)

    return train_data, test_data


def combine_train_test_data(train_data, test_data):
    combined_data = {}
    for key in train_data.keys():
        combined_data[key] = np.concatenate([train_data[key], test_data[key]], axis=0)
    return combined_data


def filter_data_for_5_ids(data):
    unique_ids = np.unique(data["ids"])
    selected_ids = np.random.choice(unique_ids, 2, replace=False)

    filtered_sequences = []
    filtered_targets = []
    filtered_dates = []  # Dates for each day in the sequence
    target_dates = []  # Target date for each sequence

    for id_ in selected_ids:
        id_filter = data["ids"] == id_
        sequences = data["sequences"][id_filter]
        targets = data["targets"][id_filter]
        dates = data["dates"][id_filter]

        # Skip the last sequence of each ID as its target date is not available
        if len(sequences) > 1:
            filtered_sequences.extend(sequences[:-1])
            filtered_targets.extend(targets[:-1])
            filtered_dates.extend(dates[:-1])  # All dates except for the last sequence
            target_dates.extend(
                dates[1:, -1]
            )  # Target date is the last date of the next sequence
        else:
            print(f"Insufficient data for ID {id_}")

    return (
        np.array(filtered_sequences),
        np.array(filtered_targets),
        np.array(filtered_dates),
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
        plt.scatter(
            dates,
            data["predictions"],
            label=f"Predicted - ID {id_}",
            alpha=0.7,
            color=colors(i),
            marker="x",
        )

    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.title("Predictions vs Actual")
    plt.legend()
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=100))
    plt.gcf().autofmt_xdate()
    plt.show()


def main():
    data_path = "./Datasets/Sequences"
    model_path = "./Models/356-IDS_10-Epochs.keras"

    # Load the model
    model = load_model(model_path)

    # Load data from a random train and test file
    train_data, test_data = load_data_from_random_file(data_path)

    # Combine train and test data
    combined_data = combine_train_test_data(train_data, test_data)

    # Filter for 5 unique IDs and prepare data for prediction
    sequences, targets, all_dates, target_dates, selected_ids = filter_data_for_5_ids(
        combined_data
    )
    combined_input = np.stack([sequences, all_dates], axis=-1)

    # Generate predictions
    predictions = model.predict(combined_input)

    # Organize data by ID for plotting
    id_data = {}
    sequence_index = 0  # Index to keep track of sequences
    for id_ in selected_ids:
        id_length = (
            np.sum(combined_data["ids"] == id_) - 1
        )  # Minus 1 due to skipping the last sequence

        id_data[id_] = {
            "dates": target_dates[sequence_index : sequence_index + id_length],
            "actuals": targets[sequence_index : sequence_index + id_length],
            "predictions": predictions[
                sequence_index : sequence_index + id_length
            ].flatten(),
        }
        sequence_index += id_length

    # Plotting
    plot_predictions(id_data)


if __name__ == "__main__":
    main()
