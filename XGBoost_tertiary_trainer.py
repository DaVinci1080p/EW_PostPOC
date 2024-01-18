import csv
import os
import random
from datetime import datetime

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def load_additional_data(csv_file, target_dates):
    """
    Load and preprocess additional data from a CSV file.

    Parameters:
    csv_file (str): Path to the CSV file.
    target_dates (numpy.ndarray): Array of target Unix timestamps.

    Returns:
    X_add (numpy.ndarray): Additional target values for training corresponding to target dates.
    missing_dates (list): Dates in target_dates with no corresponding value in brent data.
    """
    # Load the CSV file into a DataFrame
    additional_data = pd.read_csv(csv_file, parse_dates=["period"], dayfirst=False)
    additional_data["period"] = pd.to_datetime(additional_data["period"])
    additional_data["period"] = additional_data["period"].dt.date

    # Convert target_dates from Unix timestamps to datetime.date objects
    target_dates_as_date = [datetime.fromtimestamp(ts).date() for ts in target_dates]

    # Filter the DataFrame based on the condition
    filtered_data = additional_data[
        additional_data["period"].isin(target_dates_as_date)
    ]

    # Extract the 'value' corresponding to the filtered dates
    X_add = filtered_data["value"].values

    return X_add


def load_saved_data_generator(save_path, data_type, tertiary, num_epochs=2):
    """
    Generator function to yield data for one ID at a time.

    Parameters:
    save_path (str): Directory path where the data files are stored.
    data_type (str): Type of data to load (e.g., 'train', 'test', 'val').
    num_epochs (int): Number of epochs to iterate over the data.

    Yields:
    numpy.ndarray: Combined input features for a single ID.
    numpy.ndarray: Target values for a single ID.
    """
    if tertiary == "bert":
        brent_csv_path = "./Datasets/Brent_spotPrice_Extrapolated.csv"
    elif tertiary == "gas":
        brent_csv_path = "./Datasets/Gas_spotPrice_Extrapolated.csv"

    for epoch in range(num_epochs):
        print(f"Starting epoch {epoch + 1}/{num_epochs}")
        file_list = [
            f
            for f in os.listdir(save_path)
            if f.startswith(f"{data_type}_data_") and f.endswith(".npz")
        ]
        random.shuffle(file_list)  # Shuffle file list

        for file_count, selected_file in enumerate(file_list, 1):
            data = np.load(os.path.join(save_path, selected_file), allow_pickle=True)
            sequences, targets, dates, ids = (
                data["sequences"],
                data["targets"],
                data["dates"],
                data["ids"],
            )
            print(f"Processing file {file_count}/{len(file_list)}: {selected_file}")

            for selected_id in np.unique(ids):
                target_dates = []
                id_filter = ids == selected_id
                id_sequences = sequences[id_filter]
                id_dates = dates[id_filter]

                if len(id_sequences) > 1:
                    target_dates.extend(
                        id_dates[1:, -1]
                    )  # Target date is last date of next sequence
                target_dates = np.array(target_dates)
                brent = load_additional_data(brent_csv_path, target_dates)

                combined_input = np.stack(
                    [sequences[id_filter], dates[id_filter]], axis=-1
                ).reshape(-1, 14)
                combined_input = combined_input[:-1]

                # Reshape brent to match the number of rows in combined_input
                brent = brent.reshape(-1, 1)

                # Concatenate brent as an additional feature
                combined_input = np.concatenate([combined_input, brent], axis=1)

                targets_resize = targets[id_filter]
                targets_resize = targets_resize[:-1]
                yield combined_input, targets_resize


def train_and_evaluate_model(
    sequence_path, tertiary, num_epochs=1, early_stopping_rounds=5
):
    """
    Trains and evaluates an XGBoost regression model.

    Parameters:
    sequence_path (str): Path to the dataset directory.
    num_epochs (int): Number of training epochs.

    Prints:
    The RMSE, R2, MAE, and MSE metrics after evaluation on the test dataset.
    Saves the trained model.
    """
    model = xgb.XGBRegressor(
        objective="reg:squarederror",
        colsample_bytree=0.3,
        learning_rate=0.001,
        max_depth=5,
        alpha=10,
        n_estimators=10,
        verbosity=1,
        early_stopping_rounds=early_stopping_rounds,
    )
    is_model_fitted = False

    # Prepare validation data
    X_val, y_val = next(
        load_saved_data_generator(sequence_path, "val", tertiary, num_epochs=1)
    )

    # Train model with the training data
    for X_train, y_train in load_saved_data_generator(
        sequence_path, "train", tertiary, num_epochs
    ):
        model.fit(
            X_train,
            y_train,
            xgb_model=model.get_booster() if is_model_fitted else None,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        is_model_fitted = True
    print("Completed training on the current batch.")

    # Evaluate model with the test data
    all_preds, all_y_test = [], []
    for X_test, y_test in load_saved_data_generator(
        sequence_path, "test", tertiary, num_epochs=1
    ):
        all_preds.extend(model.predict(X_test))
        all_y_test.extend(y_test)

    # Calculate and print the performance metrics
    rmse = np.sqrt(mean_squared_error(all_y_test, all_preds))
    r2 = r2_score(all_y_test, all_preds)
    mae = mean_absolute_error(all_y_test, all_preds)
    mse = mean_squared_error(all_y_test, all_preds)
    print(f"RMSE: {rmse}, R2: {r2}, MAE: {mae}, MSE: {mse}")

    # Save Model with a unique filename
    base_filename = f"./Models/XGBoost/xgb_{tertiary}-model_epochs-{num_epochs}"
    extension = ".json"
    counter = 0
    filename = f"{base_filename}{extension}"
    while os.path.isfile(filename):
        counter += 1
        filename = f"{base_filename}_{counter}{extension}"
    model.save_model(filename)
    print(f"Model saved as {filename}")

    # Save Test Metrics to CSV
    metrics_filename = f"{base_filename}_{counter}_performance.csv"
    metrics = {"RMSE": rmse, "R2": r2, "MAE": mae, "MSE": mse}
    with open(metrics_filename, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=metrics.keys())
        writer.writeheader()
        writer.writerow(metrics)
    print(f"Model performance saved as {metrics_filename}")


if __name__ == "__main__":
    sequence_path = "./Datasets/Sequences"
    train_and_evaluate_model(sequence_path, "gas", num_epochs=1)
