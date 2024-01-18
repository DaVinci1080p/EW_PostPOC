import datetime
import os
import random

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import xgboost as xgb
from matplotlib import colormaps
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def load_xgb_model(model_path):
    """
    Load an XGBoost model from a given path.

    Parameters:
    model_path (str): Path to the saved XGBoost model.

    Returns:
    xgb.Booster: Loaded XGBoost model.
    """
    model = xgb.Booster()
    model.load_model(model_path)
    return model


def calculate_metrics(actuals, predictions):
    """
    Calculate and return regression metrics.

    Parameters:
    actuals (numpy.ndarray): Actual values.
    predictions (numpy.ndarray): Predicted values.

    Returns:
    tuple: RMSE, R2, MAE, and MSE metrics.
    """
    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    r2 = r2_score(actuals, predictions)
    mae = mean_absolute_error(actuals, predictions)
    mse = mean_squared_error(actuals, predictions)
    return rmse, r2, mae, mse


def load_data_from_random_file(save_path):
    """
    Load data from a randomly selected file in a given path.

    Parameters:
    save_path (str): Directory path where the data files are stored.

    Returns:
    dict: Loaded data from the file.
    """
    file_numbers = [
        f.split("_")[2].split(".")[0]
        for f in os.listdir(save_path)
        if f.startswith("val_data_") and f.endswith(".npz")
    ]
    selected_file_number = random.choice(file_numbers)
    selected_file = f"val_data_{selected_file_number}.npz"
    print(f"Loaded val data from file: {selected_file}")
    return np.load(os.path.join(save_path, selected_file), allow_pickle=True)


def filter_data_for_ids(data, num_ids=2):
    """
    Filter data for a specified number of unique IDs.

    Parameters:
    data (dict): Data containing 'sequences', 'targets', 'dates', and 'ids'.
    num_ids (int): Number of unique IDs to filter.

    Returns:
    tuple: Filtered sequences, targets, dates, target dates, and selected IDs.
    """
    unique_ids = np.unique(data["ids"])
    if len(unique_ids) < num_ids:
        raise ValueError("Not enough unique IDs in the data.")

    selected_ids = np.random.choice(unique_ids, num_ids, replace=False)
    filtered_sequences, filtered_targets, filtered_dates, target_dates = [], [], [], []

    for id_ in selected_ids:
        id_filter = data["ids"] == id_
        sequences, targets, dates = (
            data["sequences"][id_filter],
            data["targets"][id_filter],
            data["dates"][id_filter],
        )
        if len(sequences) > 1:
            filtered_sequences.extend(sequences[:-1])
            filtered_targets.extend(targets[:-1])
            filtered_dates.extend(dates[:-1])  # Exclude last sequence date
            target_dates.extend(
                dates[1:, -1]
            )  # Target date is last date of next sequence
        else:
            print(f"Insufficient data for ID {id_}")

    return (
        np.array(filtered_sequences),
        np.array(filtered_targets),
        np.array(filtered_dates),
        np.array(target_dates),
        selected_ids,
    )


def flatten_input(sequences, dates):
    """
    Flatten and combine sequences and dates into a single input array.

    Parameters:
    sequences (numpy.ndarray): Sequence data.
    dates (numpy.ndarray): Corresponding dates for each sequence.

    Returns:
    numpy.ndarray: Flattened input array.
    """
    return np.stack([sequences, dates], axis=-1).reshape(sequences.shape[0], -1)


def plot_predictions(id_data):
    """
    Plot predictions and actual values for different IDs.

    Parameters:
    id_data (dict): Dictionary containing dates, actuals, and predictions for each ID.
    """
    plt.figure(figsize=(14, 6))
    colors = colormaps.get_cmap("tab10")  # Updated method to get colormap

    for i, (id_, data) in enumerate(id_data.items()):
        # Convert dates to a plot-friendly format
        dates = [datetime.datetime.fromtimestamp(d) for d in data["dates"]]

        # Plotting actual values
        plt.scatter(
            dates,
            data["actuals"],
            label=f"Actual - ID {id_}",
            alpha=0.7,
            color=colors(i),
        )

        # Plotting predicted values
        plt.scatter(
            dates,
            data["predictions"],
            label=f"Predicted - ID {id_}",
            alpha=0.7,
            color=colors(i),
            marker="x",
        )

        # Calculate and display metrics
        rmse, r2, mae, mse = calculate_metrics(data["actuals"], data["predictions"])
        metrics_text = (
            f"ID {id_} - RMSE: {rmse:.5f}, R2: {r2:.5f}, MAE: {mae:.5f}, MSE: {mse:.5f}"
        )
        plt.text(
            0.05,
            0.95 - 0.05 * i,
            metrics_text,
            transform=plt.gca().transAxes,
            fontsize=9,
            verticalalignment="top",
        )

    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.title("Predictions vs Actual")
    plt.legend(
        loc="upper left",
        bbox_to_anchor=(1, 1),
        borderaxespad=1,
    )  # Moving the legend outside of the plot
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    plt.gca().xaxis.set_major_locator(
        mdates.DayLocator(interval=100)
    )  # Adjust the interval as needed
    plt.gcf().autofmt_xdate()  # Auto-format the x-axis date labels

    # Adjust subplot parameters to give the plot more room
    plt.subplots_adjust(right=0.7)

    # When saving the figure, make sure to include the extra space for the legend
    base_filename = "./Plots/XGB_Baseline_Model"
    extension = ".png"
    counter = 0
    filename = f"{base_filename}{extension}"
    while os.path.isfile(filename):
        counter += 1
        filename = f"{base_filename}{counter}{extension}"
    plt.savefig(f"{filename}", bbox_inches="tight")
    plt.show()


def main():
    """
    Main function to execute the model prediction and plotting process.
    """
    data_path = "./Datasets/Sequences"
    model_path = "./Models/XGBoost/xgb_model_epochs-1.json"
    num_ids = 2

    print("Loading model...")
    model = load_xgb_model(model_path)

    # feature importance
    # Custom feature names mapping
    custom_feature_names = {
        "f0": "Value1",
        "f1": "Value2",
        "f2": "Value3",
        "f3": "Value4",
        "f4": "Value5",
        "f5": "Value6",
        "f6": "Value7",
        "f7": "Date1",
        "f8": "Date2",
        "f9": "Date3",
        "f10": "Date4",
        "f11": "Date5",
        "f12": "Date6",
        "f13": "Date7",
    }
    feature_importance = model.get_score(
        importance_type="weight"
    )  # 'weight' can be replaced with 'gain' or 'cover'
    # Convert feature importance dictionary to lists for plotting
    features, importances = zip(*feature_importance.items())

    # Replace default feature names with custom names
    custom_features = [custom_feature_names.get(f, f) for f in features]

    print(custom_features)
    print(importances)

    # Sorting the features by importance
    sorted_indices = sorted(
        range(len(importances)), key=lambda i: importances[i], reverse=True
    )
    sorted_features = [custom_features[i] for i in sorted_indices]
    sorted_importances = [importances[i] for i in sorted_indices]

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.barh(sorted_features, sorted_importances)
    plt.xlabel("Feature Importance")
    plt.ylabel("Features")
    plt.title("Feature Importance in XGBoost Model")
    plt.gca().invert_yaxis()  # Invert y-axis to have the most important feature on top

    # When saving the figure, make sure to include the extra space for the legend
    base_filename = "./Plots/XGB_Baseline_Model_FeatureImportance"
    extension = ".png"
    counter = 0
    filename = f"{base_filename}{extension}"
    while os.path.isfile(filename):
        counter += 1
        filename = f"{base_filename}{counter}{extension}"
    plt.savefig(f"{filename}", bbox_inches="tight")

    plt.show()
    print("Loading data...")
    data = load_data_from_random_file(data_path)

    print("Filtering data for selected IDs...")
    sequences, targets, all_dates, target_dates, selected_ids = filter_data_for_ids(
        data, num_ids=num_ids
    )

    print(f"Selected IDs for prediction: {selected_ids}")
    combined_input = flatten_input(sequences, all_dates)

    print("Making predictions...")
    dmatrix = xgb.DMatrix(combined_input)
    predictions = model.predict(dmatrix)

    id_data = {}
    sequence_index = 0
    for id_ in selected_ids:
        id_length = np.sum(data["ids"] == id_) - 1
        id_data[id_] = {
            "dates": target_dates[sequence_index : sequence_index + id_length],
            "actuals": targets[sequence_index : sequence_index + id_length],
            "predictions": predictions[sequence_index : sequence_index + id_length],
        }
        sequence_index += id_length

    print("Plotting predictions...")
    plot_predictions(id_data)


if __name__ == "__main__":
    main()
