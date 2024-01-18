import datetime
import os
import random

import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.layers import LSTM, Dense
from keras.models import Sequential
from keras.optimizers import Adam


def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))


def r_squared(y_true, y_pred):
    SS_res = K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return 1 - (SS_res / (SS_tot + K.epsilon()))


# Function to load data, including sequences, targets, and dates
def load_saved_data_generator(save_path, data_type):
    file_list = [
        f
        for f in os.listdir(save_path)
        if f.startswith(f"{data_type}_data_") and f.endswith(".npz")
    ]

    while True:
        random.shuffle(file_list)  # Shuffle file list
        for selected_file in file_list:
            print("\nLoaded data from file: ", selected_file)
            data = np.load(os.path.join(save_path, selected_file), allow_pickle=True)

            sequences = data["sequences"]  # 2D: (num_samples, 7)
            targets = data["targets"]  # 1D: (num_samples,)
            dates = data["dates"]  # 2D: (num_samples, 7)
            ids = data["ids"]  # 1D: (num_samples,)

            unique_ids = np.unique(ids)
            for selected_id in unique_ids:
                id_filter = ids == selected_id
                id_sequences = sequences[id_filter]
                id_dates = dates[id_filter]
                id_targets = targets[id_filter]

                # Combine sequences with processed dates
                # This will result in shape: (num_sequences, 7, 2)
                combined_input = np.stack([id_sequences, id_dates], axis=-1)

                yield combined_input, id_targets


# Define LSTM model
def define_lstm_model(sequence_length, num_features):
    print("Defining the LSTM model...")
    model = Sequential(
        [
            LSTM(
                200,
                activation="tanh",  # tryout tanh
                return_sequences=True,
                input_shape=(sequence_length, num_features),
            ),
            LSTM(200),
            Dense(1),
        ]
    )
    compiler = Adam(learning_rate=0.001)
    model.compile(
        optimizer=compiler,
        loss="mean_squared_error",
        metrics=["mean_absolute_error", root_mean_squared_error, r_squared],
    )
    model.summary()

    return model


# Function to calculate steps per epoch
def calculate_steps_per_epoch(path, data_type):
    unique_id_count = 0
    file_list = [
        f
        for f in os.listdir(path)
        if f.startswith(f"{data_type}_data_") and f.endswith(".npz")
    ]
    for file in file_list:
        data = np.load(os.path.join(path, file))
        unique_ids = np.unique(data["ids"])
        unique_id_count += len(unique_ids)
    print("Output of 'calculate_steps_per_epoch': ", unique_id_count)
    return unique_id_count


# Main function for training and evaluation
def main():
    # Parameters
    sequence_path = "./Datasets/Sequences"
    sequence_length = 7
    epochs = 100
    num_features = 2  # value and date
    log_dir = "./Models/logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir, histogram_freq=1
    )

    # Set the global policy to use float64
    tf.keras.backend.set_floatx("float64")

    # Define and compile the LSTM model
    model = define_lstm_model(sequence_length, num_features)

    # Training
    print("Starting model training...")
    train_generator = load_saved_data_generator(sequence_path, "train")
    steps_per_epoch_train = calculate_steps_per_epoch(sequence_path, "train")
    model.fit(
        train_generator,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch_train,
        callbacks=[tensorboard_callback],
    )

    # Evaluation
    print("Evaluating the model...")
    test_generator = load_saved_data_generator(sequence_path, "test")
    steps_per_epoch_test = calculate_steps_per_epoch(sequence_path, "test")
    model.evaluate(test_generator, steps=steps_per_epoch_test)

    # Model Saving
    print("saving model ...")
    model.save(f"./Models/{steps_per_epoch_train}-IDS_{epochs}-Epochs.keras")
    print(
        "model saved as: ",
        f"./Models/{steps_per_epoch_train}-IDS_{epochs}-Epochs.keras",
    )


if __name__ == "__main__":
    main()
