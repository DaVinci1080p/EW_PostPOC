import numpy as np
import pandas as pd


def visualize_npz_file(file_path):
    # Load the npz file with allow_pickle set to True
    with np.load(file_path, allow_pickle=True) as data:
        # Extract arrays from the npz file
        sequences = data["sequences"]
        targets = data["targets"]
        dates = data["dates"]
        ids = data["ids"]

        # Convert the arrays into a pandas DataFrame
        df = pd.DataFrame(
            {
                "Sequence": list(sequences),
                "Target": targets,
                "Dates": list(dates),
                "ID": ids,
            }
        )

        # Adjust pandas display settings for full content view
        pd.set_option("display.max_columns", None)  # Show all columns
        pd.set_option("display.max_colwidth", None)  # Show full width of columns
        pd.set_option(
            "display.max_rows", None
        )  # Show all rows (use with caution for very large DataFrames)

        # Display the first few rows of the DataFrame
        print(df.head())  # Display first few rows of the DataFrame

        # Display the last few rows of the DataFrame
        print("\nLast few rows:")
        print(df.tail())  # By default, tail() shows the last 5 rows

        # Count unique IDs
        print("Amount of unique IDs:", df["ID"].nunique())

        # Find the maximum value within each sequence and then find the maximum per ID
        df["Max_In_Sequence"] = df["Sequence"].apply(lambda x: np.max(x))
        max_per_id = (
            df.groupby("ID")["Max_In_Sequence"].max().reset_index(name="Max_Per_ID")
        )

        print("\nMaximum value in sequences per ID:")
        print(max_per_id)

        return df


# Example usage
file_path = "./Datasets/Sequences/test_data_0.npz"
df = visualize_npz_file(file_path)
