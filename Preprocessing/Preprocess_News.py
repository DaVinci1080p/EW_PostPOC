import numpy as np
import pandas as pd
from scipy.sparse import hstack, save_npz
from sklearn.feature_extraction.text import TfidfVectorizer

# Load and clean data
path = "./Datasets/baselineData_PreProcessed_Clean.csv"
df = pd.read_csv(path)
df["Title"] = df["Title"].str.strip().str.lower().str.replace(r"\s+", " ", regex=True)

# Convert dates to Unix timestamp (numerical)
df["Publication Date"] = pd.to_datetime(df["Publication Date"])
df["Publication Date"] = (
    df["Publication Date"].astype(np.int64) // 10**9
)  # Convert to seconds

# Vectorize text data
vectorizer = TfidfVectorizer(max_features=1000)  # Adjust 'max_features' as needed
X = vectorizer.fit_transform(df["Title"])

# Combine dates with TF-IDF matrix
dates = df["Publication Date"].values[:, None]  # Reshape to match X dimensions
combined_matrix = hstack([X, dates])

print(dates.shape)
# Save the combined matrix
save_npz("./Datasets/tfidf_matrix_with_dates.npz", combined_matrix)

# Optionally, save the TF-IDF model
import joblib

joblib.dump(vectorizer, "./Datasets/tfidf_model.pkl")
