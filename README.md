# EW_PostPOC

EW_PostPOC is dedicated to the Post Proof of Concept (PoC) phase of an AI-Model test platform for a thesis by Roy Wendries. It encompasses various scripts and tools used in different stages of machine learning model development, including preprocessing, training, prediction, and testing.

## Directory Structure Overview

### 1. Preprocessing Scripts (`Preprocessing`)
This directory contains scripts for preprocessing datasets, preparing them for subsequent machine learning processes.

- **Preprocessing_SequenceGeneration.py:** Converts meter data files into .npz format, generating sequences of dates and values along with the target value.
- **Preprocessing_Brent-Gas.py:** Processes Brent and Gas data to address timeseries gaps, ensuring continuity and completeness.

### 2. Training Scripts (`Training`)
This section houses scripts for training different machine learning models.

- **LSTM_Model_Trainer.py:** Facilitates training of Long Short-Term Memory (LSTM) models.
- **XGBoost_Trainer.py:** A script for training models using the XGBoost framework.
- **XGBoost_Tertiary_Trainer.py:** Specialized XGBoost training script that incorporates additional model data, currently optimized for Gas and Brent data.

### 3. Prediction and Validation Scripts (`Predictions`)
Scripts in this directory are used for plotting predictions and validating the performance of trained machine learning models.

- **LSTM_Model_Predictions.py:** Visualizes the actual vs. predicted values of LSTM models.
- **XGBoost_Plotter.py:** Plots actual vs. predicted values of XGBoost models using validation data.
- **XGBoost_Tertiary_Plotter.py:** Graphs the actual vs. predicted values of XGBoost models with tertiary data on validation data sets.

### 4. Testing and Visualization Scripts (`Testing`)
This directory is designated for scripts that assist in data visualization and debugging.

- **PlotDatasets.py:** Generates plots for five preprocessed IDs, aiding in visual analysis.
- **TESTING_View_Sequences.py:** Displays sequences from .npz files, showing initial and final rows, along with the maximum value per sequence and ID.
- **TESTING_View_Total-Maxes.py:** Exhibits the maximum value for each ID and enumerates occurrences of these maximum values across IDs.

### 5. Tertiary Dataset Scripts (`Tertiary`)
This directory contains scripts for creating and managing tertiary datasets, essential for enhanced model training and analysis.

- **API-Script_Oil_Gas.py:** Retrieves oil and gas data through specific APIs, integrating it into the model's data pool.
- **Extract-Merge_Datasets.py:** Combines various datasets, facilitating the creation of comprehensive tertiary datasets for advanced model training.
