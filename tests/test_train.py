# tests/test_train.py
import pandas as pd
import os
import pickle
# Import the training function from your src folder
from src.train import train_model 

# --- Setup/Validation Test ---
def test_data_shape_validation():
    """Ensure the minimal dataset structure is correct."""
    try:
        df = pd.read_csv('data/dataset.csv')
        # Check for the expected number of columns (3: feature_1, feature_2, target)
        assert df.shape[1] == 3
        # Check if there are at least 2 rows
        assert df.shape[0] >= 2
    except FileNotFoundError:
        # DVC should ensure the file exists, but this handles simple errors
        assert False, "data/dataset.csv not found. Check DVC/Git status."

# --- Functional Test ---
def test_model_training_output():
    """Test if the training function runs and creates the model file."""
    model_path = 'models/model.pkl'

    # 1. Clean up existing model if it exists
    if os.path.exists(model_path):
        os.remove(model_path)

    # 2. Run the training function
    train_model()

    # 3. Check if the output file was created
    assert os.path.exists(model_path)

# --- Integration Test ---
def test_model_can_predict():
    """Test if the saved model can be loaded and used for prediction."""
    model_path = 'models/model.pkl'

    # 1. Load the model
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    # 2. Prepare sample input (must match the format used in training: 2 features)
    sample_input = pd.DataFrame({'feature_1': [10.0], 'feature_2': [11.0]})

    # 3. Predict and ensure output is a valid array/list
    prediction = model.predict(sample_input)

    assert len(prediction) == 1
    assert prediction[0] in [0, 1] # Check if prediction is binary (0 or 1)