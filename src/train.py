# src/train.py
import pandas as pd
from sklearn.linear_model import LogisticRegression
import pickle
import os

def train_model():
    # Ensure models directory exists
    os.makedirs('models', exist_ok=True)

    print("1. Loading dataset from data/dataset.csv...")
    try:
        # Load the data we tracked in Task 1.2
        df = pd.read_csv('data/dataset.csv')

        # Split features (X) and target (y)
        X = df[['feature_1', 'feature_2']]
        y = df['target']
    except FileNotFoundError:
        print("ERROR: data/dataset.csv not found. Did you run 'dvc pull' if the data was missing?")
        return

    print(f"2. Training model with {len(df)} rows...")
    model = LogisticRegression()
    model.fit(X, y)

    print("3. Saving model to models/model.pkl...")
    # Save the trained model object using pickle
    with open('models/model.pkl', 'wb') as f:
        pickle.dump(model, f)

    print("4. Model training and saving complete.")

if __name__ == "__main__":
    train_model()