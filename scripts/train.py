# train.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load dataset
def load_data():
    df = pd.read_csv('data/fake_accounts.csv')
    print("✅ Data loaded successfully!")
    return df

# Train model
def train_model(df):
    X = df.drop("label", axis=1)
    y = df["label"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)
    print(f"🎯 Model accuracy: {accuracy:.2f}")

    # Save the model
    with open('models/fake_acc_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    print("📦 Model saved to models/fake_acc_model.pkl")

# Run everything
if __name__ == "__main__":
    df = load_data()
    train_model(df)
