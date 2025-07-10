import pandas as pd
import pickle

def load_model():
    with open('models/fake_acc_model.pkl', 'rb') as f:
        model = pickle.load(f)
    print("✅ Model loaded successfully!")
    return model

def predict_new(model, input_data):
    df = pd.DataFrame([input_data])
    prediction = model.predict(df)[0]
    print(f"Prediction: {'Fake Account' if prediction == 1 else 'Real Account'}")

if __name__ == "__main__":
    model = load_model()
    # Example input: change the values to test
    new_account = {
        'followers': 500,
        'friends': 300,
        'posts': 50,
        'age': 3
    }
    predict_new(model, new_account)
