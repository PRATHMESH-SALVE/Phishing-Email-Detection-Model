import joblib

model = joblib.load("models/phishing_model.pkl")

while True:
    email = input("\nEnter email text:\n")

    prediction = model.predict([email])

    if prediction[0] == 1:
        print("\n⚠️ PHISHING EMAIL")
    else:
        print("\n✅ SAFE EMAIL")