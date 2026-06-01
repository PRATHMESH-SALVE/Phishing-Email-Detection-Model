import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix

# Load dataset
df = pd.read_csv("dataset/phishing_email.csv")

# Remove rows with missing values
df = df.dropna(subset=["text_combined", "label"])

# Convert labels to integers
df["label"] = df["label"].astype(int)

# Features and labels
X = df["text_combined"]
y = df["label"]

print("Dataset Shape:", df.shape)
print("\nLabel Counts:")
print(y.value_counts())

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Create ML pipeline
model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("classifier", MultinomialNB())
])

# Train model
print("\nTraining model...")
model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, predictions)

print(f"\nAccuracy: {accuracy:.4f}")

# Confusion Matrix
cm = confusion_matrix(y_test, predictions)

print("\nConfusion Matrix:")
print(cm)

# Save model
joblib.dump(model, "models/phishing_model.pkl")

print("\nModel saved successfully!")