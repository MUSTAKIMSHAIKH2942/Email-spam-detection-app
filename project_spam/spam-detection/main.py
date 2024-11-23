from scripts.train_model import train_and_evaluate_model



if __name__ == "__main__":
    # Define dataset path
    filepath = "data/spam.csv"

    print("Training the model...")
    model, vectorizer = train_and_evaluate_model(filepath)

    # Save the model and vectorizer for later use
    import joblib
    joblib.dump(model, "spam_classifier.pkl")
    joblib.dump(vectorizer, "vectorizer.pkl")

    print("Model and vectorizer saved successfully!")
