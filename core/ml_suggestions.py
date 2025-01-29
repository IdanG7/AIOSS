import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm
import pickle


# Function to add custom features to the dataset
def add_features(data):
    """
    Add custom features to the dataset.
    :param data: Pandas DataFrame with a 'password' column.
    :return: DataFrame with additional features.
    """
    data["length"] = data["password"].apply(len)
    data["has_numbers"] = data["password"].apply(
        lambda x: any(char.isdigit() for char in x)
    )
    data["has_special"] = data["password"].apply(
        lambda x: any(char in "!@#$%^&*()-_=+[]{}|;:<>,.?/" for char in x)
    )
    data["has_upper"] = data["password"].apply(
        lambda x: any(char.isupper() for char in x)
    )
    return data


# Function to train the password classification model
def train_password_model(csv_file, output_model="models/password_strength_model.pkl"):
    """
    Train a RandomForest model to classify weak and strong passwords with a progress bar.
    :param csv_file: Path to the labeled CSV file (passwords and labels).
    :param output_model: Path to save the trained model.
    """
    # Load the labeled dataset
    print("Loading dataset...")
    data = pd.read_csv(csv_file)

    # Add additional custom features
    print("Adding custom features...")
    data = add_features(data)

    # Feature extraction: Character-level n-grams
    print("Extracting character n-grams...")
    vectorizer = CountVectorizer(analyzer="char", ngram_range=(1, 3), max_features=1000)
    X_text = vectorizer.fit_transform(data["password"]).toarray()

    # Combine n-grams with custom features
    custom_features = data[["length", "has_numbers", "has_special", "has_upper"]]
    X = pd.concat([pd.DataFrame(X_text), custom_features], axis=1)

    # Ensure all column names are strings
    X.columns = X.columns.astype(str)

    # Convert labels to binary (0=weak, 1=strong)
    y = data["label"].map({"weak": 0, "strong": 1})

    # Set up RandomForestClassifier with a progress bar
    n_estimators = 100  # Number of trees in the forest
    model = RandomForestClassifier(
        n_estimators=n_estimators, random_state=42, warm_start=True
    )

    print("Training the model...")
    # Add trees to the model incrementally, updating the progress bar
    with tqdm(total=n_estimators, desc="Training Progress") as pbar:
        for i in range(1, n_estimators + 1):
            model.set_params(n_estimators=i)  # Increment the number of trees
            model.fit(X, y)  # Continue training
            pbar.update(1)  # Update the progress bar

    # Save the trained model and vectorizer
    print("Saving the model...")
    with open(output_model, "wb") as model_file:
        pickle.dump((vectorizer, model), model_file)

    print(f"Model trained and saved to {output_model}")


# Function to load a trained model and vectorizer
def load_model(model_path):
    """
    Load a trained model and vectorizer from disk.
    :param model_path: Path to the saved model file.
    :return: Tuple of (vectorizer, model).
    """
    with open(model_path, "rb") as model_file:
        vectorizer, model = pickle.load(model_file)
    return vectorizer, model


# Function to generate suggestions for improving a password
def get_suggestions(model, vectorizer, password):
    """
    Use the ML model to generate suggestions for improving the password.
    :param model: Trained RandomForestClassifier model.
    :param vectorizer: Vectorizer used for feature extraction.
    :param password: Password to analyze.
    :return: List of improvement suggestions.
    """
    # Extract features from the password
    X_text = vectorizer.transform([password]).toarray()
    length = len(password)
    has_numbers = any(char.isdigit() for char in password)
    has_special = any(char in "!@#$%^&*()-_=+[]{}|;:<>,.?/" for char in password)
    has_upper = any(char.isupper() for char in password)

    # Combine n-gram features with custom features
    X = pd.concat(
        [
            pd.DataFrame(X_text),
            pd.DataFrame(
                [[length, has_numbers, has_special, has_upper]],
                columns=["length", "has_numbers", "has_special", "has_upper"],
            ),
        ],
        axis=1,
    ).values

    # Get prediction probabilities
    prediction = model.predict_proba(X)[0]

    # Generate suggestions based on the model's classification
    suggestions = []
    if prediction[0] > 0.7:  # Threshold for weak passwords
        suggestions.append("This password is classified as weak by the ML model.")

        # Add specific suggestions based on features
        if length < 12:
            suggestions.append(
                "Consider increasing the password length to at least 12 characters."
            )
        if not has_numbers:
            suggestions.append("Add at least one numeric character (e.g., 1, 2, 3).")
        if not has_special:
            suggestions.append("Add at least one special character (e.g., @, $, %, &).")
        if not has_upper:
            suggestions.append("Include at least one uppercase letter.")

    return suggestions
