from flask import Flask, request, render_template, jsonify
from core.ml_suggestions import load_model, get_suggestions

app = Flask(__name__)

# Load the trained model and vectorizer
vectorizer, model = load_model("models/password_strength_model.pkl")


def extract_features(password):
    """
    Extract both character n-grams and additional password features.
    This function must match the feature extraction used during training.
    """
    import pandas as pd

    # Transform the password using the same vectorizer
    X_text = vectorizer.transform([password]).toarray()

    # Compute custom features (must match training logic)
    length = len(password)
    has_numbers = any(char.isdigit() for char in password)
    has_special = any(char in "!@#$%^&*()-_=+[]{}|;:<>,.?/" for char in password)
    has_upper = any(char.isupper() for char in password)

    # Combine n-grams with the four additional features
    X = pd.concat(
        [
            pd.DataFrame(X_text),
            pd.DataFrame(
                [[length, has_numbers, has_special, has_upper]],
                columns=["length", "has_numbers", "has_special", "has_upper"],
            ),
        ],
        axis=1,
    ).values  # Convert to array for prediction

    return X


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze_password():
    password = request.form.get("password")
    if not password:
        return jsonify({"error": "No password provided"}), 400

    # Extract full feature set
    X = extract_features(password)

    # Make prediction
    prediction = model.predict(X)[0]
    classification = "weak" if prediction == 0 else "strong"

    # Get suggestions
    suggestions = get_suggestions(model, vectorizer, password)

    return jsonify(
        {
            "password": password,
            "classification": classification,
            "suggestions": suggestions,
        }
    )


if __name__ == "__main__":
    app.run(debug=True)
