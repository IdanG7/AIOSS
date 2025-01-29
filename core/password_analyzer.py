from core.database_utils import check_common_passwords
from core.ml_suggestions import generate_ml_suggestions
from zxcvbn import zxcvbn
import re


class PasswordAnalyzer:
    def __init__(
        self,
        db_name="data/passwords.db",
        ml_model="models/password_strength_model.pkl",
        min_length=8,
        require_special=False,
        require_numbers=False,
    ):
        """
        Initialize the Password Analyzer.
        """
        self.db_name = db_name
        self.min_length = min_length
        self.require_special = require_special
        self.require_numbers = require_numbers

        # Load ML model for suggestions
        self.ml_model, self.vectorizer = generate_ml_suggestions.load_model(ml_model)

    def check_strength(self, password):
        """
        Analyze password strength.
        """
        feedback_list = []
        compromised = False

        # Check if password is common
        if check_common_passwords(self.db_name, password):
            compromised = True
            feedback_list.append(
                "This password is among the top 1,000,000 most common passwords."
            )

        # Apply custom rules
        if len(password) < self.min_length:
            feedback_list.append(
                f"Password must be at least {self.min_length} characters long."
            )
        if self.require_special and not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
            feedback_list.append(
                "Password must contain at least one special character."
            )
        if self.require_numbers and not re.search(r"[0-9]", password):
            feedback_list.append(
                "Password must contain at least one numeric character."
            )

        # Analyze with zxcvbn
        zxcvbn_analysis = zxcvbn(password)
        zxcvbn_score = zxcvbn_analysis["score"]
        feedback_list.extend(zxcvbn_analysis["feedback"]["suggestions"])

        # Add ML-based suggestions
        ml_feedback = generate_ml_suggestions.get_suggestions(
            self.ml_model, self.vectorizer, password
        )
        feedback_list.extend(ml_feedback)

        return {
            "zxcvbn_score": zxcvbn_score,
            "compromised": compromised,
            "feedback": feedback_list,
        }
