import csv


def create_labeled_passwords_csv(output_file="data/passwords_labeled.csv"):
    """
    Create a CSV file with passwords and their labels (weak/strong).
    """
    # Example weak and strong passwords
    weak_passwords = [
        "123456",
        "password",
        "qwerty",
        "admin123",
        "letmein",
        "iloveyou",
        "welcome",
        "monkey",
        "football",
        "abc123",
        "1q2w3e4r",
        "123123",
        "qwertyuiop",
        "admin",
    ]
    strong_passwords = [
        "My$up3rS3cur3P@ssw0rd!",
        "G00dPa$$word!",
        "Str0ng!Password123",
        "P@ssw0rd2023$",
        "BetterP@ss123!",
        "Y0uC@nTrustM3!",
        "R@ndom$ecure!",
        "A8*jsL!0kPz@f3n#qT",  # 18 characters, fully random
        "7uRe$C@d0z&!12PqX3w",  # 20 characters, randomized
        "Z3@cLu9#Pq*Rt!Mx8v",  # 18 characters, mixed with special characters
        "F%8R$2n0wXy7!Pk@Tql",  # 20 characters, fully random
        "Jd@q3&12Pq$Mx7vF!9n",  # 18 characters, randomized
        "PzM7!rT2$Xq3Fj0k@wn",  # 18 characters, high randomness
        "L%F9!J3uZ#kP@Tq0wRn",  # 20 characters
        "R$2nF7w@Pq#J9!kXtMl",  # Fully randomized
        "3F!XqP$T7n@wkZ#9J2v",  # High entropy
        "TqR9!kF$P@3JwXn7Z2Ml",  # Another example
    ]

    # Write data to a CSV file
    with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        # Write header row
        writer.writerow(["password", "label"])

        # Write weak passwords with the label 'weak'
        for pw in weak_passwords:
            writer.writerow([pw, "weak"])

        # Write strong passwords with the label 'strong'
        for pw in strong_passwords:
            writer.writerow([pw, "strong"])

    print(f"Labeled passwords CSV created: {output_file}")


# Run the function when the script is executed
if __name__ == "__main__":
    create_labeled_passwords_csv()
