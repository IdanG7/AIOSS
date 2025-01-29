import csv
import random
import secrets
import string


def generate_strong_passwords(num_passwords, length=16):
    """
    Generate a list of strong passwords with random characters.
    :param num_passwords: Number of passwords to generate.
    :param length: Length of each password.
    :return: List of strong passwords.
    """
    strong_passwords = []
    all_characters = (
        string.ascii_letters + string.digits + "!@#$%^&*()-_=+[]{}|;:<>,.?/"
    )
    for _ in range(num_passwords):
        password = "".join(secrets.choice(all_characters) for _ in range(length))
        strong_passwords.append(password)
    return strong_passwords


def process_datasets(weak_files, num_weak_samples, strong_file, output_csv):
    """
    Combine weak and strong passwords into a labeled CSV file.
    :param weak_files: List of paths to weak password files.
    :param num_weak_samples: Number of weak passwords to sample from each file.
    :param strong_file: File containing strong passwords (generated or provided).
    :param output_csv: Path to output the labeled CSV file.
    """
    weak_passwords = set()

    # Read and sample weak passwords from each file
    for file in weak_files:
        with open(file, "r", encoding="utf-8", errors="ignore") as f:
            passwords = [line.strip() for line in f]
            sampled_passwords = random.sample(
                passwords, min(num_weak_samples, len(passwords))
            )
            weak_passwords.update(sampled_passwords)

    # Generate or load strong passwords
    strong_passwords = generate_strong_passwords(len(weak_passwords))

    # Write combined dataset to a CSV
    with open(output_csv, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["password", "label"])  # Header

        # Write weak passwords
        for password in weak_passwords:
            writer.writerow([password, "weak"])

        # Write strong passwords
        for password in strong_passwords:
            writer.writerow([password, "strong"])

    print(f"Labeled dataset saved to {output_csv}")


# Example Usage
if __name__ == "__main__":
    # Paths to your weak password files (Top 1M and RockYou)
    weak_files = ["data/top1000000.txt", "data/RockYou.txt"]

    # Number of weak passwords to sample from each file
    num_weak_samples = 100000  # Adjust this based on balance needs

    # Output CSV file
    output_csv = "data/passwords_labeled.csv"

    # Process datasets and generate labeled CSV
    process_datasets(weak_files, num_weak_samples, None, output_csv)
