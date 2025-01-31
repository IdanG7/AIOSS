import csv
import secrets
import string


def generate_strong_password(min_length=5):
    """
    Generate a single strong password with random length between min_length and 100.
    :param min_length: Minimum length of the password.
    :return: A strong password.
    """
    # Generate a random length between min_length and 100
    length = secrets.randbelow(96) + min_length  # This gives range 5-100

    all_characters = (
        string.ascii_letters + string.digits + "!@#$%^&*()-_=+[]{}|;:<>,.?/"
    )

    # Ensure we have at least one of each required type
    password = [
        secrets.choice(string.ascii_lowercase),  # At least one lowercase
        secrets.choice(string.ascii_uppercase),  # At least one uppercase
        secrets.choice(string.digits),  # At least one number
        secrets.choice("!@#$%^&*()-_=+[]{}|;:<>,.?/"),  # At least one special char
    ]

    # Fill the rest with random characters
    remaining_length = length - len(password)
    password.extend(secrets.choice(all_characters) for _ in range(remaining_length))

    # Shuffle the password
    password_list = list(password)
    secrets.SystemRandom().shuffle(password_list)
    return "".join(password_list)


def process_datasets(weak_files, output_csv):
    """
    Process all weak passwords and generate matching number of strong passwords.
    :param weak_files: List of paths to weak password files.
    :param output_csv: Path to output the labeled CSV file.
    """
    # Read all weak passwords
    weak_passwords = set()
    for file in weak_files:
        try:
            with open(file, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    password = line.strip()
                    if password:  # Skip empty lines
                        weak_passwords.add(password)
        except Exception as e:
            print(f"Error reading file {file}: {str(e)}")

    print(f"Total unique weak passwords found: {len(weak_passwords)}")

    # Generate matching number of strong passwords
    strong_passwords = set()
    while len(strong_passwords) < len(weak_passwords):
        strong_passwords.add(generate_strong_password())

    # Write combined dataset to CSV
    with open(output_csv, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["password", "label"])  # Header

        # Write all passwords
        for password in weak_passwords:
            writer.writerow([password, "weak"])
        for password in strong_passwords:
            writer.writerow([password, "strong"])

    print(f"Created balanced dataset with {len(weak_passwords)} passwords of each type")
    print(f"Labeled dataset saved to {output_csv}")


# Example Usage
if __name__ == "__main__":
    # Paths to your weak password files
    weak_files = ["data/top1000000.txt", "data/RockYou.txt"]

    # Output CSV file
    output_csv = "data/passwords_labeled.csv"

    # Process datasets and generate labeled CSV
    process_datasets(weak_files, output_csv)
