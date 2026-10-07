import os
import pandas as pd


# ==========================================
# 1. Dataset Paths
# ==========================================

dataset_path = os.path.join(
    "dataset",
    "Final Brain Tumor Dataset v2 N20260529"
)

csv_path = "csv"

# CSV file name
csv_file_name = "Final_Brain_Tumor_Dataset_v2.csv"

# Create CSV folder if it doesn't exist
os.makedirs(csv_path, exist_ok=True)


# ==========================================
# 2. Check Dataset Path
# ==========================================

print("\n==========================================")
print("DATASET PATH")
print("==========================================")

print(
    os.path.abspath(dataset_path)
)


if not os.path.exists(dataset_path):

    print("\nERROR: Dataset folder was not found!")

    print(
        "Path:",
        os.path.abspath(dataset_path)
    )

    exit()


# ==========================================
# 3. Function to Create Dataset DataFrame
# ==========================================

def create_dataset_csv(dataset_path):

    data = []

    # Get class folders
    for class_name in os.listdir(dataset_path):

        class_path = os.path.join(
            dataset_path,
            class_name
        )

        # Skip files
        if not os.path.isdir(class_path):
            continue

        # Get images inside class folder
        for file_name in os.listdir(class_path):

            # Check image file extensions
            if file_name.lower().endswith(
                (".jpg", ".jpeg", ".png")
            ):

                image_path = os.path.join(
                    class_path,
                    file_name
                )

                data.append({
                    "image_path": image_path,
                    "label": class_name,
                    "dataset": "dataset3"
                })

    return pd.DataFrame(data)


# ==========================================
# 4. Create Dataset DataFrame
# ==========================================

dataset_df = create_dataset_csv(
    dataset_path
)


# ==========================================
# 5. Display Dataset Information
# ==========================================

print("\n==========================================")
print("FINAL BRAIN TUMOR DATASET")
print("==========================================")

print(
    "Total images:",
    len(dataset_df)
)


# ==========================================
# 6. First 10 Records
# ==========================================

print("\n==========================================")
print("FIRST 10 RECORDS")
print("==========================================")

print(
    dataset_df.head(10)
)


# ==========================================
# 7. Class Distribution
# ==========================================

print("\n==========================================")
print("CLASS DISTRIBUTION")
print("==========================================")

print(
    dataset_df["label"].value_counts()
)


# ==========================================
# 8. Dataset Distribution
# ==========================================

print("\n==========================================")
print("DATASET DISTRIBUTION")
print("==========================================")

print(
    dataset_df["dataset"].value_counts()
)


# ==========================================
# 9. Save CSV
# ==========================================

combined_csv = os.path.join(
    csv_path,
    csv_file_name
)

dataset_df.to_csv(
    combined_csv,
    index=False
)


# ==========================================
# 10. CSV Information
# ==========================================

print("\n==========================================")
print("CSV INFORMATION")
print("==========================================")

print(
    "Rows:",
    dataset_df.shape[0]
)

print(
    "Columns:",
    dataset_df.shape[1]
)

print("\nColumns:")

print(
    list(dataset_df.columns)
)


# ==========================================
# 11. Check Missing Values
# ==========================================

print("\n==========================================")
print("MISSING VALUES")
print("==========================================")

print(
    dataset_df.isnull().sum()
)


# ==========================================
# 12. CSV File Location
# ==========================================

print("\n==========================================")
print("CSV FILE LOCATION")
print("==========================================")

print(
    os.path.abspath(combined_csv)
)


# ==========================================
# 13. Final Message
# ==========================================

print("\n==========================================")
print("DATASET CSV CREATED SUCCESSFULLY")
print("==========================================")

print(
    "\nCSV file:"
)

print(
    os.path.abspath(combined_csv)
)

print("\nDone!")