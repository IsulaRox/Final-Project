import os
import pandas as pd


# ==========================================
# 1. Dataset Paths
# ==========================================

train_path = os.path.join(
    "dataset",
    "BT-MRI Dataset",
    "BT-MRI Dataset",
    "Training"
)

test_path = os.path.join(
    "dataset",
    "BT-MRI Dataset",
    "BT-MRI Dataset",
    "Testing"
)

csv_path = "csv"

# Create csv folder if it doesn't exist
os.makedirs(csv_path, exist_ok=True)


# ==========================================
# 2. Check Dataset Paths
# ==========================================

print("\n========== DATASET PATHS ==========")

print("Training path:")
print(os.path.abspath(train_path))

print("\nTesting path:")
print(os.path.abspath(test_path))


if not os.path.exists(train_path):
    print("\nERROR: Training folder was not found!")
    print("Path:", os.path.abspath(train_path))
    exit()

if not os.path.exists(test_path):
    print("\nERROR: Testing folder was not found!")
    print("Path:", os.path.abspath(test_path))
    exit()


# ==========================================
# 3. Function to Create Dataset DataFrame
# ==========================================

def create_dataset_csv(dataset_path, dataset_type):

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
                    "dataset": dataset_type
                })

    return pd.DataFrame(data)


# ==========================================
# 4. Create Training Dataset
# ==========================================

train_df = create_dataset_csv(
    train_path,
    "train"
)

print("\n==========================================")
print("BT-MRI TRAINING DATA")
print("==========================================")

print("Training images:", len(train_df))

print("\nFirst 10 training records:")
print(train_df.head(10))


# ==========================================
# 5. Create Testing Dataset
# ==========================================

test_df = create_dataset_csv(
    test_path,
    "test"
)

print("\n==========================================")
print("BT-MRI TESTING DATA")
print("==========================================")

print("Testing images:", len(test_df))

print("\nFirst 10 testing records:")
print(test_df.head(10))


# ==========================================
# 6. Combine Training + Testing
# ==========================================

combined_df = pd.concat(
    [train_df, test_df],
    ignore_index=True
)


# ==========================================
# 7. Save Combined CSV
# ==========================================

combined_csv = os.path.join(
    csv_path,
    "BT-MRI_Dataset.csv"
)

combined_df.to_csv(
    combined_csv,
    index=False
)


# ==========================================
# 8. Dataset Information
# ==========================================

print("\n==========================================")
print("BT-MRI DATASET CREATED SUCCESSFULLY")
print("==========================================")

print("\nTotal images:", len(combined_df))
print("Training images:", len(train_df))
print("Testing images:", len(test_df))


# ==========================================
# 9. Class Distribution
# ==========================================

print("\n==========================================")
print("CLASS DISTRIBUTION")
print("==========================================")

print(
    combined_df["label"].value_counts()
)


# ==========================================
# 10. Training / Testing Distribution
# ==========================================

print("\n==========================================")
print("TRAIN / TEST DISTRIBUTION")
print("==========================================")

print(
    combined_df["dataset"].value_counts()
)


# ==========================================
# 11. First 10 Combined Records
# ==========================================

print("\n==========================================")
print("FIRST 10 COMBINED RECORDS")
print("==========================================")

print(
    combined_df.head(10)
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
# 13. CSV Shape
# ==========================================

print("\n==========================================")
print("CSV INFORMATION")
print("==========================================")

print("Rows:", combined_df.shape[0])
print("Columns:", combined_df.shape[1])

print("\nColumns:")
print(
    list(combined_df.columns)
)


# ==========================================
# 14. Check for Missing Values
# ==========================================

print("\n==========================================")
print("MISSING VALUES")
print("==========================================")

print(
    combined_df.isnull().sum()
)


# ==========================================
# 15. Final Message
# ==========================================

print("\n==========================================")
print("DONE!")
print("==========================================")

print(
    "\nBT-MRI CSV file successfully created:"
)

print(
    os.path.abspath(combined_csv)
)