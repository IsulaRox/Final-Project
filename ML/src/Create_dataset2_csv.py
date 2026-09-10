import os
import pandas as pd


# ==========================================
# 1. Dataset Paths
# ==========================================

train_path = "dataset/Training"
test_path = "dataset/Testing"

csv_path = "csv"

# Create csv folder if it doesn't exist
os.makedirs(csv_path, exist_ok=True)


# ==========================================
# 2. Function to Create Dataset DataFrame
# ==========================================

def create_dataset_csv(dataset_path, dataset_type):

    data = []

    for class_name in os.listdir(dataset_path):

        class_path = os.path.join(dataset_path, class_name)

        if not os.path.isdir(class_path):
            continue

        for file_name in os.listdir(class_path):

            if file_name.lower().endswith((".jpg", ".jpeg", ".png")):

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
# 3. Create Training Dataset
# ==========================================

train_df = create_dataset_csv(
    train_path,
    "train"
)

print("\n========== TRAINING DATA ==========")
print("Training images:", len(train_df))
print(train_df.head(10))


# ==========================================
# 4. Create Testing Dataset
# ==========================================

test_df = create_dataset_csv(
    test_path,
    "test"
)

print("\n========== TESTING DATA ==========")
print("Testing images:", len(test_df))
print(test_df.head(10))


# ==========================================
# 5. Combine Training + Testing
# ==========================================

combined_df = pd.concat(
    [train_df, test_df],
    ignore_index=True
)


# ==========================================
# 6. Save Combined CSV as dataset1.csv
# ==========================================

combined_csv = os.path.join(
    csv_path,
    "dataset1.csv"
)

combined_df.to_csv(
    combined_csv,
    index=False
)


# ==========================================
# 7. Dataset Information
# ==========================================

print("\n==========================================")
print("DATASET CREATED SUCCESSFULLY")
print("==========================================")

print("Total images:", len(combined_df))
print("Training images:", len(train_df))
print("Testing images:", len(test_df))


# ==========================================
# 8. Class Distribution
# ==========================================

print("\n========== CLASS DISTRIBUTION ==========")

print(combined_df["label"].value_counts())


# ==========================================
# 9. Train / Test Distribution
# ==========================================

print("\n========== TRAIN / TEST DISTRIBUTION ==========")

print(combined_df["dataset"].value_counts())


# ==========================================
# 10. First 10 Combined Records
# ==========================================

print("\n========== FIRST 10 RECORDS ==========")

print(combined_df.head(10))


# ==========================================
# 11. CSV File Location
# ==========================================

print("\n========== CSV FILE LOCATION ==========")

print(os.path.abspath(combined_csv))