import os
import pandas as pd


# ==========================================
# 1. Dataset Paths
# ==========================================

train_path = "datasset/train"
test_path = "datasset/test"

csv_path = "csv"

# Create CSV folder if it doesn't exist
os.makedirs(csv_path, exist_ok=True)


# ==========================================
# 2. Function to Create Dataset DataFrame
# ==========================================

def create_dataset_csv(dataset_path, dataset_type):

    data = []

    for class_name in os.listdir(dataset_path):

        class_path = os.path.join(dataset_path, class_name)

        # Make sure it is a folder
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
# 3. Create Train Dataset
# ==========================================

train_df = create_dataset_csv(
    train_path,
    "train"
)

print("\nFirst 10 TRAIN records:")
print(train_df.head(10))


# ==========================================
# 4. Create Test Dataset
# ==========================================

test_df = create_dataset_csv(
    test_path,
    "test"
)

print("\nFirst 10 TEST records:")
print(test_df.head(10))


# ==========================================
# 5. Combine Train + Test
# ==========================================

combined_df = pd.concat(
    [train_df, test_df],
    ignore_index=True
)


# ==========================================
# 6. Save Combined CSV
# ==========================================

combined_csv = os.path.join(
    csv_path,
    "dataset.csv"
)

combined_df.to_csv(
    combined_csv,
    index=False
)


# ==========================================
# 7. Display Dataset Information
# ==========================================

print("\n==========================================")
print("Dataset created successfully!")
print("==========================================")

print("\nTotal images:", len(combined_df))

print("Train images:", len(train_df))

print("Test images:", len(test_df))


# ==========================================
# 8. Class Distribution
# ==========================================

print("\nClass distribution:")

print(
    combined_df["label"].value_counts()
)


# ==========================================
# 9. Display First 10 Combined Records
# ==========================================

print("\nFirst 10 combined records:")

print(
    combined_df.head(10)
)


# ==========================================
# 10. CSV File Location
# ==========================================

print("\nCSV file created at:")

print(os.path.abspath(combined_csv))