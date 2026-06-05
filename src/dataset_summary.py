# Week 2 Day 5: Dataset summary for SignAssist CV
# Goal:
# 1. Load the local landmark CSV dataset
# 2. Count total samples
# 3. Count samples per gesture label
# 4. Check the number of columns

import os
import pandas as pd


DATASET_PATH = "data/gesture_landmarks.csv"


def main():
    if not os.path.exists(DATASET_PATH):
        print(f"Dataset not found: {DATASET_PATH}")
        print("Run src/collect_landmark_data.py first to collect samples.")
        return

    df = pd.read_csv(DATASET_PATH)

    print("Dataset summary")
    print("----------------")
    print(f"Dataset path: {DATASET_PATH}")
    print(f"Total samples: {len(df)}")
    print(f"Total columns: {len(df.columns)}")

    print("\nSamples per label:")
    print(df["label"].value_counts())

    expected_columns = 1 + (21 * 3)

    print("\nColumn check:")
    print(f"Expected columns: {expected_columns}")
    print(f"Actual columns: {len(df.columns)}")

    if len(df.columns) == expected_columns:
        print("Column count looks correct.")
    else:
        print("Warning: Column count does not match expected landmark format.")

    print("\nFirst 3 rows:")
    print(df.head(3))


if __name__ == "__main__":
    main()