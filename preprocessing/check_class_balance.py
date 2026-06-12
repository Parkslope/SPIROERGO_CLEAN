import pandas as pd

# Read the CSV file
df = pd.read_csv('new_subgroup_classified.csv', sep=';')

# Display basic info
print("Dataset Overview:")
print(f"Total samples: {len(df)}")
print(f"\nColumns: {df.columns.tolist()}")

# Check class distribution
print("\n" + "="*50)
print("CLASS BALANCE ANALYSIS")
print("="*50)

class_counts = df['binary_class'].value_counts().sort_index()
print(f"\nClass distribution:")
for class_label, count in class_counts.items():
    percentage = (count / len(df)) * 100
    print(f"  Class {class_label}: {count} samples ({percentage:.2f}%)")

print(f"\nClass ratio (Class 0 : Class 1): {class_counts[0]}:{class_counts[1]}")
print(f"Imbalance ratio: {max(class_counts) / min(class_counts):.2f}:1")

# Check split distribution
print("\n" + "="*50)
print("SPLIT DISTRIBUTION")
print("="*50)
print(df['split'].value_counts())

# Check fold distribution
print("\n" + "="*50)
print("FOLD DISTRIBUTION")
print("="*50)
print("\nTest Folds:")
print(df['fold_test'].value_counts().sort_index())
print("\nValidation Folds:")
print(df['fold_val'].value_counts().sort_index())
