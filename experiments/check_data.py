#STEP 2
import pandas as pd

# Load dataset
df = pd.read_csv("data/resume_dataset.csv")

print("Dataset Shape:", df.shape)
print("\nColumn Names:")
print(df.columns)

print("\nFirst 3 Rows:")
print(df.head(3))

# If Category column exists
if 'Category' in df.columns:
    print("\nUnique Categories:")
    print(df['Category'].unique())
    print("\nNumber of Categories:", df['Category'].nunique())
