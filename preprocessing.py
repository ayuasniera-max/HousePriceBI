import pandas as pd

# Load dataset
df = pd.read_csv("data/train.csv")

print("Original shape:")
print(df.shape)

# Missing values check
missing = df.isnull().sum()

print("\nTop missing values:")
print(missing.sort_values(ascending=False).head(10))

# Remove columns with too many missing values
threshold = len(df)*0.5

df = df.dropna(thresh=threshold, axis=1)

# Fill numerical missing values
num_cols = df.select_dtypes(include=['int64','float64']).columns

for col in num_cols:
    df[col] = df[col].fillna(df[col].median())

# Fill categorical values
cat_cols = df.select_dtypes(include=['object']).columns

for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# Feature Engineering
currentYear = 2026

df['HouseAge'] = currentYear - df['YearBuilt']

# Price Category
df['PriceCategory'] = pd.cut(
    df['SalePrice'],
    bins=[0,100000,200000,400000,1000000],
    labels=['Low','Medium','High','Luxury']
)

print("\nNew shape:")
print(df.shape)

print("\nPreview:")
print(df[['SalePrice','HouseAge','PriceCategory']].head())

# save cleaned data
df.to_csv("data/cleaned_house.csv",index=False)

print("\nSaved as cleaned_house.csv")