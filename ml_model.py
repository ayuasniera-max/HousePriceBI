import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# Load dataset
df = pd.read_csv("data/cleaned_house.csv")

# Features
X = df[
    [
        "GrLivArea",
        "GarageCars",
        "BedroomAbvGr",
        "OverallQual",
        "HouseAge"
    ]
]

# Target
y = df["SalePrice"]

# Train model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X, y)