import pandas as pd

# load cleaned dataset
df = pd.read_csv("data/cleaned_house.csv")

print("\n===== ROLL UP =====")

rollup = df.groupby("PriceCategory")["SalePrice"].mean()

print(rollup)


print("\n===== DRILL DOWN =====")

drill = df.groupby(
    ["Neighborhood","HouseStyle"]
)["SalePrice"].mean()

print(drill.head(10))


print("\n===== SLICE =====")

slice_data = df[
    df["PriceCategory"]=="Luxury"
]

print(slice_data[
    ["SalePrice","Neighborhood"]
].head())


print("\n===== DICE =====")

dice = df[
    (df["Neighborhood"]=="NAmes")
    &
    (df["PriceCategory"]=="High")
]

print(
    dice[
        ["SalePrice","Neighborhood","PriceCategory"]
    ].head()
)