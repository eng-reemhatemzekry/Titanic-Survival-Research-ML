import pandas as pd

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    # Title extraction
    df["Title"] = df["Name"].str.extract(" ([A-Za-z]+)\.", expand=False)

    # Family features
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)

    # Cabin info
    df["Deck"] = df["Cabin"].astype(str).str[0]
    df["Deck"] = df["Deck"].replace("n", "U")

    # Fare per person
    df["FarePerPerson"] = df["Fare"] / df["FamilySize"]

    # Ticket group size
    df["TicketGroup"] = df.groupby("Ticket")["Ticket"].transform("count")

    return df
