import marimo

__generated_with = "0.23.16"
app = marimo.App()


@app.cell
def _():
    ''' IMPORTS '''
    import marimo as mo
    import polars as pl
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    from pathlib import Path

    return Path, mo, pl, plt, sns, variance_inflation_factor


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Scribble Time
    """)
    return


@app.cell
def _(pl):
    raw_data = "../../data/raw/Telco-Churn.csv"
    raw_df = pl.read_csv(raw_data)

    raw_df
    return (raw_df,)


@app.cell
def _(raw_df):
    raw_df.head(1)
    return


@app.cell
def _(pl, raw_df):
    # shape -> (7043, 21)

    # all datatypes
    pl.DataFrame({"column": raw_df.columns, "dtype": [str(d) for d in raw_df.dtypes]})
    return


@app.cell
def _(raw_df):
    raw_df.describe(())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    - based on this data we can see most datas have 2-4 categories (at most)
    - thereby we use a mix of using 0-1 and One-Hot Encoding to make it more uniform
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Encodings
    """)
    return


@app.cell
def _(pl, raw_df):
    # 1. Binary (0-1) Encoding
    cols = [
        "gender",
        "Partner",
        "Dependents",
        "PhoneService",
        "PaperlessBilling",
        "Churn",
    ]
    binary_df = raw_df.with_columns(
        pl.col(cols).replace({"Yes": 1, "No": 0, "Male" : 1, "Female" : 0}).cast(pl.Int8)
    )

    # 2. One Hot Encoding
    multi_cat_cols = [
        'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
        'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
        'Contract', 'PaymentMethod'
    ]

    encoded_df = binary_df.to_dummies(multi_cat_cols)

    encoded_df.head()
    return (encoded_df,)


@app.cell
def _(encoded_df, pl):
    # Data Cleaning
    clean_df = (
        encoded_df
        .with_columns(
            pl.col("TotalCharges").cast(pl.Float16)
        )
        .drop("customerID")
    )

    print(clean_df.schema)
    return (clean_df,)


@app.cell
def _(clean_df, plt, sns):
    # Inspecting Correlation Matrix to check for Corr

    pandas_df = clean_df.to_pandas()

    corr_matrix = pandas_df.corr(numeric_only=True)
    # churn_corr = corr_matrix['Churn'].sort_values(ascending=False)
    # print(churn_corr)

    churn_corr = corr_matrix[['Churn']].sort_values(by='Churn', ascending=False)
    # Plot heatmap
    plt.figure(figsize=(4, 12))
    sns.heatmap(churn_corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Correlation of Features with Churn')
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Conclusions

    Negative (less likely to churn):
    - tenure (-0.35) → Customers who are retained are less likely to churn.
    - Contract_Two year (-0.3) → Customers on long-term subscriptions churn much less.
    - Contract_One year (-0.18) → Weaker than two-year contracts.

    Positive (more likely to churn):
    - InternetService_Fiber optic (+0.31) → Fiber optic users are likely to churn more.
    - PaymentMethod_Electronic check (+0.30) → Customers paying via e-checks churn more.

    ---
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Checking for Multi-Collinearity using VIF

    **VIF**: The variance inflation factor (VIF) measures how much the variance of an estimated regression coefficient increases because your predictor variables are correlated.
    """)
    return


@app.cell
def _(clean_df, pl, variance_inflation_factor):
    # Collapse redundant columns
    internet_service_cols = [
        "OnlineSecurity_No internet service",
        "OnlineBackup_No internet service",
        "DeviceProtection_No internet service",
        "TechSupport_No internet service",
        "StreamingTV_No internet service",
        "StreamingMovies_No internet service",
    ]

    X = (
        clean_df
        .with_columns(
            pl.any_horizontal(
                pl.col(internet_service_cols)
            )
            .cast(pl.Int8)
            .alias("No_internet_service"),

            pl.col("MultipleLines_No phone service")
            .cast(pl.Int8)
            .alias("No_phone_service"),
        )
        .drop(
            internet_service_cols + [
                "MultipleLines_No phone service"
            ]
        )
    )

    # Prepare X
    X = X.drop([
        "Churn",

        # Perfect redundancy
        "PhoneService",

        # Reference categories for one-hot groups
        "InternetService_No",
        "OnlineSecurity_No",
        "OnlineBackup_No",
        "DeviceProtection_No",
        "TechSupport_No",
        "StreamingTV_No",
        "StreamingMovies_No",
        "Contract_Month-to-month",
        "PaymentMethod_Mailed check",
    ])

    # Remove NaN / -inf / +inf
    X = (
        X
        .with_columns(
            pl.col(pl.Float32, pl.Float16)
            .replace(
                [float("inf"), float("-inf")],
                None
            )
        )
        .drop_nulls()
    )


    # Convert to NumPy for statsmodels

    X_np = X.to_numpy()

    # Run VIF
    vif_data = pl.DataFrame({
        "feature": X.columns,
        "VIF": [
            variance_inflation_factor(X_np, i)
            for i in range(X_np.shape[1])
        ],
    }).sort(
        "VIF",
        descending=True
    )

    vif_data
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    as there is multi collinerity between the features -
    1. We might use tree based models (LightGBM, XGBoost, RF Classifiers)
    2. We might use L1/L2 Regularization
    """)
    return


@app.cell
def _(clean_df):
    clean_df["Churn"].value_counts()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    - As this is a churn prediction model, we shall primarily focus on the recall (i.e. improvising the retention of the customers)
    - However, we evaluate the ROC-AUC and PR-AUC if our business wants a ranking of churn risk
    """)
    return


@app.cell
def _(Path, clean_df):

    target_dir = Path.cwd().parent.parent/"data"/"processed"
    output_file = target_dir / "output.csv"

    clean_df.write_csv(output_file)
    return


if __name__ == "__main__":
    app.run()
