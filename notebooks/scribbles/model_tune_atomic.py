import marimo

__generated_with = "0.23.16"
app = marimo.App()


@app.cell
def _():
    ''' IMPORTS '''
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    import xgboost
    import lightgbm

    from sklearn.metrics import (
        classification_report, 
        precision_score, 
        recall_score, 
        f1_score,
        roc_auc_score
    )

    import optuna
    import mlflow

    import polars as pl
    import time

    return (
        RandomForestClassifier,
        classification_report,
        f1_score,
        mlflow,
        optuna,
        pl,
        precision_score,
        recall_score,
        roc_auc_score,
        time,
        train_test_split,
    )


@app.cell
def _(pl, train_test_split):
    data = "../../data/processed/output.csv"
    df = pl.read_csv(data)

    # Prepare data
    X = df.drop('Churn')
    y = df['Churn']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    THRESHOLD = 0.3  # lower than 0.5 to boost recall (see next to choose the right value)
    return THRESHOLD, X_test, X_train, y_test, y_train


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Choosing the Optimal Model
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 1. Random Forest
    """)
    return


@app.cell
def _(
    RandomForestClassifier,
    THRESHOLD,
    X_test,
    X_train,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    time,
    y_test,
    y_train,
):
    rf = RandomForestClassifier(
        n_estimators=300,
        class_weight='balanced',   # handles imbalance for you
        random_state=42,
        n_jobs=-1
    )

    # Compute Training Time
    start_train_rf = time.time()
    rf.fit(X_train, y_train)
    train_time_rf = time.time() - start_train_rf
    print(f"Training time: {train_time_rf:.2f}s")

    # Compute Testing Time
    start_test_rf = time.time()
    probs1 = rf.predict_proba(X_test)[:, 1]
    y_pred = (probs1 >= THRESHOLD).astype(int)
    test_time_rf = time.time() - start_test_rf
    print(f"Testing time: {test_time_rf:.2f}s")


    print(classification_report(y_test, y_pred, digits=3))

    # threshold tuning
    print("Threshold tuning for RandomForest")
    print(f"{'S.no':<8}{'Thresh':<8}{'Precision':<12}{'Recall':<8}{'F1-Score':<8}")
    for _idx,_thresh in enumerate([0.25, 0.30, 0.35, 0.40, 0.45, 0.50]):
        _preds = (probs1 >= _thresh).astype(int)
        _prec = precision_score(y_test, _preds, pos_label=1)
        _rec = recall_score(y_test, _preds, pos_label=1)
        _f1 = f1_score(y_test, _preds, pos_label=1)
        print(f"{_idx:<8}{_thresh:<8}{_prec:<12.3f}{_rec:<8.3f}{_f1:<8.3f}")
    return (y_pred,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    we see that random forest is able to catch upto 88.5% of actual churners (refer to Recall).
    However, the model still takes 100ms to actually return the inference, which is quite inefficient.

    Thereby, we can use grad-boosting algorithms to improve on this.

    XGBoost and LightGBM are two such Gradient Boosting Algorithms which we'll test.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 2. LightGBM
    """)
    return


@app.cell
def _(
    THRESHOLD,
    X_test,
    X_train,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    time,
    y_test,
    y_train,
):
    from lightgbm import LGBMClassifier

    lg = LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        class_weight='balanced',
        n_jobs=-1,
        random_state=42
    )



    _start_train = time.perf_counter()
    lg.fit(X_train, y_train)
    _train_time = time.perf_counter() - _start_train

    _start_test = time.perf_counter()
    _probs = lg.predict_proba(X_test)[:, 1]
    _y_preds = (_probs >= THRESHOLD).astype(int)   # same threshold as RF for fair comparison
    _test_time = time.perf_counter() - _start_test

    print(f"Training time: {_train_time:.4f}s")
    print(f"Testing time: {_test_time:.4f}s")
    print(classification_report(y_test, _y_preds, digits=3))

    # threshold tuning
    print("Threshold tuning for LightGBM")
    print(f"{'S.no':<8}{'Thresh':<8}{'Precision':<12}{'Recall':<8}{'F1-Score':<8}")
    for _idx,_thresh in enumerate([0.15, 0.20, 0.25, 0.30, 0.35, 0.40]):
        _preds = (_probs >= _thresh).astype(int)
        _prec = precision_score(y_test, _preds, pos_label=1)
        _rec = recall_score(y_test, _preds, pos_label=1)
        _f1 = f1_score(y_test, _preds, pos_label=1)
        print(f"{_idx:<8}{_thresh:<8}{_prec:<12.3f}{_rec:<8.3f}{_f1:<8.3f}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    we can thus infer that LightGBM is much faster as compared to RF, and also gives the similar performance with a loss in recall ranging from 1-2% and an decrease in inference time by >100ms
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 3. XGBoost
    """)
    return


@app.cell
def _(
    THRESHOLD,
    X_test,
    X_train,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    time,
    y_test,
    y_train,
):
    from xgboost import XGBClassifier

    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    print("The pos_weight is:", scale_pos_weight)

    xg = XGBClassifier(
        n_estimators =  500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        scale_pos_weight=scale_pos_weight,
        eval_metric='logloss'
    )

    _start_train = time.perf_counter()
    xg.fit(X_train, y_train)
    _train_time = time.perf_counter() - _start_train

    _start_test = time.perf_counter()
    _probs = xg.predict_proba(X_test)[:, 1]
    _y_preds = (_probs >= THRESHOLD).astype(int)   # same threshold as RF for fair comparison
    _test_time = time.perf_counter() - _start_test

    print(f"Training time: {_train_time:.4f}s")
    print(f"Testing time: {_test_time:.4f}s")
    print(classification_report(y_test, _y_preds, digits=3))

    # threshold tuning
    print("Threshold tuning for XGBoost")
    print(f"{'S.no':<8}{'Thresh':<8}{'Precision':<12}{'Recall':<8}{'F1-Score':<8}")
    for _idx,_thresh in enumerate([0.15, 0.20, 0.25, 0.30, 0.35, 0.40]):
        _preds = (_probs >= _thresh).astype(int)
        _prec = precision_score(y_test, _preds, pos_label=1)
        _rec = recall_score(y_test, _preds, pos_label=1)
        _f1 = f1_score(y_test, _preds, pos_label=1)
        print(f"{_idx:<8}{_thresh:<8}{_prec:<12.3f}{_rec:<8.3f}{_f1:<8.3f}")


    return XGBClassifier, scale_pos_weight


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    xgboost is the fastest of the three, offering the same performance as LightGBM while being >50% faster
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Model Hyperparam Tuning
    The selected model is thus (shockingly) XGBoost
    """)
    return


@app.cell
def _(
    THRESHOLD,
    XGBClassifier,
    X_test,
    X_train,
    optuna,
    recall_score,
    y_test,
    y_train,
):
    # We use optuna for HyperParameter Tuning


    # Objective function for Optuna wherein -> we feeding in the start and end ranges for each param we want to tune
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 300, 800),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 0, 5),
            "reg_alpha": trial.suggest_float("reg_alpha", 0, 5),
            "reg_lambda": trial.suggest_float("reg_lambda", 0, 5),
            "random_state": 42,
            "n_jobs": -1,
            "scale_pos_weight": (y_train == 0).sum() / (y_train == 1).sum(),
            "eval_metric": "logloss"
        }
    
        model = XGBClassifier(**params)
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)[:, 1]
        y_pred = (proba >= THRESHOLD).astype(int)  # Keep your tuned threshold
        return recall_score(y_test, y_pred, pos_label=1)  # Optimize recall for churners

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=30)
    return (study,)


@app.cell
def _(study):
    print("Best Params for the Model: ", study.best_params)
    print("Best Recall Value Obtained: ", study.best_value)
    return


@app.cell
def _(
    THRESHOLD,
    XGBClassifier,
    X_test,
    X_train,
    classification_report,
    scale_pos_weight,
    study,
    time,
    y_test,
    y_train,
):
    # Updating our Model in adherance to the best params

    _scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    # Get the Best Parameters obtained from Optuna
    _best_params = study.best_params
    _best_params.update({
        "random_state": 42,
        "n_jobs": -1,
        "scale_pos_weight": scale_pos_weight,
        "eval_metric": "logloss"
    })

    xg_opt = XGBClassifier(**_best_params)

    # Training
    _start_train = time.perf_counter()
    xg_opt.fit(X_train, y_train)
    _train_time = time.perf_counter() - _start_train

    # Testing
    _start_test = time.perf_counter()
    _probs = xg_opt.predict_proba(X_test)[:, 1]
    _y_preds = (_probs >= THRESHOLD).astype(int)   # same threshold as RF for fair comparison
    _test_time = time.perf_counter() - _start_test

    # Outputs
    print(f"Training time: {_train_time:.4f}s")
    print(f"Testing time: {_test_time:.4f}s")
    print(classification_report(y_test, _y_preds, digits=3))

    return


@app.cell
def _(
    THRESHOLD,
    XGBClassifier,
    X_test,
    X_train,
    classification_report,
    f1_score,
    mlflow,
    precision_score,
    recall_score,
    roc_auc_score,
    study,
    time,
    y_pred,
    y_test,
    y_train,
):
    # We use MLFlow here to track our Model Runs
    import os

    project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    mlflow.set_tracking_uri(f"file://{project_root}/mlruns")
    mlflow.set_experiment("Telco Churn Runs using XGBoost")

    with mlflow.start_run():
        _scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

        # from optuna:
        best_params = study.best_params
        best_params_mlf = {
        **study.best_params, 
        "random_state": 42, "n_jobs": -1, 
        "scale_pos_weight": _scale_pos_weight, 
        "eval_metric": "logloss"
        }

        mlflow.log_params(best_params_mlf)

        xg_mlf = XGBClassifier(**best_params_mlf)

        # Training
        start_train = time.perf_counter()
        xg_mlf.fit(X_train, y_train)
        train_time = time.perf_counter() - start_train

        # Testing
        start_test = time.perf_counter()
        probs = xg_mlf.predict_proba(X_test)[:, 1]
        y_preds = (probs >= THRESHOLD).astype(int)
        test_time = time.perf_counter() - start_test

        # Outputs
        mlflow.log_metric("train_time", train_time)
        mlflow.log_metric("test_time", test_time)

        # Metrics
        precision = precision_score(y_test, y_preds, pos_label=1)
        recall = recall_score(y_test, y_preds, pos_label=1)
        f1 = f1_score(y_test, y_preds, pos_label=1)
        auc = roc_auc_score(y_test, probs)

        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1", f1)
        mlflow.log_metric("roc_auc", auc)

        # Save model
        mlflow.xgboost.log_model(xg_mlf, "model")
        print(classification_report(y_test, y_pred, digits=3))
    return


if __name__ == "__main__":
    app.run()
