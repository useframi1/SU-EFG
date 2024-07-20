import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import (
    classification_report,
    precision_score,
    recall_score,
    accuracy_score,
    f1_score,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers import Dense, Dropout
from xgboost import XGBClassifier


def _initialization(train_df: pd.DataFrame, test_df: pd.DataFrame):
    X_train = train_df.iloc[:, :-1]
    y_train = train_df.iloc[:, -1]
    X_test = test_df.iloc[:, :-1]
    y_test = test_df.iloc[:, -1]

    return X_train, y_train, X_test, y_test


def _K_Folds(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    KF: KFold,
    model,
    early_stopping: EarlyStopping = None,
):
    precisions = []
    recalls = []
    accuracies = []
    f1_scores = []
    fold = 1

    for train, test in KF.split(X_train.to_numpy(), y_train.to_numpy()):
        print(f"##### FOLD: {fold} #####")

        # Fit the model
        if type(model) == GaussianNB:
            model.fit(X_train.to_numpy()[train], y_train.to_numpy()[train])

            # Predict on the test set
            predictions = model.predict(X_train.to_numpy()[test])
        else:
            model.fit(
                X_train.to_numpy()[train],
                y_train.to_numpy()[train],
                epochs=200,
                batch_size=32,
                validation_data=(X_train.to_numpy()[test], y_train.to_numpy()[test]),
                callbacks=[early_stopping],
                verbose=0,
            )

            # Predict on the test set
            y_pred = model.predict(X_train.to_numpy()[test])

            predictions = []
            for pred in y_pred:
                if pred > 0.5:
                    predictions.append(1)
                else:
                    predictions.append(0)

        # Evaluate the model
        precision = precision_score(
            y_true=y_train.to_numpy()[test],
            y_pred=predictions,
            zero_division=0,
            average="weighted",
        )
        recall = recall_score(
            y_true=y_train.to_numpy()[test],
            y_pred=predictions,
            zero_division=0,
            average="weighted",
        )
        accuracy = accuracy_score(y_true=y_train.to_numpy()[test], y_pred=predictions)
        f1 = f1_score(
            y_true=y_train.to_numpy()[test],
            y_pred=predictions,
            zero_division=0,
            average="weighted",
        )

        # Store the result
        precisions.append(precision)
        recalls.append(recall)
        accuracies.append(accuracy)
        f1_scores.append(f1)

        fold += 1

    print("Mean Scores:")
    print(f"Mean Precision = {np.mean(precisions)}")
    print(f"Mean Recall = {np.mean(recalls)}")
    print(f"Mean Accuracy = {np.mean(accuracies)}")
    print(f"Mean F1 score = {np.mean(f1_scores)}")

    return np.mean(f1_scores)


def _Naive_Bayes(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model,
    KF: KFold,
    best_model,
    best_score: float,
):
    NB = model

    NB_score = _K_Folds(X_train=X_train, y_train=y_train, KF=KF, model=NB)

    predictions = NB.predict(X_test)
    print(classification_report(y_test, predictions, zero_division=0))

    return (NB_score, NB) if NB_score > best_score else (best_score, best_model)


def _Neural_Network(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model,
    KF: KFold,
    best_model,
    best_score: float,
):

    early_stopping = EarlyStopping(
        monitor="val_loss", patience=10, mode="min", verbose=1
    )

    nn = model
    nn.add(Dense(256, activation="relu"))
    nn.add(Dropout(0.2))
    nn.add(Dense(1, activation="sigmoid"))

    nn.compile(optimizer="adam", loss="binary_crossentropy")

    NN_score = _K_Folds(
        X_train=X_train, y_train=y_train, KF=KF, early_stopping=early_stopping, model=nn
    )

    y_pred = nn.predict(X_test)

    predictions = []
    for pred in y_pred:
        if pred > 0.5:
            predictions.append(1)
        else:
            predictions.append(0)

    print(classification_report(y_test, predictions, zero_division=0))

    return (NN_score, nn) if NN_score > best_score else (best_score, best_model)


def _train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    KF: KFold,
    model,
    param_grid,
    best_model,
    best_score: float,
):
    if type(model) not in [GaussianNB, Sequential]:
        grid_search = GridSearchCV(
            estimator=model, param_grid=param_grid, cv=KF, verbose=0
        )
        grid_search.fit(X_train, y_train)

        model_best = grid_search.best_estimator_

        predictions = model_best.predict(X_test)
        model_best_score = f1_score(
            y_true=y_test,
            y_pred=predictions,
            zero_division=0,
            average="weighted",
        )
        print(model_best_score)
        print(classification_report(y_true=y_test, y_pred=predictions, zero_division=0))

        return (
            (model_best_score, model_best)
            if model_best_score > best_score
            else (best_score, best_model)
        )

    elif type(model) == GaussianNB:
        return _Naive_Bayes(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            model=model,
            KF=KF,
            best_model=best_model,
            best_score=best_score,
        )

    else:
        return _Neural_Network(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            model=model,
            KF=KF,
            best_model=best_model,
            best_score=best_score,
        )


def get_best_model(train_df: pd.DataFrame, test_df: pd.DataFrame):
    best_model = None
    best_score = 0
    KF = KFold(n_splits=5)

    X_train, y_train, X_test, y_test = _initialization(
        train_df=train_df, test_df=test_df
    )

    models = [
        KNeighborsClassifier(),
        GaussianNB(),
        DecisionTreeClassifier(),
        RandomForestClassifier(n_jobs=5),
        XGBClassifier(random_state=42),
        LogisticRegression(n_jobs=5),
        SVC(),
        Sequential(),
    ]

    param_grids = [
        {"n_neighbors": list(range(1, int(np.sqrt(len(X_train))), 2))},
        {},
        {
            "max_depth": [None, 10, 20, 30],
            "class_weight": [
                None,
                {0: 1, 1: 2},
                {0: 1, 1: 3},
                {0: 1.5, 1: 2.5},
                {0: 1.5, 1: 3},
            ],
            "ccp_alpha": [0, 1, 0.01, 0.001],
            "min_samples_split": [2, 3, 4],
        },
        {
            "n_estimators": [100, 200],
            "max_depth": [10, 20, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "class_weight": ["balanced", "balanced_subsample", None],
        },
        {
            "n_estimators": [100, 200, 300],
            "learning_rate": [0.01, 0.1, 0.2],
            "max_depth": [3, 5, 7],
            "subsample": [0.7, 0.8, 0.9],
            "colsample_bytree": [0.7, 0.8, 0.9],
        },
        {
            "tol": [0.001, 0.0001, 0.00001],
            "solver": ["lbfgs", "liblinear"],
            "max_iter": [2000, 6000, 10000],
            "class_weight": [
                None,
                {0: 1, 1: 2},
                {0: 1, 1: 3},
                {0: 1.5, 1: 2.5},
                {0: 1.5, 1: 3},
            ],
        },
        {
            "tol": [0.001, 0.0001, 0.00001],
            "kernel": ["poly", "rbf", "sigmoid"],
            "class_weight": [
                None,
                {0: 1, 1: 2},
                {0: 1, 1: 2.5},
                {0: 1.5, 1: 2.5},
                {0: 1, 1: 3},
            ],
        },
        {},
    ]

    for i in range(len(models)):
        print(f"##### Training {models[i]} #####")
        best_score, best_model = _train_model(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            model=models[i],
            param_grid=param_grids[i],
            KF=KF,
            best_model=best_model,
            best_score=best_score,
        )
        print(f"Best score: {best_score}")
        print(f"Best model: {best_model}\n")

    return best_model
