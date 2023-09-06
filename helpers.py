import pandas as pd
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from config import *


def loading_data():
    scoutium_att = pd.read_csv(attribute_path, sep = ";")
    scoutium_pot = pd.read_csv(label_path, sep = ";")
    df = pd.merge(scoutium_att, scoutium_pot, how = "inner", on = ["task_response_id", "match_id", "evaluator_id", "player_id"])
    return df

def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

def one_hot_encoder(dataframe, categorical_cols, drop_first = True):
    dataframe = pd.get_dummies(dataframe, columns = categorical_cols, drop_first = drop_first)
    return dataframe

def data_processing(dataframe):
    temp_df = dataframe.copy()
    temp_df.columns = [col.upper() for col in temp_df.columns]
    temp_df = temp_df[~(temp_df["POSITION_ID"] == 1)].reset_index(drop = True)
    temp_df = temp_df[~(temp_df["POTENTIAL_LABEL"] == "below_average")].reset_index(drop = True)

    new_df = temp_df.pivot_table(values = "ATTRIBUTE_VALUE",
                                 index = ["PLAYER_ID", "POSITION_ID", "POTENTIAL_LABEL"],
                                 columns = "ATTRIBUTE_ID").reset_index()

    new_df.columns = [str(col) for col in new_df.columns]

    att_cols = new_df.iloc[:, 3:].columns
    new_df["TOTAL_SCORE"] = new_df[att_cols].sum(axis = 1) / len(att_cols)

    label_encoder(new_df, "POTENTIAL_LABEL")

    new_df = one_hot_encoder(new_df, ["POSITION_ID"])

    new_df.columns = [col.upper() for col in new_df.columns]

    num_cols = [col for col in new_df.columns if new_df[col].dtypes == "float64"]

    ss = StandardScaler()
    for col in num_cols:
        new_df[col] = ss.fit_transform(new_df[[col]])

    y = new_df["POTENTIAL_LABEL"]
    X = new_df.drop(columns = ["POTENTIAL_LABEL", "PLAYER_ID"])

    return X, y

def base_models(X, y, scoring = "roc_auc"):
    print("Base Models....")
    classifiers = [('LR', LogisticRegression()),
                   ('KNN', KNeighborsClassifier()),
                   ("SVC", SVC(random_state = random_state)),
                   ("CART", DecisionTreeClassifier(random_state = random_state)),
                   ("RF", RandomForestClassifier(random_state = random_state)),
                   ('Adaboost', AdaBoostClassifier(random_state = random_state)),
                   ('GBM', GradientBoostingClassifier(random_state = random_state)),
                   ('XGBoost', XGBClassifier(eval_metric = 'logloss', random_state = random_state)),
                   ('LightGBM', LGBMClassifier(random_state = random_state)),
                   ('CatBoost', CatBoostClassifier(verbose=False, random_state = random_state))
                   ]

    for name, classifier in classifiers:
        cv_results = cross_validate(classifier, X, y, cv = 3, scoring = scoring)
        print(f"{scoring}: {round(cv_results['test_score'].mean(), 4)} ({name}) ")

def hyperparameter_optimization(X, y, cv=3, scoring="roc_auc"):
    print("Hyperparameter Optimization....")
    best_models = {}
    for name, classifier, params in classifiers:
        print(f"########## {name} ##########")
        cv_results = cross_validate(classifier, X, y, cv=cv, scoring=scoring)
        print(f"{scoring} (Before): {round(cv_results['test_score'].mean(), 4)}")

        gs_best = GridSearchCV(classifier, params, cv=cv, n_jobs=-1, verbose=False).fit(X, y)
        final_model = classifier.set_params(**gs_best.best_params_)

        cv_results = cross_validate(final_model, X, y, cv=cv, scoring=scoring)
        print(f"{scoring} (After): {round(cv_results['test_score'].mean(), 4)}")
        print(f"{name} best params: {gs_best.best_params_}", end="\n\n")
        best_models[name] = final_model
    return best_models

def voting_classifier(best_models, X, y):
    print("Voting Classifier...")

    voting_clf = VotingClassifier(estimators=[('SVC', best_models["SVC"]),
                                              ('RF', best_models["RF"]),
                                              ('LightGBM', best_models["LightGBM"])],
                                  voting='soft').fit(X, y)

    cv_results = cross_validate(voting_clf, X, y, cv=3, scoring=["accuracy", "f1", "roc_auc"])
    print(f"Accuracy: {cv_results['test_accuracy'].mean()}")
    print(f"F1Score: {cv_results['test_f1'].mean()}")
    print(f"ROC_AUC: {cv_results['test_roc_auc'].mean()}")
    return voting_clf