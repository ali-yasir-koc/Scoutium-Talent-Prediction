import pandas as pd
import numpy as np
import seaborn as sns
import joblib
import warnings
from sklearn.exceptions import ConvergenceWarning
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, VotingClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC

warnings.simplefilter("ignore", category = ConvergenceWarning)
warnings.simplefilter(action = 'ignore', category = FutureWarning)

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: '%.3f' % x)


################################################
# 1. Exploratory Data Analysis
################################################

def check_df(dataframe, head = 5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

def columns_info(dataframe):
    columns, dtypes, unique, nunique, nulls = [], [], [], [], []

    for cols in dataframe.columns:
        columns.append(cols)
        dtypes.append(dataframe[cols].dtype)
        unique.append(dataframe[cols].unique())
        nunique.append(dataframe[cols].nunique())
        nulls.append(dataframe[cols].isnull().sum())

    return pd.DataFrame({"Columns": columns,
                         "Data_Type": dtypes,
                         "Unique_Values": unique,
                         "Number_of_Unique": nunique,
                         "Missing_Values": nulls})


########################## Loading  The Data  ###########################

scoutium_att = pd.read_csv("datasets/scoutium_attributes.csv", sep = ";")
scoutium_pot = pd.read_csv("datasets/scoutium_potential_labels.csv", sep = ";")
df_att = scoutium_att.copy()
df_pot = scoutium_pot.copy()

df_att.head()
df_pot.head()

df = pd.merge(df_att, df_pot, how = "inner", on = ["task_response_id", "match_id", "evaluator_id", "player_id"])

check_df(df)

columns_info(df)

df.head()

##########################   Function   ###########################

def loading_data():
    scoutium_att = pd.read_csv("datasets/scoutium_attributes.csv", sep = ";")
    scoutium_pot = pd.read_csv("datasets/scoutium_potential_labels.csv", sep = ";")
    df = pd.merge(scoutium_att, scoutium_pot, how = "inner", on = ["task_response_id", "match_id", "evaluator_id", "player_id"])
    return df


df = loading_data()


################################################
# 2. Data Preprocessing & Feature Engineering
################################################

def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe


def one_hot_encoder(dataframe, categorical_cols, drop_first = False):
    dataframe = pd.get_dummies(dataframe, columns = categorical_cols, drop_first = drop_first)
    return dataframe


df = df[~(df["position_id"] == 1)].reset_index(drop = True)
df = df[~(df["potential_label"] == "below_average")].reset_index(drop = True)

new_df = df.pivot_table(values = "attribute_value",
                        index = ["player_id", "position_id", "potential_label"],
                        columns = "attribute_id").reset_index()

new_df.head()

new_df.columns = [str(col) for col in new_df.columns]
new_df.columns = [col.upper() for col in new_df.columns]

att_cols = new_df.iloc[:, 3:].columns
new_df["TOTAL_SCORE"] = new_df[att_cols].sum(axis = 1) / len(att_cols)

########################## Label Encode  ###########################

label_encoder(new_df, "POTENTIAL_LABEL")

new_df.info()

########################## One-Hot Encode  ###########################

new_df = one_hot_encoder(new_df, ["POSITION_ID"])

new_df.columns = [col.upper() for col in new_df.columns]

########################## Standardisation  ###########################

num_cols = [col for col in new_df.columns if new_df[col].dtypes == "float64"]

ss = StandardScaler()
for col in num_cols:
    new_df[col] = ss.fit_transform(new_df[[col]])
new_df.head()


########################## Function  ###########################

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


X, y = data_processing(df)


######################################################
# 3. Base Models
######################################################

def base_models(X, y, scoring = "roc_auc"):
    print("Base Models....")
    classifiers = [('LR', LogisticRegression()),
                   ('KNN', KNeighborsClassifier()),
                   ("SVC", SVC()),
                   ("CART", DecisionTreeClassifier(random_state = 17)),
                   ("RF", RandomForestClassifier(random_state = 17)),
                   ('Adaboost', AdaBoostClassifier(random_state = 17)),
                   ('GBM', GradientBoostingClassifier(random_state = 17)),
                   ('XGBoost', XGBClassifier(eval_metric = 'logloss', random_state = 17)),
                   ('LightGBM', LGBMClassifier(random_state = 17)),
                   # ('CatBoost', CatBoostClassifier(verbose=False))
                   ]

    for name, classifier in classifiers:
        cv_results = cross_validate(classifier, X, y, cv = 3, scoring = scoring)
        print(f"{scoring}: {round(cv_results['test_score'].mean(), 4)} ({name}) ")


base_models(X, y, scoring = "accuracy")

######################################################
# 4. Automated Hyperparameter Optimization
######################################################

knn_params = {"n_neighbors": range(2, 50)}

svc_params = {"gamma": [0.1, 1, 10, 100],
              "C": [0.1, 1, 10, 100, 1000]}

cart_params = {'max_depth': range(1, 20),
               "min_samples_split": range(2, 30)}

rf_params = {"max_depth": [8, 15, None],
             "max_features": [5, 7, "auto"],
             "min_samples_split": [15, 20],
             "n_estimators": [200, 300]}

xgboost_params = {"learning_rate": [0.1, 0.01],
                  "max_depth": [5, 8],
                  "n_estimators": [100, 200]}

lightgbm_params = {"learning_rate": [0.01, 0.1],
                   "n_estimators": [300, 500]}


classifiers = [('KNN', KNeighborsClassifier(), knn_params),
               ('SVC', SVC(probability = True, random_state = 17), svc_params),
               ("CART", DecisionTreeClassifier(random_state = 17), cart_params),
               ("RF", RandomForestClassifier(random_state = 17), rf_params),
               ('XGBoost', XGBClassifier(eval_metric='logloss', random_state = 17), xgboost_params),
               ('LightGBM', LGBMClassifier(random_state = 17), lightgbm_params)]



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


best_models = hyperparameter_optimization(X, y)


######################################################
# 5. Stacking & Ensemble Learning
######################################################

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


voting_clf = voting_classifier(best_models, X, y)


######################################################
# 6. Prediction for a New Observation
######################################################

X.columns
random_user = X.sample(1, random_state=45)
voting_clf.predict(random_user)

joblib.dump(voting_clf, "voting_clf2.pkl")

new_model = joblib.load("voting_clf2.pkl")
new_model.predict(random_user)