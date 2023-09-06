import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


random_state = 17

attribute_path = "datasets/scoutium_attributes.csv"
label_path = "datasets/scoutium_potential_labels.csv"

output_path = "miuul_homework/Scoutium/voting_clf_scoutium.pkl"

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
               ('SVC', SVC(probability = True, random_state = random_state), svc_params),
               ("CART", DecisionTreeClassifier(random_state = random_state), cart_params),
               ("RF", RandomForestClassifier(random_state = random_state), rf_params),
               ('XGBoost', XGBClassifier(eval_metric='logloss', random_state = random_state), xgboost_params),
               ('LightGBM', LGBMClassifier(random_state = random_state), lightgbm_params)]

