import joblib
import helpers
import os
from config import output_path

def main():
    print(os.getcwd())
    df = helpers.loading_data()
    X, y = helpers.data_processing(df)
    helpers.base_models(X, y)
    best_models = helpers.hyperparameter_optimization(X, y)
    voting_clf = helpers.voting_classifier(best_models, X, y)
    joblib.dump(voting_clf, output_path)
    return voting_clf


if __name__ == "__main__":
    print("başla")
    try:
        main()
    except Exception as e:
        print(e)

