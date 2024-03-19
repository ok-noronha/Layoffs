from flask import Flask, request, render_template
from flask_mysqldb import MySQL
import joblib
import pickle
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import RandomOverSampler
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV
import time

app = Flask(__name__)

# MySQL configuration (replace with your MySQL server details)
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = '0000'
app.config['MYSQL_DB'] = 'flask'

mysql = MySQL(app)

app.secret_key = 'your_secret_key'

def read_dataset():
    return pd.read_csv("../dataset/origs.csv")

def preprocess_data(df):
    # Preprocess the data
    features = ['company', 'industry', 'date', 'country', 'total_laid_off']
    X = df[features]
    y = df['Reason for layoffs']  # 'extra_column' is the target variable we want to predict and rename as 'reason'

    # Convert categorical features to numerical using CountVectorizer
    vectorizer = CountVectorizer()
    X_vectorized = vectorizer.fit_transform(X.astype(str).apply(lambda x: ' '.join(x), axis=1))
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)
    return X_train,y_train, X_test, y_test, vectorizer

def train_model(X_train, y_train, X_test,y_test):

    try:
        nb_model = joblib.load('../models/model_nb.joblib')
    except:
        nb_model = MultinomialNB()
    nb_model.fit(X_train, y_train)
    joblib.dump(nb_model, '../models/model_nb.joblib')

    

    oversampler = RandomOverSampler(random_state=42)
    X_resampled, y_resampled = oversampler.fit_resample(X_train, y_train)

    try:
        rf_model = joblib.load('../models/model_rf.joblib')
    except:
        # Experiment with Different Models (Random Forest)
        rf_model = RandomForestClassifier(random_state=42)
    rf_model.fit(X_resampled, y_resampled)
    joblib.dump(rf_model,'../models/model_rf.joblib')
    
    param_grid = {'alpha': [0.1, 1.0, 10.0]}
    try:
        nbbest_model = joblib.load('../models/model_nbbest.joblib')
    except:
        nbbest_model = MultinomialNB()
    grid_search = GridSearchCV(nb_model, param_grid, cv=5)
    grid_search.fit(X_resampled, y_resampled)
    best_nb_model = grid_search.best_estimator_
    joblib.dump(best_nb_model, '../models/model_nbbest.joblib')

    try:
        ensemble_model = joblib.load("../models/model_ensemble.joblib")
    except:
        ensemble_model = VotingClassifier(estimators=[('nb', best_nb_model), ('rf', rf_model)], voting='soft')
    ensemble_model.fit(X_resampled, y_resampled)
    joblib.dump(ensemble_model, "../models/model_ensemble.joblib")

    return nb_model, rf_model, best_nb_model, ensemble_model

def get_models():
    return joblib.load('../models/model_nb.joblib'),joblib.load('../models/model_rf.joblib'),joblib.load('../models/model_nbbest.joblib'),joblib.load("../models/model_ensemble.joblib")

def evaluate_model(models, X_test, y_test):
    evaluation_results = []
    for model in models:
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        classification_report_str = classification_report(y_test, y_pred)
        confusion_matrix_arr = confusion_matrix(y_test, y_pred)
        cross_val_scores = cross_val_score(model, X_test, y_test, cv=5)
        evaluation_results.append({
            "model_name": type(model).__name__,
            "accuracy": accuracy,
            "classification_report": classification_report_str,
            "confusion_matrix": confusion_matrix_arr,
            "cross_val_scores": cross_val_scores.tolist()  # Convert to list for easier JSON serialization
        })
    return evaluation_results
    # y_pred = model.predict(X_test)
    # accuracy = accuracy_score(y_test, y_pred)
    # print(f'Accuracy: {accuracy}')
    # print('\nClassification Report:\n', classification_report(y_test, y_pred))
    # print('\nConfusion Matrix:\n', confusion_matrix(y_test, y_pred))
    # return accuracy,classification_report(y_test, y_pred), confusion_matrix(y_test, y_pred),cross_val_score(model, X_test, y_test, cv=5)

@app.route("/", methods=["GET",])
def route():
    return render_template("index.html")

@app.route("/", methods=['POST'])
def pred():
    df = read_dataset()
    df.dropna(inplace=True)
    # Drop columns with any NaN values
    df.dropna(axis=1, inplace=True)
    Xt, yt, xtt, ytt, vect = preprocess_data(df)
    try:
        nb, rf, nbbest, ensemble = get_models()
    except:
        return train()
    models=[nb, rf, nbbest, ensemble]
    new_data = {
        'company_name': [request.form.get("company"),],
        'industry': [request.form.get("industry"),],
        'date': [request.form.get("date"),],
        'country': [request.form.get("country"),],
        'total_laid_off': [request.form.get("total_laid_off"),],
    }
    print(new_data)
    new_df = pd.DataFrame(new_data)
    new_X_vectorized = vect.transform(new_df.astype(str).apply(lambda x: ' '.join(x), axis=1))
    nb_predictions = nb.predict(new_X_vectorized)
    rf_predictions = rf.predict(new_X_vectorized)
    nbbest_predictions = nbbest.predict(new_X_vectorized)
    ensemble_predictions = ensemble.predict(new_X_vectorized)
    # xttvec = vect.transform(xtt.astype(str).apply(lambda x: ' '.join(x), axis=1))
    print(evaluate_model(rf,xtt,ytt))
    return render_template("index.html", data=new_data, predictions=[rf_predictions[0], nb_predictions[0], nbbest_predictions[0], ensemble_predictions[0]], metrics=[evaluate_model(x,xtt,ytt) for x in models])

@app.route("/results",methods=["GET"])
def train():
    df = read_dataset()
    df.dropna(inplace=True)
    df.dropna(axis=1, inplace=True)
    Xt, yt, xtt, ytt, vect = preprocess_data(df)
    nb, rf, nbbest, ensemble = train_model(Xt, yt, xtt, ytt)
    models=[nb, rf, nbbest, ensemble]
    evaluation_results = evaluate_model(models, xtt, ytt)
    print(evaluation_results)
    return render_template('index.html', evaluation_results=evaluation_results)


if __name__ == "__main__":
    app.run(debug=True)