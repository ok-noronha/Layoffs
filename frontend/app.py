from flask import Flask, request, render_template, redirect, session, url_for
from flask_mysqldb import MySQL
import datetime
from models import User
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import joblib
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
import numpy as np
from imblearn.over_sampling import RandomOverSampler
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import RandomOverSampler
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree


app = Flask(__name__)

# MySQL configuration (replace with your MySQL server details)
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = '0000'
app.config['MYSQL_DB'] = 'flask'

mysql = MySQL(app)

app.secret_key = 'your_secret_key'


@app.route("/",methods=["GET",])
def route():
    return render_template("index.html")

@app.route("/", methods=['POST'])
def login():
    df = read_dataset()
    [rf,nb,ensemble,] = all_3_models()
    new_data = {
        'company_name': [request.form.get("company"),],
        'industry': [request.form.get("industry"),],
        'date': [request.form.get("date"),],
        'country': [request.form.get("country"),],
        'total_laid_off': [request.form.get("total_laid_off"),],
    }
    new_df = pd.DataFrame(new_data)
    # Preprocess the new data
    new_X_vectorized = vectorizer.transform(new_df.astype(str).apply(lambda x: ' '.join(x), axis=1))
    # Make predictions on the new data
    rf_predictions = rf.predict(new_X_vectorized)
    nb_predictions = nb.predict(new_X_vectorized)
    ensemble_predictions = ensemble.predict(new_X_vectorized)
    return render_template("index.html", data = new_data, predictions=[rf_predictions, nb_predictions, ensemble_predictions])

app.run(debug=True)

def evaluate_model(model, X_test, y_test):
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f'Accuracy: {accuracy}')
        print('\nClassification Report:\n', classification_report(y_test, y_pred))
        print('\nConfusion Matrix:\n', confusion_matrix(y_test, y_pred))
        cv_scores = cross_val_score(model, X_vectorized, y, cv=5)
        print('\nCross-Validation Scores:', cv_scores)
        return [cv_scores,accuracy]

def read_dataset():
    return pd.read_csv("../dataset/origs.csv")

def all_3_models(df):
    features = ['company', 'industry', 'date', 'country', 'total_laid_off']
    X = df[features]
    y = df['Reason for layoffs']

    # Convert categorical features to numerical using CountVectorizer
    vectorizer = CountVectorizer()
    X_vectorized = vectorizer.fit_transform(X.astype(str).apply(lambda x: ' '.join(x), axis=1))

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)

    # Address Class Imbalance using RandomOverSampler
    oversampler = RandomOverSampler(random_state=42)
    X_resampled, y_resampled = oversampler.fit_resample(X_train, y_train)

    # Experiment with Different Models (Random Forest)
    rf_model = RandomForestClassifier(random_state=42)
    rf_model.fit(X_resampled, y_resampled)

    # Hyperparameter Tuning using GridSearchCV
    param_grid = {'alpha': [0.1, 1.0, 10.0]}
    nb_model = MultinomialNB()
    grid_search = GridSearchCV(nb_model, param_grid, cv=5)
    grid_search.fit(X_resampled, y_resampled)
    best_nb_model = grid_search.best_estimator_

    # Ensemble Methods (Voting Classifier)
    from sklearn.ensemble import VotingClassifier
    ensemble_model = VotingClassifier(estimators=[('nb', best_nb_model), ('rf', rf_model)], voting='soft')
    ensemble_model.fit(X_resampled, y_resampled)
    
    # Evaluate Naive Bayes model
    print("Evaluation Metrics for Naive Bayes:")
    evaluate_model(best_nb_model, X_test, y_test)

    # Evaluate Random Forest model
    print("\nEvaluation Metrics for Random Forest:")
    evaluate_model(rf_model, X_test, y_test)

    # Evaluate Ensemble model
    print("\nEvaluation Metrics for Ensemble:")
    evaluate_model(ensemble_model, X_test, y_test)

    # Bias Assessment
    # Check class distribution in the target variable
    class_distribution = df['Reason for layoffs'].value_counts(normalize=True)
    print('\nClass Distribution:')
    print(class_distribution)

    # Explore bias across different subgroups (e.g., industries)
    subgroup_bias = df.groupby('industry')['Reason for layoffs'].value_counts(normalize=True).unstack()
    print('\nSubgroup Bias (Industry):')
    subgroup_bias_cleaned = subgroup_bias.dropna()
    print(subgroup_bias_cleaned)

    return[rf_model,best_nb_model,ensemble_model]

