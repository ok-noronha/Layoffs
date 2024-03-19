from flask import Flask, request, render_template
from flask_mysqldb import MySQL
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import RandomOverSampler
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV

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
    y = df['Reason for layoffs']

    # Convert categorical features to numerical using CountVectorizer
    vectorizer = CountVectorizer()
    X_vectorized = vectorizer.fit_transform(X.astype(str).apply(lambda x: ' '.join(x), axis=1))
    feature_names = vectorizer.get_feature_names_out()

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test

def train_model(X_train, X_test, y_train, y_test):

    nb_model = MultinomialNB()
    nb_model.fit(X_train, y_train)

    oversampler = RandomOverSampler(random_state=42)
    X_resampled, y_resampled = oversampler.fit_resample(X_train, y_train)

    # Experiment with Different Models (Random Forest)
    rf_model = RandomForestClassifier(random_state=42)
    rf_model.fit(X_resampled, y_resampled)
    
    param_grid = {'alpha': [0.1, 1.0, 10.0]}
    nbbest_model = MultinomialNB()
    grid_search = GridSearchCV(nb_model, param_grid, cv=5)
    grid_search.fit(X_resampled, y_resampled)
    best_nb_model = grid_search.best_estimator_

    ensemble_model = VotingClassifier(estimators=[('nb', best_nb_model), ('rf', rf_model)], voting='soft')
    ensemble_model.fit(X_resampled, y_resampled)

    return nb_model, rf_model, best_nb_model, ensemble_model


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy}')
    print('\nClassification Report:\n', classification_report(y_test, y_pred))
    print('\nConfusion Matrix:\n', confusion_matrix(y_test, y_pred))
    return accuracy

@app.route("/", methods=["GET",])
def route():
    return render_template("index.html")

@app.route("/", methods=['POST'])
def login():
    df = read_dataset()
    df.dropna(inplace=True)
    # Drop columns with any NaN values
    df.dropna(axis=1, inplace=True)
    Xt, yt, xtt, ytt = preprocess_data(df)
    nb, rf, nbbest, ensemble = train_model(Xt, yt, xtt, ytt)
    new_data = {
        'company_name': [request.form.get("company"),],
        'industry': [request.form.get("industry"),],
        'date': [request.form.get("date"),],
        'country': [request.form.get("country"),],
        'total_laid_off': [request.form.get("total_laid_off"),],
    }
    print(new_data)
    new_df = pd.DataFrame(new_data)
    vectorizer = CountVectorizer()
    new_X_vectorized = vectorizer.transform(new_df.astype(str).apply(lambda x: ' '.join(x), axis=1))
    nb_predictions = nb.predict(new_X_vectorized)
    rf_predictions = rf.predict(new_X_vectorized)
    nbbest_predictions = nbbest.predict(new_X_vectorized)
    ensemble_predictions = ensemble.predict(new_X_vectorized)
    return render_template("index.html", data=new_data, predictions=[rf_predictions, nb_predictions, nbbest_predictions, ensemble_predictions])

if __name__ == "__main__":
    app.run(debug=True)