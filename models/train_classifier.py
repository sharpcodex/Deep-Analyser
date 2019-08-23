import pickle
import re
import sys
import warnings

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sqlalchemy import create_engine


def load_data(database_filepath):
    """
    Load data from the provided sqlLite database into a pandas data frame.

    Parameters:
        database_filepath: File path to sqlLite database
    Output:
        x: features data frame
        y: labels data frame
        category_names: Categories names
    """

    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('df', engine)

    x = df['message']
    y = df.iloc[:, 4:]
    category_names = list(df.columns[4:])

    return x, y, category_names


def tokenize(text):
    """
    Tokenize, clean, and lemmatize the provided text.

    Parameters:
        text: Text to tokenize, clean, and lemmatize
    Output:
        text_out: Tokenized, cleaned, and lemmatized text
    """

    # Tokenize and clean
    lower = re.sub(r"[^a-zA-Z0-9]", ' ', text.lower())
    tokens = word_tokenize(lower)
    tokens = [t for t in tokens if t not in stopwords.words('english')]

    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    text_out = [lemmatizer.lemmatize(t).strip() for t in tokens]

    return text_out


def build_model():
    """
    Build a Machine Learning pipeline using TfidfTransformer, RandomForestClassifier, and GridSearchCV.

    Output:
        a GridSearchCV object
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    parameters = {
        # 'tfidf__smooth_idf': [True, False],
        # 'clf__estimator__n_estimators': [50, 500],
        # 'clf__estimator__max_features': ['auto', 'sqrt', 'log2'],
        # 'clf__estimator__max_depth': [4, 5, 6, 7, 8],
        # 'clf__estimator__min_samples_split': [2, 3, 4, 5],
        # 'clf__estimator__min_samples_leaf': [2, 5, 10, 20],
        # 'clf__estimator__criterion': ['gini', 'entropy']
    }

    return GridSearchCV(pipeline, param_grid=parameters, verbose=10)


def evaluate_model(model, x, y, category_names):
    """
    Evaluate the provided model.

    Arguments:
        model -> Model to be evaluated
        x -> Test features data frame
        y -> Test labels data frame
        category_names -> Categories names
    """

    y_pred = model.predict(x)
    reports = {category_names[i]: (
        accuracy_score(y.iloc[:, i].values, y_pred[:, i]),
        classification_report(y.iloc[:, i].values, y_pred[:, i])
    ) for i in range(len(category_names))}

    for k, v in reports.items():
        print("Category: {}, Accuracy = {}\n{}\n\n".format(k, v[0], v[1]))


def save_model(model, model_filepath):
    """
    Save the provided model to a Pickle file.

    Parameters:
        model: Model to be saved
        model_filepath: destination path to save the output Pickle file
    """
    pickle.dump(model, open(model_filepath, "wb"))


def main():
    warnings.filterwarnings("ignore")
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]

        print('\nDownloading required NLTK data...')
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('wordnet')

        print('\nLoading data...\n    DATABASE: {}'.format(database_filepath))
        x, y, category_names = load_data(database_filepath)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

        print('\nBuilding model...')
        model = build_model()

        print('\nTraining model...')
        model.fit(x_train, y_train)

        print('\nEvaluating model...')
        evaluate_model(model, x_test, y_test, category_names)

        print('\nSaving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('\nTrained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database as the first argument and the filepath '
              'of the pickle file to save the model to as the second argument. \n\nExample: python train_classifier.py '
              '../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
