import json

import pandas as pd
import plotly
from flask import Flask
from flask import render_template, request
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from plotly import graph_objs
from sklearn.externals import joblib
from sqlalchemy import create_engine

app = Flask(__name__)


def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


# load data
database_filename = "sqlite:///../data/DisasterResponse.db"
table_name = "df"
model_filename = "../data/classifier.pkl"

engine = create_engine(database_filename)
df = pd.read_sql_table(table_name, engine)
model = joblib.load(model_filename)


# index web page displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    # extract genre data needed for visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    # extract categories data needed for visuals
    categories = df.iloc[:, 4:]
    category_corr = categories.corr().values
    categories_top_5 = categories.mean().sort_values(ascending=False)[1:6]
    categories_less_5 = categories.mean().sort_values(ascending=True)[1:6]
    categories_names = list(categories.columns)

    # create visuals
    graphs = [
        {
            'data': [
                graph_objs.Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {
            'data': [
                graph_objs.Bar(
                    x=list(categories_top_5.index),
                    y=categories_top_5
                )
            ],

            'layout': {
                'title': 'Top 5 Message Categories',
                'xaxis': {
                    'title': "Categories"
                }
            }
        },
        {
            'data': [
                graph_objs.Bar(
                    x=list(categories_less_5.index),
                    y=categories_less_5
                )
            ],

            'layout': {
                'title': 'Less 5 Message Categories',
                'xaxis': {
                    'title': "Categories"
                }
            }
        },
        {
            'data': [
                graph_objs.Heatmap(
                    x=categories_names,
                    y=categories_names[::-1],
                    z=category_corr
                )
            ],

            'layout': {
                'title': 'Categories Heatmap'
            }
        }
    ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graph_json = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graph_json)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '')

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='127.0.0.1', port=3001, debug=True)


if __name__ == '__main__':
    main()
