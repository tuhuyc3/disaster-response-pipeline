import sys
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger','punkt_tab','averaged_perceptron_tagger_eng'])
import os

import re
import pickle
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.metrics import classification_report


def load_data(database_filepath):
    """
    INPUT:
    database_filepath - 
    
    OUTPUT:
    X - messages (input variable) 
    y - categories of the messages (output variable)
    category_names - category name for y
    """
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('disaster_response', con = engine)
    
    X = df['message']
    y = df.iloc[:,4:]
    category_names = y.columns
    return X, y, category_names


def tokenize(text):
    """
    INPUT:
    text - raw text
    
    OUTPUT:
    clean_tokens - tokenized messages
    """
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
    
    tokens = word_tokenize(text)

    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
        
    return clean_tokens

from sklearn.base import BaseEstimator, TransformerMixin
import nltk
import pandas as pd

class StartingVerbExtractor(BaseEstimator, TransformerMixin):
    """
    A transformer that extracts whether the first word of each message 
    in a text is a verb or not. This can help in classifying messages 
    based on their starting verbs.
    """

    def starting_verb(self, text):
        """
        Check if the first word of the given text is a verb.

        Parameters:
        text (str): The input text to analyze.

        Returns:
        bool: True if the first word is a verb or 'RT', False otherwise.
        """
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(nltk.word_tokenize(sentence))
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return True
        return False

    def fit(self, X, y=None):
        """
        Fit the transformer to the data (not required for this transformer).

        Parameters:
        X (pd.Series or array-like): The input data to fit.
        y (None): Ignored.

        Returns:
        self: The fitted transformer.
        """
        return self

    def transform(self, X):
        """
        Transform the input data by applying the starting verb extraction.

        Parameters:
        X (pd.Series or array-like): The input data to transform.

        Returns:
        pd.DataFrame: A DataFrame containing boolean values indicating 
                      whether each message starts with a verb.
        """
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)


def build_model(clf = RandomForestClassifier()):
    """
    INPUT:
    clf - classifier model (If none is inputted, the function will use default 'AdaBoostClassifier' model) 
    
    OUTPUT:
    cv = ML model pipeline after performing grid search
    """
    pipeline = Pipeline([
        ('features', FeatureUnion([
            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),
            ('starting_verb', StartingVerbExtractor())
        ])),
        ('clf', MultiOutputClassifier(clf))
    ])
    
    param_grid = {
        'clf__estimator__max_depth': [50, 100],
        'clf__estimator__n_estimators': [10, 20, 30]
        }

        
    cv = GridSearchCV(pipeline, param_grid=param_grid, cv=5, n_jobs=-1, verbose=3) 
    
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """
    INPUT:
    model - ML model
    X_test - test messages
    Y_test - categories for test messages
    category_names - category name for Y
    
    OUTPUT:
    none - print scores (precision, recall, f1-score) for each output category of the dataset.
    """
    Y_pred_test = model.predict(X_test)
    print(classification_report(Y_test.values, Y_pred_test, target_names=category_names))


def save_model(model, model_filepath):
    """
    INPUT:
    model - ML model
    model_filepath - location to save the model
    
    OUTPUT:
    none
    """
    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()