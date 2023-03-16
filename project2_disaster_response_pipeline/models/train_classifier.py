import sys
import pandas as pd
from sqlalchemy import create_engine
import regex as re
import nltk
nltk.download(['punkt', 'stopwords', 'wordnet', 'omw-1.4'])
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
import pickle


def load_data(database_filepath: str):
    """
    load data from database and extract X, Y, and category names
    :param database_filepath: filepath to database
    :return: X, Y, and category names
    """
    engine = create_engine(database_filepath)
    df = pd.read_sql_table('disaster_response', con=engine)
    X = df['message']
    Y = df.iloc[:, 4:]
    category_names = Y.columns
    return X, Y, category_names


def tokenize(text: str):
    """
    clean and tokenize text
    :param text: raw text
    :return: cleaned and tokenized text list
    """
    # replace url
    text = re.sub('http[^ ]+', 'urlplaceholder', text)

    # uncapitalize
    text = text.lower()

    # remove punctuation
    text = re.sub(r'\p{P}+', ' ', text)

    # remove extra space
    text = re.sub('\s+', ' ', text)
    text = text.strip()

    # tokenize
    token_list = word_tokenize(text)

    # remove stopwords
    sw = stopwords.words('english')
    token_list = [w for w in token_list if w not in sw]

    # lemmatize
    lemmatizer = WordNetLemmatizer()
    token_list = [lemmatizer.lemmatize(w) for w in token_list]

    return token_list


def build_model():
    """
    build classification model
    :return: model
    """
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(tokenizer=tokenize)),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    # grid search
    parameters = {
        'clf__estimator__n_estimators': [10, 100],
        'clf__estimator__min_samples_split': [2, 3],
        'clf__estimator__warm_start': [True, False]
    }

    cv = GridSearchCV(pipeline, parameters)

    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """
    evaluate the model
    :param model: the classifier
    :param X_test: x test data
    :param Y_test: y test data
    :param category_names: column names of y
    :return: None
    """
    Y_pred = model.predict(X_test)
    Y_pred_df = pd.DataFrame(Y_pred, columns=category_names)
    for col in category_names:
        print(col)
        print(classification_report(Y_test[col], Y_pred_df[col]))


def save_model(model, model_filepath):
    """
    dump the model to specified location
    :param model: the classifier
    :param model_filepath: filepath to model
    :return: None
    """
    pickle.dump(model.best_estimator_, open(model_filepath, 'wb'))


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
        print('Please provide the filepath of the disaster messages database ' \
              'as the first argument and the filepath of the pickle file to ' \
              'save the model to as the second argument. \n\nExample: python ' \
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()