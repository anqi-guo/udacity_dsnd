import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    '''
    read in meessages data and categories data, and merge them on id column
    :param messages_filepath: path including the file name
    :param categories_filepath: path including the file name
    :return: merged data
    '''
    # load messages dataset
    messages = pd.read_csv(messages_filepath)

    # load categories dataset
    categories = pd.read_csv(categories_filepath)

    # merge datasets
    df = messages.merge(categories, on='id')
    return df


def clean_data(df):
    '''
    simplify categories column, and drop dupliates
    :param df: merged data
    :return: cleaned data
    '''
    # create a dataframe of the 36 individual category columns
    categories = df.categories.str.split(';', expand=True)

    # extract a list of new column names for categories
    row = categories.iloc[0]
    category_colnames = [r.split('-')[0] for r in row]

    # rename the columns of `categories`
    categories.columns = category_colnames

    # Convert category values to just numbers 0 or 1
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str.split('-', expand=True)[1]

        # convert column from string to numeric
        categories[column] = categories[column].astype(float)

    # Replace categories column in df with new category columns
    df.drop(columns=['categories'], inplace=True)
    df = df.join(categories)

    # Remove duplicates
    df.drop_duplicates(inplace=True)

    # Replace values 2 with 1
    df.replace(2, 1, inplace=True)

    return df


def save_data(df, database_filename):
    '''
    save data to sqlite database
    :param df: data
    :param database_filename: database
    :return: None
    '''
    engine = create_engine(database_filename)
    df.to_sql('disaster_response', engine, index=False, if_exists='replace')


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories ' \
              'datasets as the first and second argument respectively, as ' \
              'well as the filepath of the database to save the cleaned data ' \
              'to as the third argument. \n\nExample: python process_data.py ' \
              'disaster_messages.csv disaster_categories.csv ' \
              'DisasterResponse.db')


if __name__ == '__main__':
    main()