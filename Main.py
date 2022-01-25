# ! pip install nltk
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
import pandas as pd
import torch
import nltk
# from google.colab import drive
from sklearn import model_selection
from nltk.corpus import stopwords
stop = stopwords.words('english')
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

# drive.mount('/content/drive')


lemmatizer = WordNetLemmatizer()

def read_data():
    return pd.read_csv('/content/drive/MyDrive/IMDB_Dataset.csv')


def split_data(instance):
    x = instance.values[:, 0]
    y = instance.values[:, 1]
    x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, stratify=True,
                                                                        test_size=0.2, random_state=42)
    x_train, x_validate, y_train, y_validate = model_selection.train_test_split(x_train, y_train, stratify= True,
                                                                                test_size=0.1, random_state=42)
    return x_train, x_test, y_train, y_test

def data_preprocessing(data):
    data["review"] = data["review"].str.lower()
    data["review"] = data['review'].str.replace('[^\w\s]','')
    data["review"] = data['review'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
    data["review"] = data['review'].apply(lambda x: ' '.join(lemmatizer.lemmatize(word, pos="a") for word in x.split()))
    data["review"] = data['review'].apply(lambda x: ' '.join(lemmatizer.lemmatize(word, pos="s") for word in x.split()))
    data["review"] = data['review'].apply(lambda x: ' '.join(lemmatizer.lemmatize(word, pos="v") for word in x.split()))
    # data["review"] = data['review'].apply(lambda x: ' '.join(lemmatizer.lemmatize(word, pos="j") for word in x.split()))
    data["review"] = data['review'].apply(lambda x: ' '.join(lemmatizer.lemmatize(word, pos="n") for word in x.split()))
    data["review"] = data['review'].apply(lambda x: ' '.join(lemmatizer.lemmatize(word, pos="r") for word in x.split()))
    return data

data = read_data()
data_processed = data_preprocessing(data)
print(data_processed)
# split_data(data_proccessed)

