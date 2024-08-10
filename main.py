import nltk
import re
from datasets import load_dataset
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()
def get_dataset_info(ds):
    train_ds = ds['train']
    test_ds = ds['test']
    train_ds_size = len(ds['train'])
    test_ds_size = len(ds['test'])
    ds_header = list(ds['train'][0].keys())
    return train_ds, test_ds, train_ds_size, test_ds_size, ds_header
def preprocess(text):
    text = text.lower()
    # Remove special characters and numbers
    text = re.sub(r'\W+', ' ', text)
    words = word_tokenize(text)
    # Remove stopwords and stem the words
    words = [word for word in words if word not in stop_words]
    # Join the words back into a string
    return ' '.join(words)
def pre_process_dataset(ds, ds_header):
    preprocessed_data = {}
    for header in ds_header:
        preprocessed_data[header] = []
    for entry in ds:
        context, question, answer = entry['context'], entry['question'], entry['answer']
        if context != None and question != None and answer != None:
           preprocessed_data['context'].append(preprocess(context))
           preprocessed_data['question'].append(preprocess(question))
           preprocessed_data['answer'].append(preprocess(answer))
        break
    print(preprocessed_data['context'][0])
    print(preprocessed_data['question'][0])
    print(preprocessed_data['answer'][0])





ds = load_dataset("neural-bridge/rag-dataset-12000")
train_ds, test_ds, train_ds_size, test_ds_size, ds_header = get_dataset_info(ds)
pre_process_dataset(train_ds,ds_header)