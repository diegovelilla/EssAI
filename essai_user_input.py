# --- IMPORTS ---

import torch
from datasets import Dataset
from transformers import Trainer, TrainingArguments, DataCollatorWithPadding
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
stopwords = set(stopwords.words('english'))

# -------------------------

# --- INPUT ---

input_list = [""" WRITE HERE YOUR FIRST ESSAY """,
              """ WRITE HERE YOUR SECOND ESSAY """]

# -------------

# --- USEFUL FUNCTIONS ----


def clean_text(text):
    """
    This funtion get's rid of nonalphabetical characters, stopwords and lower cases the text.

    Args:
    text (str): The text to be cleaned

    Returns:
    text (str): The cleaned text

    Example:
    df['text'] = df['text'].apply(clean_text)
    """
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = text.lower()
    words = text.split()
    text = [word for word in words if not word in stopwords]
    text = ' '.join(words)
    return text


def tokenize_function(dataframe):
    """
    This funtion tokenizes the 'text' field of the dataframe.

    Args:
    dataframe (pandas.DataFrame): The dataframe to be tokenized

    Returns:
    dataframe (pandas.DataFrame): The tokenized dataframe

    Example and output:
    train_dataset_token = train_dataset.map(tokenize_function, batched=True)
    """
    return tokenizer(dataframe["text"], truncation=True)


def compute_metrics(eval_pred):
    """
    This funtion computes the accuracy, precision, recall and f1 score of the model.

    It'is passed to the trainer and it outputs when evaluating the model.

    Args:
    eval_pred (tuple): The predictions and labels of the model

    Returns:
    dict: The accuracy, precision, recall and f1 score of the model

    Example:
    >>> trainer.evaluate()
    {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }
    """
    predictions, labels = eval_pred
    predictions = predictions.argmax(axis=-1)
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='binary')
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

# -------------------------

# --- LOADING THE MODEL ---


# Load the initial tokenizer and model to set the number of labels its going to classify as 2
checkpoint = "diegovelilla/EssAI"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

# -------------------------

# --- DATA PREPROCESSING ---

n_input = len(input_list)

# Now we convert the input to a dataset
df = pd.DataFrame({'text': input_list})


# Get rid of nonalphatetical characters, stopwords and we lower case it.
df['text'] = df['text'].apply(clean_text)

# We convert the pandas dataframe into hugging face datasets and tokenize both of them
ds = Dataset.from_pandas(df)
ds_token = ds.map(tokenize_function, batched=True)

# Drop columns that are not necessary and set the dataset format to pytorch tensors
ds_token = ds_token.remove_columns(["text", "token_type_ids"])
ds_token.set_format(type='torch', columns=['input_ids', 'attention_mask'])

# -------------------------

# --- INSTANTIATING TRAINER ----

# We instantiate a DataCollatorWithPadding in order to pad the inputs in batches while training
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Create the training arguments
training_args = TrainingArguments(".")

# Create the trainer
trainer = Trainer(
    model,
    training_args,
    eval_dataset=ds_token,
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# -------------------------

# --- PREDICT ---

# We predict and then format the output
predictions = trainer.predict(ds_token)
predictions = torch.from_numpy(predictions.predictions)
predictions = torch.nn.functional.softmax(predictions, dim=-1)

print('\n\n')
for i in range(n_input):
    index = torch.argmax(predictions[i])
    print(f'{i+1}: HUMAN {round(predictions[i][0].item() * 100, 2)}% of confidence.') if index == 0 else print(
        f'{i+1}: AI {round(predictions[i][1].item() * 100, 2)}% of confidence.')

# -------------------------
