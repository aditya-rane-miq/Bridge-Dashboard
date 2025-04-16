# bias_detection_utils.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import torch

def label_map(label):
    mapping = {
        "Sentiment Bias": 0,
        "Coded Bias": 1,
        "Sentiment & Coded Bias": 2,
        "Communal Bias": 3,
        "Neutral/Positive": 4
    }
    return mapping[label]


def tokenize_function(examples):
    return tokenizer(examples['feedback_text'], padding="max_length", truncation=True)


def train_model():
    train_df = pd.read_csv("Feedbacks/gender_bias_training_dataset.csv")
    train_df['label'] = train_df['label'].apply(label_map)
    train_dataset = Dataset.from_pandas(train_df[['feedback_text', 'label']])

    global tokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=5)

    train_dataset = train_dataset.map(tokenize_function, batched=True)
    train_dataset = train_dataset.train_test_split(test_size=0.1)

    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset['train'],
        eval_dataset=train_dataset['test'],
        tokenizer=tokenizer
    )

    trainer.train()
    model.save_pretrained("./sentiment_bias_model")
    tokenizer.save_pretrained("./sentiment_bias_model")

def predict_bias(input_df):
    from transformers import BertTokenizer, BertForSequenceClassification, Trainer
    from datasets import Dataset

    test_df = input_df.copy()
    test_dataset = Dataset.from_pandas(test_df[['feedback_text']])

    tokenizer = BertTokenizer.from_pretrained("./sentiment_bias_model")
    model = BertForSequenceClassification.from_pretrained("./sentiment_bias_model")

    def tokenize_function(examples):
        return tokenizer(examples['feedback_text'], padding="max_length", truncation=True)

    test_dataset = test_dataset.map(tokenize_function, batched=True)

    trainer = Trainer(model=model, tokenizer=tokenizer)
    predictions = trainer.predict(test_dataset)
    predicted_labels = predictions.predictions.argmax(axis=1)

    label_map_back = {
        0: "Sentiment Bias",
        1: "Coded Bias",
        2: "Sentiment & Coded Bias",
        3: "Communal Bias",
        4: "Neutral/Positive"
    }

    test_df['predicted_bias_flag'] = [label_map_back[label] for label in predicted_labels]
    return test_df

def show_bias_visualizations(test_df):
    import streamlit as st

    gender_counts = test_df['gender'].value_counts()
    biased_labels = ["Sentiment Bias", "Coded Bias", "Sentiment & Coded Bias", "Communal Bias"]

    biased_reviews_women = test_df[test_df['gender'] == 'female']['predicted_bias_flag'].isin(biased_labels).sum()
    biased_reviews_men = test_df[test_df['gender'] == 'male']['predicted_bias_flag'].isin(biased_labels).sum()

    women_percentage = (biased_reviews_women / gender_counts['female']) * 100
    men_percentage = (biased_reviews_men / gender_counts['male']) * 100

    st.metric("Biased Reviews for Women (%)", f"{women_percentage:.2f}")
    st.metric("Biased Reviews for Men (%)", f"{men_percentage:.2f}")

    # Department-wise bias
    biased_reviews_per_dept = test_df[test_df['predicted_bias_flag'].isin(biased_labels)]
    dept_bias = biased_reviews_per_dept.groupby('department')['predicted_bias_flag'].count().sort_values(ascending=False)
    st.bar_chart(dept_bias)

    # Manager-wise bias
    manager_bias = biased_reviews_per_dept.groupby('manager_name')['predicted_bias_flag'].count().sort_values(ascending=False).head(5)
    st.bar_chart(manager_bias)

    # Gender-wise bias type distribution
    gender_bias_distribution = test_df.groupby('gender')['predicted_bias_flag'].value_counts().unstack().fillna(0)
    gender_bias_distribution_percentage = gender_bias_distribution.div(gender_bias_distribution.sum(axis=1), axis=0) * 100
    st.write("### Gender-wise Bias Type Distribution")
    st.bar_chart(gender_bias_distribution_percentage)

test_df = predict_bias() 
show_bias_visualizations(test_df) 