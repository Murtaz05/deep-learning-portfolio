import pandas as pd
import numpy as np
import os
import torch

from torch.utils.data import Dataset

import re
from bs4 import BeautifulSoup

# %%
dataset_folder_path = '/home/murtaza/University_Data/deep_learning/assignment4/dataset'

# %%
answers_df = pd.read_csv(f'{dataset_folder_path}/Answers.csv', encoding='latin1')
questions_df = pd.read_csv(f'{dataset_folder_path}/Questions.csv', encoding='latin1')
tags_df = pd.read_csv(f'{dataset_folder_path}/Tags.csv', encoding='latin1')


# %%
questions_df.head()

# %%
answers_df.head()

# %%
tags_df.head()

# %%

questions_df.shape, answers_df.shape, tags_df.shape

# %%
# Step 1: Rename columns for clarity
questions_df = questions_df.rename(columns={
    'Id': 'question_id',
    'Score': 'question_score',
    'Title': 'question_title',
    'Body': 'question_body'
})

answers_df = answers_df.rename(columns={
    'ParentId': 'question_id',
    'Body': 'answer_body',
    'Score': 'answer_score'
})

tags_df = tags_df.rename(columns={'Id': 'question_id'})

questions_df['question_text'] = questions_df['question_title'] + ' ' + questions_df['question_body']

# %%
# Sort by score so highest answer comes first per question
answers_df_sorted = answers_df.sort_values(by=['question_id', 'answer_score'], ascending=[True, False])

# Drop duplicates to keep only the top answer per question
top_answers_df = answers_df_sorted.drop_duplicates(subset='question_id', keep='first')


# %%

# Step 2: Merge answers with questions
df = pd.merge(
    questions_df[['question_id', 'question_text']],
    top_answers_df[['question_id', 'answer_body']],
    on='question_id',
    how='inner'
)


# %%
df = df.sample(n=1000, random_state=42).reset_index(drop=True)


# %%


def preprocess_text(text):
    if pd.isnull(text):
        return ""

    # 1. Convert to lowercase
    text = text.lower()
    
    # 2. Remove HTML tags using BeautifulSoup
    text = BeautifulSoup(text, "html.parser").get_text()
    
    # 3. Replace code blocks with [code] (optional enhancement)
    text = re.sub(r'```.*?```', '[code]', text, flags=re.DOTALL)  # multiline code blocks
    text = re.sub(r'<code>.*?</code>', '[code]', text, flags=re.DOTALL)  # <code> tags

    # 4. Remove special characters (keeping basic punctuation)
    text = re.sub(r'[^a-zA-Z0-9\s.,!?]', '', text)
    
    # 5. Add start/end tokens
    text = f"<murtaza_msds24040> {text.strip()} </murtaza_msds24040>"
    
    return text


# %%
df['question_text'] = df['question_text'].apply(preprocess_text)
df['answer_body'] = df['answer_body'].apply(preprocess_text)


# %%
from sklearn.model_selection import train_test_split

train_df, temp_df = train_test_split(df, test_size=0.4, random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

print(f"Train: {len(train_df)}, Validation: {len(val_df)}, Test: {len(test_df)}")


# %% [markdown]
# 3. Tokenization

# %%
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("openai-community/openai-gpt")
tokenizer.add_special_tokens({'pad_token': '[PAD]','additional_special_tokens': ['<murtaza_msds24040>', '</murtaza_msds24040>']})
tokenizer.pad_token = '[PAD]'


# %%
tokenizer.model_max_length

# %%
def tokenize_text(text, max_length=512):
    return tokenizer(
        text,
        padding='max_length',
        truncation=True,
        max_length=max_length,
        return_tensors='pt'
    )


# %%
train_question_encodings = tokenize_text(train_df['question_text'].tolist())
train_answer_encodings = tokenize_text(train_df['answer_body'].tolist())

val_question_encodings = tokenize_text(val_df['question_text'].tolist())
val_answer_encodings = tokenize_text(val_df['answer_body'].tolist())

test_question_encodings = tokenize_text(test_df['question_text'].tolist())
test_answer_encodings = tokenize_text(test_df['answer_body'].tolist())


# %%
def mask_pad_tokens(labels, pad_token_id):
    # labels is a tensor of shape (batch_size, seq_len)
    return torch.where(labels == pad_token_id, torch.tensor(-100), labels)

train_answer_labels = mask_pad_tokens(train_answer_encodings['input_ids'], tokenizer.pad_token_id)
val_answer_labels = mask_pad_tokens(val_answer_encodings['input_ids'], tokenizer.pad_token_id)
test_answer_labels = mask_pad_tokens(test_answer_encodings['input_ids'], tokenizer.pad_token_id)


# %%
train_answer_labels

# %%
train_answer_encodings['input_ids']

# %% [markdown]
# 4. Data Loader

# %%
class QADataset(Dataset):
    def __init__(self, input_encodings, attention_masks, labels):
        self.input_ids = input_encodings
        self.attention_masks = attention_masks
        self.labels = labels

    def __len__(self):
        return self.input_ids.size(0)  # since it's already a tensor

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_masks[idx],
            'labels': self.labels[idx]
        }


# %%
train_dataset = QADataset(
    input_encodings=train_question_encodings['input_ids'],
    attention_masks=train_question_encodings['attention_mask'],
    labels=train_answer_labels  # list of tensors with -100 for padded tokens
)

val_dataset = QADataset(
    input_encodings=val_question_encodings['input_ids'],
    attention_masks=val_question_encodings['attention_mask'],
    labels=val_answer_labels
)

test_dataset = QADataset(
    input_encodings=test_question_encodings['input_ids'],
    attention_masks=test_question_encodings['attention_mask'],
    labels=test_answer_labels
)


# %%
from torch.utils.data import DataLoader
batch_size = 8
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)
