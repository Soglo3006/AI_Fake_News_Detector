import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

train_df = pd.read_csv("fake_news_train.csv")
val_df = pd.read_csv("fake_news_val.csv")

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def tokenize_dataset(df):
    tokens = tokenizer(
        list(df["content"]),
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )
    labels = torch.tensor(df["label"].values)
    return TensorDataset(tokens["input_ids"], tokens["attention_mask"], labels)

train_dataset = tokenize_dataset(train_df)
val_dataset = tokenize_dataset(val_df)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8)