import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# === 1. Chargement des données ===
train_df = pd.read_csv("fake_news_train.csv")
val_df = pd.read_csv("fake_news_val.csv")

# === 2. Tokenizer et modèle pré-entraîné ===
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# === 3. Fonction de tokenization ===
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

# === 4. Optimiseur ===
optimizer = AdamW(model.parameters(), lr=2e-5)

# === 5. Boucle d'entraînement ===
num_epochs = 3
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch in train_loader:
        input_ids, attention_mask, labels = [x.to(device) for x in batch]
        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Training Loss: {total_loss / len(train_loader):.4f}")

    # === 6. Évaluation ===
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in val_loader:
            input_ids, attention_mask, labels = [x.to(device) for x in batch]
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary')
    print(f"Validation - Acc: {acc:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")
