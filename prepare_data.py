import pandas as pd
import re
from sklearn.model_selection import train_test_split


data_fake = pd.read_csv('News/Fake.csv')
data_real = pd.read_csv('News/True.csv')


data_fake['label'] = 0  
data_real['label'] = 1  


data = pd.concat([data_fake, data_real])
data = data[['title', 'text', 'label']]


def clean_text(text):
    text = re.sub(r'<.*?>', '', str(text))             
    text = re.sub(r'http\S+', '', text)                
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)        
    text = text.lower()                                
    return text.strip()

data["title"] = data["title"].apply(clean_text)
data["text"] = data["text"].apply(clean_text)

data["content"] = data["title"] + " " + data["text"]
data = data[["content", "label"]]



train_df, temp_df = train_test_split(data, test_size=0.3, stratify=data["label"], random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df["label"], random_state=42)

train_df.to_csv("fake_news_train.csv", index=False)
val_df.to_csv("fake_news_val.csv", index=False)
test_df.to_csv("fake_news_test.csv", index=False)

data.to_csv("all_the_news.csv", index= False)

