import os
import torch
import module.utility as utility
import numpy as np
import pandas as pd

from torch import nn

from sklearn.model_selection import train_test_split
from tqdm import tqdm
from transformers import  DistilBertForSequenceClassification, DistilBertTokenizerFast, get_linear_schedule_with_warmup,  AdamW, AutoConfig
from torch.utils import data
from torch.optim import RMSprop, Adam
from safetensors.torch import load_file

#---------------Initialize---------------#
TU = utility.TextUtility()
TU.initialize_utility()
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)


#---------------Paths---------------#

_base = os.getcwd()
_data_dir = os.path.join(_base, "preprocessed-data")

# Training and Validation data
preprocessed_data_path = os.path.join(_data_dir, "preprocessed.csv")
preprocessed_data = pd.read_csv(preprocessed_data_path)
preprocessed_data.drop(labels=["Age of User", "Country", "selected_text"], axis=1, inplace=True)

# Test data
preprocessed_test_data_path = os.path.join(_data_dir, "preprocessed_test_data.csv")
preprocessed_test_data = pd.read_csv(preprocessed_test_data_path)
preprocessed_test_data.drop(labels=["Age of User", "Country"], axis=1, inplace=True)

preprocessed_data.dropna(inplace=True)
preprocessed_test_data.dropna(inplace=True)
#---------------Dataset---------------#

preprocessed_data['text'].astype(str)
preprocessed_test_data['text'].astype(str)


preprocessed_data['sentiment'] = preprocessed_data['sentiment'].astype('category').cat.codes
preprocessed_test_data['sentiment'] = preprocessed_test_data['sentiment'].astype('category').cat.codes

print(preprocessed_data.head())
print(preprocessed_test_data.head())
# preprocessed_data = preprocessed_data.iloc[1:5000]
print(preprocessed_data['text'].isna().value_counts())
print(preprocessed_data['sentiment'].isna().value_counts())

print(preprocessed_test_data['text'].isna().value_counts())
print(preprocessed_test_data['sentiment'].isna().value_counts())



# Verify number of classes
num_classes = len(preprocessed_data['sentiment'].unique())
print(f"Number of classes: {num_classes}")
#---------------Custom Dataset For Pytorch---------------#

train_text, val_text, train_sentiment, val_sentiment = train_test_split(
    preprocessed_data["text"].to_numpy(), 
    preprocessed_data['sentiment'].to_numpy(), 
    test_size=0.1
)



test_text, test_sentiment = preprocessed_test_data['text'].to_numpy(), preprocessed_test_data['sentiment'].to_numpy()

tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
train_encodings = tokenizer(list(train_text), truncation=True, padding=True)
val_encodings = tokenizer(list(val_text), truncation=True, padding=True)
test_encodings = tokenizer(list(test_text), truncation=True, padding=True)


class SentimentDataset(data.Dataset):
    def __init__(self , encodings, labels ):
        self.encodings = encodings
        self.sentiment = labels
        
    
    def __len__(self):
        return len(self.sentiment)
    
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['sentiment'] = torch.tensor(self.sentiment[idx], dtype=torch.long) 
        
    
        return item

# Hyper-parameters
BATCH_SIZE = 24
EPOCHS = 5

train_dataset = SentimentDataset(train_encodings, train_sentiment)
val_dataset = SentimentDataset(val_encodings, val_sentiment)
test_dataset = SentimentDataset(test_encodings, test_sentiment)

#--------------DataLoader--------------#
train_loader = data.DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = data.DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_loader = data.DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# Print for error checking
data = next(iter(train_loader))
print(data.keys())
print(data['input_ids'].size())
print(data['attention_mask'].size())
print(data['sentiment'].size())

data = next(iter(test_loader))
print(data.keys())
print(data['input_ids'].size())
print(data['attention_mask'].size())
print(data['sentiment'].size())

#---------------Model Definition---------------#

#configuration = AutoConfig.from_pretrained('distilbert-base-uncased') - used to train from scratch
configuration = AutoConfig.from_pretrained('models/config.json')

configuration.dropout=0.3
configuration.attention_dropout=0.2

configuration.num_labels = num_classes

#model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', config=configuration) - used to train from scratch

model = DistilBertForSequenceClassification(configuration)
weights = load_file('models/model.safetensors')
model.load_state_dict(weights)
model = model.to('cuda' if torch.cuda.is_available() else 'cpu')




#---------------Training Functionality---------------#

def train_epoch(model, data_loader, optimizer, scheduler,device, n_examples):
    model.train()
    # losses = []
    correct_predictions = 0
    
    for d in tqdm(data_loader, desc="Training"):
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        sentiments = d["sentiment"].to(device) 

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=sentiments
        )

        loss, logits = outputs.loss, outputs.logits
        
        _, preds = torch.max(logits, dim=1)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=2)

        correct_predictions += torch.sum(preds == sentiments)
        optimizer.zero_grad()
        loss.backward()
        

        optimizer.step()
        scheduler.step()
        

    return correct_predictions.double() / n_examples

def eval_model(model, data_loader, device, n_examples):
    
    correct_predictions = 0
    
    model.eval()
    with torch.inference_mode():
        for d in tqdm(data_loader, desc="Evaluating"):
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            sentiments = d["sentiment"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=sentiments
            )
            
            loss, logits = outputs['loss'], outputs['logits']

            _, preds = torch.max(logits, dim=1)

            correct_predictions += torch.sum(preds == sentiments)
            
    return correct_predictions.double() / n_examples

#init
device = 'cuda' if torch.cuda.is_available() else 'cpu'


optimizer = AdamW(params=model.parameters(), lr=5e-6, eps=1e-03, weight_decay=3)
num_training_steps = EPOCHS * len(train_loader)
num_warmup_steps = int(num_training_steps*0.1)
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=num_warmup_steps, 
    num_training_steps=num_training_steps
)

#---------------Test before training-----------------#

#test set
test_acc = eval_model(
    model,
    test_loader,
        device,
    len(test_dataset)
)


print(f'Test accuracy {test_acc}')

#---------------Actual Training---------------#
for epoch in tqdm(range(EPOCHS)):
    print(f'Epoch {epoch + 1}/{EPOCHS}')    
    print('-' * 10)

    train_acc = train_epoch(
        model,
        train_loader,
        
        optimizer,
        scheduler,
        device,
        
        len(train_dataset)
    )

    print(f'Train accuracy {train_acc}')

    val_acc = eval_model(
        model,
        val_loader,
        
        device,
        len(val_dataset)
    )
    # if val_loss < best_loss:
    #     best_loss = val_loss
    #     torch.save(model, 'model/Senlyzer_beta.pth')

    

    print(f'Validation accuracy {val_acc}')

    #test set
    test_acc = eval_model(
        model,
        test_loader,
            device,
        len(test_dataset)
    )


    print(f'Test accuracy {test_acc}')

    print()


print("end")


model.save_pretrained("models_new/")