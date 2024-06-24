import os
import torch
import module.utility as utility
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from matplotlib import rc
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from torch import nn, optim
from transformers import BertTokenizer, BertModel, BertForSequenceClassification, get_linear_schedule_with_warmup, AdamW
from torch.utils import data
from torch.optim import RMSprop
from pylab import rcParams

#---------------Model Definition---------------#
class SentimentClassifier(nn.Module):
    def __init__(self, n_classes):
        super(SentimentClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.drop = nn.Dropout(p=0.50)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)
        self.sofmax = nn.Softmax(dim=1)
    
    def forward(self, input_ids, attention_mask):
        pooled_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )[1]
        output = self.drop(pooled_output)
        output = self.out(output)
        output = self.sofmax(output)
        return output

#---------------Training Functionality---------------#

def train_epoch(model, data_loader, loss_fn, optimizer, device, scheduler, n_examples):
    model= model.train()
    losses = []
    correct_predictions = 0

    for d in tqdm(data_loader, desc="Training"):
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        sentiments = d["sentiment"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        _, preds = torch.max(outputs, dim=1)
        loss = loss_fn(outputs, sentiments)

        correct_predictions += torch.sum(preds == sentiments)
        losses.append(loss.item())

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()

    return correct_predictions.double() / n_examples, np.mean(losses)

def eval_model(model, data_loader, loss_fn, device, n_examples):
    
    losses = []
    correct_predictions = 0
    
    
    model= model.eval()
    with torch.inference_mode():
        for d in tqdm(data_loader, desc="Evaluating"):
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            sentiments = d["sentiment"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            _, preds = torch.max(outputs, dim=1)
            
            loss = loss_fn(outputs, sentiments)

            correct_predictions += torch.sum(preds == sentiments)
            losses.append(loss.item())

    return correct_predictions.double() / n_examples, np.mean(losses)


if __name__ == "__main__":
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

    preprocessed_data['sentiment'] = preprocessed_data['sentiment'].astype('category').cat.codes
    preprocessed_test_data['sentiment'] = preprocessed_test_data['sentiment'].astype('category').cat.codes

    #---------------Custom Dataset For Pytorch---------------#
    class TweetsentimentDataset(data.Dataset):
        def __init__(self, text, sentiment, tokenizer, max_len):
            self.text = text
            self.sentiment = sentiment
            self.tokenizer = tokenizer
            self.max_len = max_len
        
        def __len__(self):
            return len(self.text)
        
        def __getitem__(self, item):
            text = str(self.text[item])
            encoding = self.tokenizer.encode_plus(
                text,
                max_length=self.max_len,
                add_special_tokens=True,
                padding='max_length',
                return_attention_mask=True,
                return_token_type_ids=False,
                return_tensors='pt' 
            )
        
            return {
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'sentiment': torch.tensor(self.sentiment[item], dtype=torch.long)
            }

    # Hyper-parameters
    MAX_LENGTH = 128
    BATCH_SIZE = 16
    EPOCHS = 3

    train_text, val_text, train_sentiment, val_sentiment = train_test_split(
        preprocessed_data["text"].to_numpy(), 
        preprocessed_data['sentiment'].to_numpy(), 
        test_size=0.2
    )

    train_dataset = TweetsentimentDataset(train_text, train_sentiment, TU.tokenizer, MAX_LENGTH)
    val_dataset = TweetsentimentDataset(val_text, val_sentiment, TU.tokenizer, MAX_LENGTH)
    test_dataset = TweetsentimentDataset(preprocessed_test_data['text'].to_numpy() , preprocessed_test_data['sentiment'].to_numpy(), TU.tokenizer, MAX_LENGTH)

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
    
    #---------------Model Definition---------------#
    model = SentimentClassifier(n_classes=preprocessed_data['sentiment'].nunique())
    model.load_state_dict(torch.load('models/Senlyzer_bert_decay_5'))
    model = model.to('cuda' if torch.cuda.is_available() else 'cpu')
    
    #init
    #use cuda for hardware acceleration if available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    loss_fn = nn.CrossEntropyLoss().to(device)
    optimizer =  AdamW(model.parameters(), lr=2e-5, correct_bias=True, weight_decay=0.1)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=0, 
        num_training_steps=total_steps
        )
    best_loss = 1000000
    
    for epoch in range(EPOCHS):
        print(f'Epoch {epoch + 1}/{EPOCHS}')    
        print('-' * 10)

        train_acc, train_loss = train_epoch(
            model,
            train_loader,
            loss_fn,
            optimizer,
            device,
            scheduler,
            len(train_dataset)
        )

        print(f'Train loss {train_loss} accuracy {train_acc}')

        val_acc, val_loss = eval_model(
            model,
            val_loader,
            loss_fn,
            device,
            len(val_dataset)
        )
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), 'models/Senlyzer_bert_with_decay_3')

        print(f'Validation loss {val_loss} accuracy {val_acc}')
        print()

    #test set
    test_acc, test_loss = eval_model(
        model,
        test_loader,
        loss_fn,
        device,
        len(test_dataset)
    )



