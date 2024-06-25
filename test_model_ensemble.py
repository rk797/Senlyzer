import os
import torch
import module.utility as utility
import numpy as np
import pandas as pd

from matplotlib import rc
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from torch import nn, optim
from transformers import AutoConfig, DistilBertForSequenceClassification, RobertaForSequenceClassification, RobertaTokenizerFast, DistilBertTokenizerFast, get_linear_schedule_with_warmup
from torch.utils import data
from torch.optim import AdamW
from torch.utils.checkpoint import checkpoint
from torch.cuda.amp import GradScaler, autocast


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
#---------------Spelitting the dataset---------------#

train_text, val_text, train_sentiment, val_sentiment = train_test_split(
    preprocessed_data["text"].to_numpy(), 
    preprocessed_data['sentiment'].to_numpy(), 
    test_size=0.1
)

test_text, test_sentiment = preprocessed_test_data['text'].to_numpy(), preprocessed_test_data['sentiment'].to_numpy()


#-------------------Defining Tokenizers------------------#
distilbert_tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

roberta_tokenizer = RobertaTokenizerFast.from_pretrained('FacebookAI/roberta-base')



#---------------Custom Dataset For Pytorch---------------#
class SentimentDataset(data.Dataset):
    def __init__(self, texts, labels, distilbert_tokenizer, roberta_tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.distilbert_tokenizer = distilbert_tokenizer
        self.roberta_tokenizer = roberta_tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        distilbert_inputs = self.distilbert_tokenizer(text, padding='max_length', truncation=True, max_length=self.max_length, return_tensors='pt')
        roberta_inputs = self.roberta_tokenizer(text, padding='max_length', truncation=True, max_length=self.max_length, return_tensors='pt')
        
        distilbert_inputs = {key: val.squeeze() for key, val in distilbert_inputs.items()}
        roberta_inputs = {key: val.squeeze() for key, val in roberta_inputs.items()}
        
        return [distilbert_inputs, roberta_inputs, torch.tensor(label, dtype=torch.long)]
    
       

# Hyper-parameters
MAX_LENGTH = 128
BATCH_SIZE = 16
EPOCHS = 5


train_dataset = SentimentDataset(train_text, train_sentiment, distilbert_tokenizer, roberta_tokenizer, max_length=MAX_LENGTH)
val_dataset = SentimentDataset(val_text, val_sentiment, distilbert_tokenizer, roberta_tokenizer, max_length=MAX_LENGTH)
test_dataset = SentimentDataset(test_text , test_sentiment, distilbert_tokenizer, roberta_tokenizer, max_length=MAX_LENGTH)


#--------------DataLoader--------------#
train_loader = data.DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

val_loader = data.DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False, 
    num_workers=4,
    pin_memory=True
)

test_loader = data.DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

#---------------Config setup---------------#
configuration_distilbert_model = AutoConfig.from_pretrained('distilbert-base-uncased')
configuration_distilbert_model.dropout=0.3
configuration_distilbert_model.num_labels = 768

configuration_roberta_model = AutoConfig.from_pretrained('FacebookAI/roberta-base')
configuration_roberta_model.dropout=0.3
configuration_roberta_model.num_labels = 768

#---------------Model Definition---------------#
class SentimentClassifier(nn.Module):
    def __init__(self,configuration_distilbert_model,  configuration_roberta_model, n_classes):
        super(SentimentClassifier, self).__init__()
        self.distilbert_model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', config=configuration_distilbert_model)
        self.roberta_model = RobertaForSequenceClassification.from_pretrained('FacebookAI/roberta-base', config=configuration_roberta_model)
        self.drop = nn.Dropout(p=0.50)
        self.classifier = nn.Linear(int((self.distilbert_model.config.hidden_size + self.roberta_model.config.hidden_size)/2), n_classes)  # Adjust the output size as per your number of classes
    
    
    def forward(self, input_ids_1, attention_mask_1, input_ids_2, attention_mask_2):
        def distilbert_forward(input_ids, attention_mask):
            return self.distilbert_model(input_ids=input_ids, attention_mask=attention_mask).logits
        
        def roberta_forward(input_ids, attention_mask):
            return self.roberta_model(input_ids=input_ids, attention_mask=attention_mask).logits
        
        # Use checkpoint to wrap these functions
        distilbert_outputs = checkpoint(distilbert_forward, input_ids_1, attention_mask_1, use_reentrant=False)
        roberta_outputs = checkpoint(roberta_forward, input_ids_2, attention_mask_2, use_reentrant=False)
        
        # Average the logits
        combined_logits = (distilbert_outputs + roberta_outputs) / 2


        output = self.drop(combined_logits)
             
        output = self.classifier(output)
        return output
    
model = SentimentClassifier(configuration_distilbert_model = configuration_distilbert_model, configuration_roberta_model= configuration_roberta_model,n_classes=preprocessed_data['sentiment'].nunique())
model.load_state_dict(torch.load('ensemble_model/model_from_ensemble_2'))

model = model.to('cuda' if torch.cuda.is_available() else 'cpu')
#printing the model to view its architecture
print(model)
print(torch.cuda.is_available())
#---------------Training Functionality---------------#


scaler = GradScaler() 

'''
Applying Mixed Precision Training using PyTorch provides torch.cuda.amp 
for easy implementation of mixed precision training, which involves two main components:
autocast: Automatically casts operations to 16-bit where possible.
GradScaler: Scales gradients to prevent underflow in 16-bit precision. 

This reduces memory usage and increases computation speed
'''

def train_epoch(model, data_loader, loss_fn, optimizer, device, scheduler, n_examples):
    model= model.train()
    losses = []
    correct_predictions = 0

    for batch  in tqdm(data_loader, desc="Training"):
        distilbert_inputs, roberta_inputs, sentiments = batch
            # distilbert_inputs, roberta_inputs, sentiments = distilbert_inputs.to(device), roberta_inputs.to(device), sentiments.to(device)
        with autocast():
            outputs = model(input_ids_1=distilbert_inputs['input_ids'].to(device),
                                        attention_mask_1=distilbert_inputs['attention_mask'].to(device),
                                        input_ids_2=roberta_inputs['input_ids'].to(device),
                                        attention_mask_2=roberta_inputs['attention_mask'].to(device)).to(device)
        
            preds = torch.argmax(outputs, dim=1).to(device)
            sentiments= sentiments.to(device)
            loss = loss_fn(outputs, sentiments)

        correct_predictions += torch.sum(preds == sentiments)
        losses.append(loss.item())

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        scheduler.step()

    return correct_predictions.double() / n_examples, np.mean(losses)

def eval_model(model, data_loader, loss_fn, device, n_examples):
    
    losses = []
    correct_predictions = 0
    
    
    model= model.eval()
    with torch.inference_mode():
        for batch in tqdm(data_loader, desc="Evaluating"):

            distilbert_inputs, roberta_inputs, sentiments = batch
            # distilbert_inputs, roberta_inputs, sentiments = distilbert_inputs.to(device), roberta_inputs.to(device), sentiments.to(device)

            outputs = model(input_ids_1=distilbert_inputs['input_ids'].to(device),
                                    attention_mask_1=distilbert_inputs['attention_mask'].to(device),
                                    input_ids_2=roberta_inputs['input_ids'].to(device),
                                    attention_mask_2=roberta_inputs['attention_mask'].to(device)).to(device)
            
            preds = torch.argmax(outputs, dim=1).to(device)
            sentiments= sentiments.to(device)
            loss = loss_fn(outputs, sentiments)

            correct_predictions += torch.sum(preds == sentiments)
            losses.append(loss.item())


    return correct_predictions.double() / n_examples, np.mean(losses)

#init
#use cuda for hardware acceleration if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
loss_fn = nn.CrossEntropyLoss().to(device)
optimizer =  AdamW(model.parameters(), lr=2e-5,  weight_decay=1)
total_steps = len(train_loader) * EPOCHS
num_warmup_steps = int(0.1 * total_steps)
scheduler = get_linear_schedule_with_warmup(
    optimizer, 
    num_warmup_steps=num_warmup_steps, 
    num_training_steps=total_steps
    )


if __name__ == "__main__":
    #test set - before training
    test_acc, test_loss = eval_model(
        model,
        test_loader,
        loss_fn,
        device,
        len(test_dataset)
    )


    print(f'Test accuracy {test_acc*100}% | loss {test_loss}')
    
    best_acc = test_acc

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

        print(f'Train accuracy {train_acc*100}% | loss {train_loss}')

        val_acc, val_loss = eval_model(
            model,
            val_loader,
            loss_fn,
            device,
            len(val_dataset)
        )
        
        print(f'Validation accuracy {val_acc *100}% | loss {val_loss}')
        print()

        #test set
        test_acc, test_loss = eval_model(
            model,
            test_loader,
            loss_fn,
            device,
            len(test_dataset)
        )


        print(f'Test accuracy {test_acc*100}% | loss {test_loss}')

        # if test_acc > best_acc:
        #     best_acc = test_acc
        #     torch.save(model.state_dict(), 'ensemble_model/model_from_ensemble_3')

    print("end")



