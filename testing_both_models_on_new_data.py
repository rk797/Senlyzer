import torch
import module.utility as utility
import numpy as np
import pandas as pd

from matplotlib import rc
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from torch import nn, optim
from transformers import  BertModel, AutoConfig, DistilBertForSequenceClassification, RobertaForSequenceClassification, RobertaTokenizerFast, DistilBertTokenizerFast, get_linear_schedule_with_warmup
from torch.utils import data
from torch.optim import AdamW
from torch.utils.checkpoint import checkpoint
from torch.cuda.amp import GradScaler, autocast

#---------------Initialize---------------#
TU = utility.TextUtility()
TU.initialize_utility()
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

#---------------Custom Dataset For Pytorch Model Ensembling---------------#
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
    
    
#---------------Custom Dataset For Pytorch Model BERT---------------#
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


preprocessed_test_data = pd.read_csv('dataset_for_model_testing/preprocessed_twitter_validation.csv')


preprocessed_test_data['text'].astype(str)
preprocessed_test_data['sentiment'] = preprocessed_test_data['sentiment'].astype('category').cat.codes


test_text, test_sentiment = preprocessed_test_data['text'].to_numpy(), preprocessed_test_data['sentiment'].to_numpy()


#-------------------Defining Tokenizers------------------#
distilbert_tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

roberta_tokenizer = RobertaTokenizerFast.from_pretrained('FacebookAI/roberta-base')

test_dataset_model_ensemble = SentimentDataset(test_text , test_sentiment, distilbert_tokenizer, roberta_tokenizer, max_length=MAX_LENGTH)
test_dataset_bert = TweetsentimentDataset(test_text, test_sentiment, TU.tokenizer, MAX_LENGTH)

#--------------DataLoader--------------#

test_loader_model_ensemble = data.DataLoader(
    test_dataset_model_ensemble,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

test_loader_bert = data.DataLoader(
    test_dataset_bert,
    batch_size=BATCH_SIZE,
    shuffle=False,

)


class SentimentClassifierBERT(nn.Module):
    def __init__(self, n_classes):
        super(SentimentClassifierBERT, self).__init__()
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
    

class SentimentClassifierEnsemble(nn.Module):
    def __init__(self,configuration_distilbert_model,  configuration_roberta_model, n_classes):
        super(SentimentClassifierEnsemble, self).__init__()
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

# BERT MODEl --------------------------------------------
bert_model = SentimentClassifierBERT(n_classes=3)
bert_model.load_state_dict(torch.load('bert_model/Senlyzer_bert_with_decay'))
bert_model = bert_model.to(device)


# Model Ensemble --------------------------------------------

configuration_distilbert_model = AutoConfig.from_pretrained('distilbert-base-uncased')
configuration_distilbert_model.dropout=0.3
configuration_distilbert_model.num_labels = 768

configuration_roberta_model = AutoConfig.from_pretrained('FacebookAI/roberta-base')
configuration_roberta_model.dropout=0.3
configuration_roberta_model.num_labels = 768

model = SentimentClassifierEnsemble(configuration_distilbert_model = configuration_distilbert_model, configuration_roberta_model= configuration_roberta_model,n_classes=3)
model.load_state_dict(torch.load('ensemble_model/model_from_ensemble_2'))

model = model.to(device)


loss_fn = nn.CrossEntropyLoss().to(device)


def eval_model_ensemble(model, data_loader, loss_fn, device, n_examples):
    
    losses = []
    correct_predictions = 0
    
    
    model= model.eval()
    with torch.inference_mode():
        for batch in tqdm(data_loader, desc="Evaluating"):

            distilbert_inputs, roberta_inputs, sentiments = batch
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

def eval_model_bert(model, data_loader, loss_fn, device, n_examples):
    
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

if __name__ =="__main__":
    #test set BERT
    test_acc, test_loss = eval_model_bert(
        bert_model,
        test_loader_bert,
        loss_fn,
        device,
        len(preprocessed_test_data)
    )

    print(f' BERT Test accuracy {test_acc*100}% | loss {test_loss}')

    #test set
    test_acc, test_loss = eval_model_ensemble(
        model,
        test_loader_model_ensemble,
        loss_fn,
        device,
        len(preprocessed_test_data)
    )


    print(f'Model Ensemble Test accuracy {test_acc*100}% | loss {test_loss}')