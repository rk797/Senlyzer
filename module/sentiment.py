import sys
import os
path = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
sys.path.append(os.path.join(path, "Senlyzer"))
from train_BERT import SentimentClassifier
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration, CLIPProcessor, CLIPModel
import torch
import module.utility as utility
import numpy as np

class Senlyzer:
    def __init__(self):
        #intialize text preprocess
        self.__preprocess = utility.TextUtility()
        self.__preprocess.initialize_utility()
        
        #transformer for zero-short image sentiment
        self.__image_sentiment_model = CLIPModel.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
        self.__image_sentiment_processor = CLIPProcessor.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
        
        #load trained text sentiment model
        self.__text_senti = SentimentClassifier(n_classes=3)
        self.__text_senti.load_state_dict(torch.load('models/Senlyzer_bert_with_decay'))
        
        self.__image = None
        self.__text = ""
        
        #transformer for image-captioning
        self.__image_caption_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
        self.__image_caption_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
        
    def image_input(self, path:str):
        img_url = rf"{path}"
        self.__image = Image.open(img_url).convert('RGB')
    def text_input(self, text):
        self.__text = str(text)

    def text_sentiment(self):
        if self.__text != "":
            token_ids, attention_mask = self.__preprocess.tokenize_text(self.__text)
            with torch.no_grad():
                output = self.__text_senti(token_ids, attention_mask)
            score, prediction = torch.max(output, dim=1)
            
            score = score.detach().numpy()
            np.set_printoptions(suppress=True)
            if (prediction == 0):
                prediction = "NEGATIVE"
            elif(prediction==1):
                prediction= "NEUTRAL"
            else:
                prediction="POSITIVE"
            return score, prediction
        
        else:
            print("---No text input---")
    
    def image_sentiment(self):
        if self.__image is not None:
            inputs = self.__image_sentiment_processor(text=["Negative", "Neutral", "Positive"], images=self.__image, return_tensors="pt", padding=True)
            outputs = self.__image_sentiment_model(**inputs)
            logits_per_image = outputs.logits_per_image
            scores = logits_per_image.softmax(dim=1)
            
            scores = scores.detach().numpy()
            np.set_printoptions(suppress=True)
            score = np.max(scores)
            prediction = np.where(scores==score)[0]
            prediction = prediction[0]
            print(prediction)
            if prediction == 0:
                prediction = "NEGATIVE"
            elif prediction ==1:
                prediction= "NEUTRAL"
            else:
                prediction="POSITIVE"
            return score, prediction
        else:
            print("---No image input---")
            
    def text_img_sentiment(self):
        if self.__text != "" and self.__image is not None:
            text_token, text_attention_mask = self.__preprocess.tokenize_text(self.__text)
            with torch.no_grad():
                ouput_text = self.__text_senti(text_token, text_attention_mask)
            scores_text = ouput_text[0].numpy()
            
            inputs = self.__image_sentiment_processor(text=["Negative", "Neutral", "Positive"], images=self.__image, return_tensors="pt", padding=True)
            outputs = self.__image_sentiment_model(**inputs)
            logits_per_image = outputs.logits_per_image
            scores_image = logits_per_image.softmax(dim=1)
            scores_image = scores_image.detach().numpy()
            np.set_printoptions(suppress=True)
            scores_image = scores_image[0]
            
            prediction_text = np.where(scores_text == np.max(scores_text))[0]
            prediction_text = prediction_text[0]
            
            prediction_image = np.where(scores_image == np.max(scores_image))[0]
            prediction_image = prediction_image[0]

            #average score of text and image
            scores = np.add(scores_text, scores_image)
            scores = np.divide(scores, 2)
            
            if prediction_text == prediction_image:
                #if the sentiment for text==image than take average max
                score = np.max(scores)
                prediction = np.where(scores==score)[0]
                prediction = prediction[0]
                if prediction == 0:
                    prediction = "NEGATIVE"
                elif prediction == 1:
                    prediction= "NEUTRAL"
                else:
                    prediction="POSITIVE"
                return score, prediction
            
            else:   
                #take score of text combine with caption of image
                inputs_caption = self.__image_caption_processor(self.__image, return_tensors="pt")
                out = self.__image_caption_model.generate(**inputs_caption)
                image_caption = self.__image_caption_processor.decode(out[0], skip_special_tokens=True)
                combine_text = self.__text +" and "+ image_caption
                
                text_token, text_attention_mask = self.__preprocess.tokenize_text(combine_text)
                with torch.no_grad():
                    ouput_text = self.__text_senti(text_token, text_attention_mask)
                scores_combine_text = ouput_text[0].numpy()
                
                #average score of combined text and the average of text and image
                average_score = np.add(scores, scores_combine_text)
                average_score = np.divide(average_score, 2)
                
                score = np.max(average_score)
                prediction = np.where(average_score==score)[0]
                prediction = prediction[0]
                print(prediction)
                if prediction == 0:
                    prediction = "NEGATIVE"
                elif prediction == 1:
                    prediction= "NEUTRAL"
                else:
                    prediction="POSITIVE"
                return score, prediction
        else:
            print("---Missing input---")
