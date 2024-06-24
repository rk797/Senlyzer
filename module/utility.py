import spacy
import pandas as pd
import re
import nltk
import ssl
import importlib

from nltk.corpus import stopwords
from contractions import fix
from textblob import TextBlob
from timeit import default_timer as timer
from transformers import BertTokenizer
from tqdm import tqdm
from os import cpu_count
from functools import lru_cache
from colorama import Fore, init
import emoji
import demoji

class TextUtility:
    slangs_df = pd.read_csv("slangs_processing/internet-slangs-to-normal-terms.csv")
    translation_table = str.maketrans({
        '□': " ",
        '�': None
    })

    def is_package_installed(self, package_name):
        """Check if a package is installed."""
        package_spec = importlib.util.find_spec(package_name)
        return package_spec is not None

    def __init__(self):
        self.nlp = None
        self.stop_words = set()
        self.tokenizer = None
        self.text = ""

    def initialize_utility(self):
        init()

        if not self.is_package_installed('nltk'):
            ssl._create_default_https_context = ssl._create_unverified_context
            nltk.download('stopwords')
        
        if self.nlp is None:
            if not self.is_package_installed('en_core_web_sm'):
                spacy.cli.download('en_core_web_sm')
            self.nlp = spacy.load("en_core_web_sm")
        
        if not self.stop_words:
            self.stop_words = set(stopwords.words('english'))

        if self.tokenizer is None:
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        tqdm.pandas()

    @lru_cache(maxsize=1024)
    def rectify_word_for_letter_reps(self, word, max_repeats=2):
        """Reduce character repetitions to a maximum number of allowed repeats."""
        if word != "" or word != " ":
            return re.sub(r'(.)\1{2,}', r'\1' * max_repeats, word)
        return word

    def rectify_word_letter_repetitions(self, max_repeats=2):
        """Reduce character repetitions in the entire text."""
        text_splitted = self.text.split(" ")
        rectified_words = [self.rectify_word_for_letter_reps(word, max_repeats) for word in text_splitted]
        self.text = " ".join(rectified_words)
        

    def rectify_internet_slangs(self):
        text_splitted = self.text.split(" ")
        for i in range(len(text_splitted)):
            slang_index = self.slangs_df.index[self.slangs_df["Slang"] == text_splitted[i]].to_list()
            if slang_index:
                text_splitted[i] = self.slangs_df.loc[slang_index[0], "Meaning"]
        self.text = " ".join(text_splitted)

    def remove_special_characters(self):
        url_pattern = re.compile(r'https?:\/\/\S*|www\.\S+')
        html_pattern = re.compile(r'<.*?>')
        mention_pattern = re.compile(r'@\S*')
        number_pattern = re.compile(r'[A-Za-z]*\d{1,3}[A-Za-z]*\d{1,10}|\d{10}')

        self.text = url_pattern.sub('URL', self.text)
        self.text = html_pattern.sub('', self.text)
        self.text = mention_pattern.sub('user', self.text)
        self.text = number_pattern.sub('NUMBER', self.text)
      

    def remove_punctuation(self):
        punctuation_pattern = re.compile(r'[^\w\s]')
        self.text = punctuation_pattern.sub('', self.text)
        self.text= self.text + " "

    def remove_numbers(self):
        number_pattern = re.compile(r'\d+')
        self.text = number_pattern.sub('', self.text)

    def remove_stopwords(self, tokens):
        if not all(isinstance(token, str) for token in tokens):
            tokens = [str(token) for token in tokens]

        # Remove stopwords
        return [token for token in tokens if token.lower() not in self.stop_words]
        


    def lemmatize_text(self, tokens, level="full"):
        lemmatized_tokens = []
        for token in tokens:
            token_doc = self.nlp(token)
            if level == "none":
                lemmatized_tokens.append(token)
            elif level == "partial":
                lemmatized_tokens.append(token if token_doc[0].pos_ in {"PROPN", "NUM"} else token_doc[0].lemma_)
            else:  # "full"
                lemmatized_tokens.append(token_doc[0].lemma_)
        return lemmatized_tokens

    def tokenize_text(self, text):
        encoding = self.tokenizer.encode_plus(
            text,
            max_length=128,
            add_special_tokens=True,
            padding='max_length',
            return_attention_mask=True,
            return_token_type_ids=False,
            return_tensors='pt'
        )
        token_ids = encoding["input_ids"]
        attention_mask = encoding["attention_mask"]
        return token_ids, attention_mask

    def tag_named_entities(self, doc):
        tagged_tokens = []
        for token in doc:
            if token.ent_type_:
                tagged_tokens.append(f"{token.ent_type_.lower()}_{token.text}")
            else:
                tagged_tokens.append(token.text)
        return tagged_tokens

   

    def expand_contractions(self):
        self.text = fix(self.text)

    def remove_extra_whitespace(self):
        whitespace_pattern = re.compile(r'\s+')
        self.text = whitespace_pattern.sub(' ', self.text).strip()

    # def correct_spelling(self):
    #     spellings_rectified = TextBlob(self.text)
    #     self.text = str(spellings_rectified.correct())
    
    def replace_emojis(self):
        self.text = emoji.demojize(self.text)
        self.text = demoji.replace(self.text)

    def preprocess_text(self, text):
        og_text = text
        self.text = text.lower()
        self.text = self.text.translate(self.translation_table)
        self.remove_extra_whitespace()
        self.rectify_internet_slangs()
        # self.correct_spelling()
        self.expand_contractions()
        self.remove_special_characters()
        self.remove_numbers()
        self.remove_punctuation()
        self.remove_extra_whitespace()
        self.replace_emojis()

        # Reduce character repetitions with a maximum allowed repeats
        self.rectify_word_letter_repetitions(max_repeats=2)

        doc = self.nlp(self.text)
        tokens = [token.text for token in doc]
        
        #tokens= self.remove_stopwords(doc)

        # Tag named entities
        tokens = self.tag_named_entities(doc)
        tokens = self.lemmatize_text(tokens, "none")
        self.text = " ".join(tokens)
        print(Fore.BLUE + og_text, end="--")
        print(Fore.GREEN + self.text)

        return self.text

    def preprocess_dataframe(self, df, text_column):
        df[text_column] = df[text_column].progress_apply(self.preprocess_text)
        return df

    def calculate_exec_time(self, func, *args):
        start_time = timer()
        result = func(*args)
        end_time = timer()
        delta = end_time - start_time
        print(Fore.BLUE + f"[+] Execution time of {func.__name__}: {delta:.6f} seconds")
        return result