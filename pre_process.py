"""# Data Pre-processing"""

import pandas as pd
import os
import module.utility as utility
from colorama import Fore, init
init()

# LINE_COUNT = 100

#-------------------INITIALIZE-------------------#
TU = utility.TextUtility()
TU.initialize_utility()
print(Fore.GREEN + '[+] Utility initialized')

#---------------------PATHS----------------------#
_base = os.getcwd()
_data_dir = os.path.join(_base, "unprocessed-data")
_train_data_path = os.path.join(_data_dir, "train.csv")

#load as data frame
train_df = pd.read_csv(_train_data_path,  encoding='latin1' ) #nrows=LINE_COUNT

print(Fore.GREEN + '[+] Loaded csv')

# Drop:
# - TextID
# - Population
# - Land area
# - Density
train_df.drop(labels=["Population -2020","Land Area", "Density", "textID","Time of Tweet"], axis=1, inplace=True)

print(Fore.GREEN + '[+] Dropped columns')
train_df.to_json()

#Checking for null/Na values
train_df.dropna(axis=0,inplace=True)

for col in train_df:
    train_df[col]=train_df[col].astype(str)

train_df['selected_text']=train_df['selected_text'].str.lower()
train_df.to_json()
print(Fore.GREEN + '[+] Starting pre-processing')
train_df = TU.calculate_exec_time(TU.preprocess_dataframe, train_df, "text")
print(Fore.GREEN + '[+] Finished')
train_df.to_csv("preprocessed-data/preprocessed.csv")
