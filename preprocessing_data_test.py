"""# Data Pre-processing"""

import pandas as pd
import module.utility as utility
from colorama import Fore, init
init()

# LINE_COUNT = 100

#-------------------INITIALIZE-------------------#
TU = utility.TextUtility()
TU.initialize_utility()
print(Fore.GREEN + '[+] Utility initialized')

#---------------------PATHS----------------------#

_train_data_path = "dataset_for_model_testing/twitter_validation.csv"
#load as data frame
train_df = pd.read_csv(_train_data_path,  encoding='latin1' ) #nrows=LINE_COUNT

print(Fore.GREEN + '[+] Loaded csv')

# Drop:

train_df.drop(labels=['tweet_id', 'entity'],  axis=1,inplace=True)

# Dropping irrelevant 
train_df.drop( train_df[train_df['sentiment'] =='Irrelevant'].index,inplace=True)

train_df['sentiment'] = train_df['sentiment'].str.lower()

print(Fore.GREEN + '[+] Dropped columns')
train_df.to_json()


for col in train_df:
    train_df[col]=train_df[col].astype(str)

train_df.to_json()
print(Fore.GREEN + '[+] Starting pre-processing')
train_df = TU.calculate_exec_time(TU.preprocess_dataframe, train_df, "text")
print(Fore.GREEN + '[+] Finished')
train_df.to_csv("dataset_for_model_testing/preprocessed_twitter_validation.csv")
