import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import module.utility  as utility 



#-------------------INITIALIZE-------------------#
TU = utility.TextUtility()
TU.initialize_utility()


#---------------------PATHS----------------------#
_base = os.getcwd()
_data_dir = os.path.join(_base, "unprocessed-data")
_test_data_path = os.path.join(_data_dir, "test.csv")
test_df = pd.read_csv(_test_data_path, encoding="latin1")

#--------------------CLEANING--------------------#
print(test_df.head())

print(test_df.isna().value_counts())

print(len(test_df))

# Removing Na values
test_df.dropna(inplace=True)

print(len(test_df))

# Drop:
# - TextID
# - Population
# - Land area
# - Density
test_df.drop(labels=["Population -2020","Land Area", "Density", "textID","Time of Tweet"], axis=1, inplace=True)

print(test_df.head())

for col in test_df:
    test_df[col]=test_df[col].astype(str)


test_df.to_json()
test_df = TU.calculate_exec_time(TU.preprocess_dataframe, test_df, "text")

test_df.to_csv("preprocessed-data/preprocessed_test_data.csv")