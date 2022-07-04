'''
Step 1 - Get breastcancer dataset

Fetches breastcancer dataset, creates necessary file structure, and applies some
preprocessing to it.

Based on code from Bekker.
'''


import numpy as np
import os
import polars as pl
import pandas as pd
import requests

import sklearn.model_selection

# Names and locations
data_folder = "./data/"
data_name = "breastcancer"
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data"

# Creation information
nb_splits = 5
test_size = 0.2

# Prepare folders
data_folder_original = os.path.join(data_folder, data_name, "original")
if not(os.path.exists(data_folder_original)):
    os.makedirs(data_folder_original)

data_folder_processed = os.path.join(data_folder, data_name, "processed")
if not(os.path.exists(data_folder_processed)):
    os.makedirs(data_folder_processed)

data_folder_partitions = os.path.join(data_folder, data_name, "processed", "partitions")
if not(os.path.exists(data_folder_partitions)):
    os.makedirs(data_folder_partitions)

# Download dataset
unprocessed_data_path = os.path.join(data_folder_original,url.split("/")[-1])
if not(os.path.exists(unprocessed_data_path)):
    r = requests.get(url, allow_redirects=True)
    open(unprocessed_data_path, 'wb').write(r.content)

#read data to pandas dataframe

#    #  Attribute                     Domain
#    -- -----------------------------------------
#    1. Sample code number            id number
#    2. Clump Thickness               1 - 10
#    3. Uniformity of Cell Size       1 - 10
#    4. Uniformity of Cell Shape      1 - 10
#    5. Marginal Adhesion             1 - 10
#    6. Single Epithelial Cell Size   1 - 10
#    7. Bare Nuclei                   1 - 10
#    8. Bland Chromatin               1 - 10
#    9. Normal Nucleoli               1 - 10
#   10. Mitoses                       1 - 10
#   11. Class:                        (2 for benign, 4 for malignant)

header = [
    "Sample code number",
    "Clump Thickness",
    "Uniformity of Cell Size",
    "Uniformity of Cell Shape",
    "Marginal Adhesion",
    "Single Epithelial Cell Size",
    "Bare Nuclei",
    "Bland Chromatin",
    "Normal Nucleoli",
    "Mitoses",
    "class",
]

multival = []

#df = pl.read_csv(unprocessed_data_path, has_header=False, new_columns=header)
df = pd.read_csv(unprocessed_data_path, names=header).replace('?', np.NaN).dropna()
df = df.drop(header[0],axis=1)

#class distribution
df["class"].value_counts()

# Make malignant (4) positive class
df.loc[df["class"]==2,"class"]=0
df.loc[df["class"]==4,"class"]=1

#binarize multivalued features
for column in multival:
    values = list(set(df[column]))
    if len(values)>2:
        df = binarize(df, column)
    elif len(values)==2:
        df.loc[df[column]==values[0],column]=-1
        df.loc[df[column]==values[1],column]=1
    else: # drop useless features
        print(column, values)
        df=df.drop(column, axis=1)

#normalize
for column in df.columns.values:
    df[column]=pd.to_numeric(df[column])

normalized_df=(df.astype(float)-df.min())/(df.max()-df.min())*2-1
normalized_df["class"] = df["class"]
df = normalized_df

#move class to back
cols = list(df.columns.values) #Make a list of all of the columns in the df
cols.pop(cols.index('class')) #Remove class from list
df = df[cols+['class']]

# make numpy array
xy = df.values

x = xy[:,:-1].astype(float)
y = xy[:,-1].astype(int)

x_pos = x[y==1]
x_neg = x[y==0]

#Save data and true classes
np.savetxt(os.path.join(data_folder_processed, data_name+"_data.csv"), x)
np.savetxt(os.path.join(data_folder_processed, data_name+"_class.csv"), y,fmt='%d')

# Different dataset partitions (train/test and class prior)
sss = sklearn.model_selection.StratifiedShuffleSplit(n_splits=nb_splits, test_size=test_size, random_state=0)
splits = list(sss.split(x,y))

#save partitions. 0 means not in data, 1 means in train partition, 2 means in test partition
for i, (train,test) in enumerate(splits):
    partition = np.zeros_like(y,dtype=int)
    partition[train]=1

    partition[test]=2
    np.savetxt(os.path.join(data_folder_partitions,data_name+"_train_test_"+str(i)+".csv"), partition, fmt='%d')
