#import pandas as pd

#data = pd.read_pickle('/scratch/kesummer/catELMo/tcr_train.pkl')
#print(data.shape)
#print(data.head())

import pickle

filepath = '/scratch/kesummer/catELMo/tcr_train.pkl'
with open(filepath, 'rb') as f:
    data = pickle.load(f)

# Check what type of object `data` is
print(type(data))

# If `data` is a dictionary, for example, you can inspect the keys:
print("Keys:", data.keys())

# Assuming the data structure is similar to what pandas was reading:
# For example, if data["epi_embeds"] is a list or NumPy array, you can check its length:
print("Length of epi_embeds:", len(data["epi_embeds"]))
print("Length of tcr_embeds:", len(data["tcr_embeds"]))
print("Length of binding:", len(data["binding"]))

