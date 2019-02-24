import pandas as pd
import matplotlib.pyplot as plt

# =====read data===============================================
bdata = open(r'D:\OneDrive - UW-Madison\Module 2\Data_Module2\business_train.json').read()
bdata = pd.read_json(r'D:\OneDrive - UW-Madison\Module 2\Data_Module2\business_train.json', orient='records', lines=True)

# rdata = pd.read_json(r'D:\OneDrive - UW-Madison\Module 2\Data_Module2\review_train.json',lines=True, orient='records')