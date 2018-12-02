#t-test
import pandas as pd
from scipy.stats import ttest_ind
import numpy as np

#import data - 
df = pd.read_csv('T-Test_STD_Data.csv')

#Seperate groups
adam = np.array(df['adam'])
sgd = np.array(df['SGD'])

#perform T-Test
print(ttest_ind(adam, sgd))


