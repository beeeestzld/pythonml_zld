# Create a simple example data frame from a CSV(comma-separated values) file

import pandas as pd
from io import StringIO

csv_data = '''A,B,C,D
1.0,2.0,3.0,4.0
5.0,6.0,,8.0
10.0,11.0,12.0,'''

# If using Python 2.7, need toconvert the string to UNICODE
# csv_data = unicode(csv_data)

df = pd.read_csv(StringIO(csv_data))
print(df)

'''
isnull Method
    return a DataFrame with Boolean values that indicate whether a cell contains a 
    numeric value (False )or if data is missing (True)
'''
print(df.isnull().sum())

# Eliminating samples or features with missing values
## Rows with missing values can be droppeed via 'dropna()' method
print(df.dropna())

## Drop columns that have at least one NaN in any row by setting axis argument to 1
print(df.dropna(axis=1))


# Imputing Missing Values
from sklearn.preprocessing import Imputer
imr = Imputer(missing_values='NaN', strategy='mean', axis=0)
imr = imr.fit(df)
imputed_data = imr.transform(df.values)
print(imputed_data)

# Create a new df with categorical data
df = pd.DataFrame([
    ['green', 'M', 10.1, 'class1'],
    ['red', 'L', 13.5, 'class2'],
    ['blue', 'XL', 15.3, 'class1']])

df.columns = ['color', 'size', 'price', 'classlabel']
print(df)

# Mapping ORDINAL features
size_mapping = {'M':1,
                'L':2,
                'XL':3}

df['size'] = df['size'].map(size_mapping)
print(df)
## transform back
# inv_size_mapping = {v: k for k, v in size_mapping.items()}

# Encoding CLASS LABELS
## Method 1
import numpy as np
class_mapping = {label: idx for idx, label in enumerate(np.unique(df['classlabel']))}
print(class_mapping)

df['classlabel'] = df['classlabel'].map(class_mapping)
print(df)

## Method 2 - LabelEncoder class in scikit-learn
inv_class_mapping = {v: k for k, v in class_mapping.items()}
df['classlabel'] = df['classlabel'].map(inv_class_mapping)

from sklearn.preprocessing import LabelEncoder
class_le = LabelEncoder()
y = class_le.fit_transform(df['classlabel'].values)

print(y)
print(class_le.inverse_transform(y))

# Performing One-Hot Encoding on NOMINAL features
from sklearn.preprocessing import OneHotEncoder
X = df[['color', 'size', 'price']].values
color_le = LabelEncoder()
X[:, 0] = color_le.fit_transform(X[:, 0])
print(X)

ohe = OneHotEncoder(categorical_features=[0])
print (ohe.fit_transform(X).toarray()) # convert sparse matrix to a dense Numpy array
# Or we can use OneHotEncoder(..., sparse=False) to return a regular NumPy array

# More convenient way
print (pd.get_dummies(df[['price', 'color', 'size']]))

