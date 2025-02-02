import json # importing the json library
import matplotlib.pyplot as plt #importing the matplotlib library
import seaborn as sns # importing the seaborn library
import numpy as np # importing the numpy library
import pandas as pd # importing the pandas library
import sys as sys # importing the sys library
import boto3 # importing the boto3 library
import os # importing the os library
import sqlalchemy # importing sqlalchemy library
import pymssql # importing mssql library
import pyodbc 

# Create the engine to connect to the PostgresSQL database
engine = sqlalchemy.create_engine('postgresql://postgres:##########@localhost:####/***_####_database')

#Read data from SQL table
sql_data = pd.read_sql_table('customer', engine)
print (sql_data.head()) # use .head() instead of sql_data_head

#Read data from SQL table
sql_data = pd.read_sql_table('employee', engine)
print (sql_data.head()) # use .head() instead of sql_data_head

#Read data from SQL table
sql_data = pd.read_sql_table('city', engine)
print (sql_data.head()) # use .head() instead of sql_data_head

#=================================
sql_data.head
sql_data.info

pandas = pd
#Run sql queries
sql_data = pandas.read_sql(
    "SELECT * FROM customer;",
    engine,
    parse_dates = ['product']
)
import sys 
test_list = [x for x in range(0,100000)]
print(sys.getsizeof(test_list))

##============================================
s = pd.Series([1,3,5, np.nan, 6, 8])
pd.Series

dates = pd.date_range("20130101", periods=12)
dates

#create a daate range for the index
dates = pd.date_range('20230101', periods=6)

# create a date range for the index
df = pd.DataFrame(np.random.randn(6,6), index=dates, columns=list("ABCDEF"))
#display the dataframe
print(df)

Explanation of each column:
==========================================================================================
    Column "A": All rows in this "A" column have the same value 1.0. It's a float64 type by default.
    Column "B": Contains a pd.Timestamp, which is a specific type in pandas for representing datetime values. This column has a single timestamp: 2024-02-02.
    Column "C": A pandas Series is used here, with 4 values all equal to 1. The index is explicitly specified as 0, 1, 2, 3, and the datatype is float32.
    Column "D": A numpy array is used here, with 4 elements, all set to 3. The datatype is specified as int32, a common integer type.
    Column "E": A pd.Categorical is used for storing categorical data. This allows efficient storage of repetitive data, with "sirmill" and "tech" values alternating.
    Column "F": A string column, where every row is assigned the string value "www.datacalculations.com". 
    
    

df2 = pd.DataFrame(
    {
        "A": 1.0,
        "B": pd.Timestamp("20250202"),
        "C": pd.Series(1, index=list(range(4)), dtype="float32"),
        "D": np.array([3] * 4, dtype="int32"),
        "E": pd.Categorical(["sirmill", "tech", "data", "calculations"]),
        "F": "www.datacalculations.com",
    }
)
df2

#DataFrame.agg() and DataFrame.transform() applies a user defined function 
# that reduces or broadcasts its result respectively.
df.agg(lambda x: np.mean(x) * 5.6)
df.agg(lambda x: x * 5.6)

OPERATING WITH ANOTHER SERIES OR DATAFRAME WITH A DIFFERENT INDEX OR COLUMN WILL ALIGN THE RESULT WITH UNION OF THE INDEX OR COLUMN LABLES. 
IN ADDITION, PANDAS AUTHOMATICALLY BROADCASTS ALONG THE SPECIFIED DIMENSION AND WILL FILL UNALIGNED LABELS WITH np.nan

In [65]: df.sub(s, axis="index")

##=================================
df2.A, df2.abs, df2.add, 
df2.add_prefix, df2.add_suffix
df2.align, df2.count, df2.index, df2.all, df2.any, 
df2.apply, df2.applymap, df2.B, df2.bool, df2.boxplot,
df2.C, df2.clip, df2.columns, df2.copy, df2.count, df2.combine, df2.D,

(0    1.0
 1    1.0
 2    1.0
 3    1.0
 Name: C, dtype: float32,
 <bound method NDFrame.clip of      A          B    C  D             E                         F
 0  1.0 2025-02-02  1.0  3       sirmill  www.datacalculations.com
 1  1.0 2025-02-02  1.0  3          tech  www.datacalculations.com
 2  1.0 2025-02-02  1.0  3          data  www.datacalculations.com
 3  1.0 2025-02-02  1.0  3  calculations  www.datacalculations.com>,
 Index(['A', 'B', 'C', 'D', 'E', 'F'], dtype='object'),
 <bound method NDFrame.copy of      A          B    C  D             E                         F
 0  1.0 2025-02-02  1.0  3       sirmill  www.datacalculations.com
 1  1.0 2025-02-02  1.0  3          tech  www.datacalculations.com
 2  1.0 2025-02-02  1.0  3          data  www.datacalculations.com
 3  1.0 2025-02-02  1.0  3  calculations  www.datacalculations.com>,
 <bound method DataFrame.count of      A          B    C  D             E                         F
 0  1.0 2025-02-02  1.0  3       sirmill  www.datacalculations.com
 1  1.0 2025-02-02  1.0  3          tech  www.datacalculations.com
 2  1.0 2025-02-02  1.0  3          data  www.datacalculations.com
 3  1.0 2025-02-02  1.0  3  calculations  www.datacalculations.com>,
 <bound method DataFrame.combine of      A          B    C  D             E                         F
 0  1.0 2025-02-02  1.0  3       sirmill  www.datacalculations.com
 1  1.0 2025-02-02  1.0  3          tech  www.datacalculations.com
 2  1.0 2025-02-02  1.0  3          data  www.datacalculations.com
 3  1.0 2025-02-02  1.0  3  calculations  www.datacalculations.com>,
 0    3
 1    3
 2    3
 3    3
 Name: D, dtype: int32)
 
 
df.to_numpy()

df2.to_numpy()

array([[1.0, Timestamp('2025-02-02 00:00:00'), 1.0, 3, 'sirmill',
        'www.datacalculations.com'],
       [1.0, Timestamp('2025-02-02 00:00:00'), 1.0, 3, 'tech',
        'www.datacalculations.com'],
       [1.0, Timestamp('2025-02-02 00:00:00'), 1.0, 3, 'data',
        'www.datacalculations.com'],
       [1.0, Timestamp('2025-02-02 00:00:00'), 1.0, 3, 'calculations',
        'www.datacalculations.com']], dtype=object)
        
df.head()

sql_data.head()

df.describe()

df.columns.tolist()

sql_data.columns.tolist()

print('Method 1:')
df.isnull().sum()/len(df)*100

print('Method 1:')
sql_data.isnull().sum()/len(df)*100

print('Method 2:')
import missingno as msno
msno.matrix(df)
plt.show()

print('Method 2:')
msno.matrix(sql_data)
plt.show()

#  sql_data is the DataFrame i want to command from
corr = sql_data.select_dtypes('number').corr()  # Compute correlation matrix for numerical columns
display(corr)
# Create a heatmap using seaborn
plt.figure(figsize=(14,10)) # Set figure size
sns.heatmap(corr, annot=True, cmap='viridis', fmt='.2f', linewidths=0.5)

# Add title and labels (optional)
plt.title('Correlation Heatmap')
plt.xlabel('Features')
plt.ylabel('Features')

# Display the plot
plt.show()

import seaborn as sns
sns.set_theme()
plt.figure(figsize=(14,10)) # Set figure size
sql_data = sns.load_dataset("penguins")
sns.histplot(df["flipper_length_mm"])


days = [1,2,3,4,5]
sleeping = [6,7,8,11,7]
eating = [2,3,4,3,2]
working = [7,8,7,2,2,]
playing = [8,5,7,8,13]
slices = [7,2,2,13] #slicing means we are converting our pie chart into 4 parts
activities = ['sleeping','eating','working','playing']
cols = ['c','m','r','b']

plt.figure(figsize=(16, 12))#Set figure size

plt.pie(slices, labels=activities, colors=cols, startangle=90,shadow=True, explode=(0,0.1,0,0), autopct='%1.1f%%')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Intrsting chart')
plt.legend()
plt.show()

t = np.arange(0., 5., 0.2)
# Blue dashes, red squares and green traingles

plt.figure(figsize=(14,10)) # Set figure size
plt.plot(t, t**2, 'b--', label = '^2') # 'rs', 'g^' # Blue dash line.
plt.plot(t, t**2.2, 'rs', label = '^2.2') # red squres (rs)
plt.plot(t, t**2.5, 'g^', label = '^2.5') # green traingle (g^)
plt.grid()
plt.legend() # add legend based on line labels
plt.show()



