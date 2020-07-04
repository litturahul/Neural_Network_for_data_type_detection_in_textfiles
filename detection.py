import pandas as pd
import re
import tensorflow as tf

df=pd.read_csv("C:/Users/Abhinay/Desktop/rahul/detect.csv",header=None)

def find_type(x):
    a=re.match(r"([a-z]|[A-Z])+",str(x))
    if a:
        return "string"
    else:
        b=re.match(r"([0-9])+",str(x))
        if b:
            if(len(b.group())==len(str(x))):
                return "int"
            else:
                return "float"
        else:
            return "other"

df1=pd.DataFrame()

for i in df.columns:
    a=df[i].to_numpy()
    values=[]
    for j in range(0,len(a)):
        values.append(find_type(a[j]))
    df1[i]=values

df1=df1.replace("int",1)
df1=df1.replace("float",2)
df1=df1.replace("string",3)

# Neural Network