import pandas as pd

df = pd.read_csv('cities1000.txt', sep='\t', header=19)
df.to_csv('cities1000CSV.csv',index=False)

