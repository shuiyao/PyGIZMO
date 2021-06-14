import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

pd.set_option('display.width', 80)
pd.set_option('display.max_columns', 5)
df = pd.read_csv("wacc.1185.csv")

# sns.lineplot(data=df, x='snapnum', y='Mgain', hue='birthTag')
grp = df.groupby(['PId','birthTag'])
x = grp['Mgain'].cumsum(skipna=True)
df2 = pd.concat([df[['PId','snapnum','birthTag']], x], axis=1)
sns.lineplot(data=df2, x='snapnum', y='Mgain', hue='birthTag')

print('DONE')
