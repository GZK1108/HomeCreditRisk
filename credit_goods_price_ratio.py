import pandas as pd

df = pd.read_csv(r'C:\Users\kk\University_task\creditdata.csv', header=0)
df['credit_goods_price_ratio']=df.apply(lambda x: x['AMT_CREDIT'] / x['AMT_GOODS_PRICE'], axis=1)
df.to_csv(r'C:\Users\kk\University_task\agprcreditdata.csv', index=False)
