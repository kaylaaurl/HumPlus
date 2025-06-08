#%%
# apriori_model.py
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import os
import pickle

#%%
# Load sales dataset
df_sales = pd.read_csv('../data/sales_dataset.csv')

#%%
# Group produk per customer → transaksi
transactions = df_sales.groupby('customer_id')['product'].apply(list).tolist()

#%%
# Encode transaksi
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df_encoded = pd.DataFrame(te_ary, columns=te.columns_)

#%%
# Apriori algorithm
frequent_itemsets = apriori(df_encoded, min_support=0.1, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.3)

#%%
print("Jumlah rules ditemukan:", len(rules))
print(rules.head())

#%%
# Simpan hasil
os.makedirs("output", exist_ok=True)
rules.to_csv("output/association_rules.csv", index=False)
rules.to_pickle("output/rules_apriori.pkl")

#%%
print("✅ Apriori selesai! Jumlah rules ditemukan:", len(rules))
print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head())

# %%
