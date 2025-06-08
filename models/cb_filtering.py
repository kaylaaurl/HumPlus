# %%
import pandas as pd
import pickle

# %%
# Load data
df_customer = pd.read_csv('../data/customer_dataset.csv')
df_sales = pd.read_csv('../data/sales_dataset.csv')

# %%
# Gabungkan data
df_customer.dropna(inplace=True)
df_sales.dropna(inplace=True)
df = pd.merge(df_sales, df_customer, on='customer_id', how='inner')

# %%
# Encode kategorikal
df_encoded = pd.get_dummies(df[['age', 'profession', 'business_type']], drop_first=True)
df_encoded['product'] = df['product']

# %%
# Buat profile-product matrix
profile_product = df_encoded.groupby('product').mean()

# Simpan model
with open('../models/model_cb.pkl', 'wb') as f:
    pickle.dump(profile_product, f)

print("✅ Model content-based berhasil disimpan ke model/model_cb.pkl")

# %%
