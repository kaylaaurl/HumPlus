# %%
import pandas as pd

# %%
# Load dataset
df_customer = pd.read_csv('../data/customer_dataset.csv')
df_sales = pd.read_csv('../data/sales_dataset.csv')

# %%
# Cek missing values dan drop jika perlu
df_customer.dropna(inplace=True)
df_sales.dropna(inplace=True)

# %%
# Cek tipe data
print(df_customer.dtypes)
print(df_sales.dtypes)

# %%
# Merge kedua data
df = pd.merge(df_sales, df_customer, on='customer_id', how='inner')
# %%
# Encode kolom kategori (profesi, jenis_usaha)
df_encoded = pd.get_dummies(df[['age', 'profession', 'business_type']], drop_first=True)

# Gabungkan dengan produk
df_encoded['product'] = df['product']

# %%
# Hitung rata-rata atau frekuensi pembelian produk berdasarkan profil
profile_product = df_encoded.groupby('product').mean()

#%%
def recommend_by_profile(profile_dict):
    profile_series = pd.Series(profile_dict)
    profile_vector = pd.get_dummies(profile_series).reindex(profile_product.columns, axis=1).fillna(0).astype(int)
    scores = profile_product.dot(profile_vector.T).sum(axis=1)
    return scores.sort_values(ascending=False).head(5)
# %%
