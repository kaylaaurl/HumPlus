# train_model.py
# %%
import pandas as pd
import pickle
import os

# %%
# Load dataset
df_customer = pd.read_csv('data/customer_dataset.csv')
df_sales = pd.read_csv('data/sales_dataset.csv')

# %%
# Preprocessing
df_customer.dropna(inplace=True)
df_sales.dropna(inplace=True)

df = pd.merge(df_sales, df_customer, on='customer_id', how='inner')

# %%
# Encoding
df_encoded = pd.get_dummies(df[['age', 'profession', 'business_type']], drop_first=True)
df_encoded['product'] = df['product']

# %%
# Hitung rata-rata profil per produk
profile_product = df_encoded.groupby('product').mean()

# %%
# Simpan model & kolom
os.makedirs("models", exist_ok=True)
with open("models/model_cb.pkl", "wb") as f:
    pickle.dump(profile_product, f)

with open("models/model_cb_columns.pkl", "wb") as f:
    pickle.dump(profile_product.columns.tolist(), f)

print("Model berhasil disimpan.")

# %%
# Fungsi rekomendasi untuk testing (opsional)
def recommend_by_profile(profile_dict):
    import pickle

    with open("models/model_cb.pkl", "rb") as f:
        profile_product = pickle.load(f)
    with open("models/model_cb_columns.pkl", "rb") as f:
        expected_columns = pickle.load(f)

    profile_series = pd.Series(profile_dict)

    profile_vector = pd.get_dummies(profile_series).reindex(expected_columns, axis=0).fillna(0).astype(int)

    if profile_vector.sum().sum() == 0:
        print("⚠️ Profil tidak cocok dengan model. Cek input.")
        return pd.Series(dtype=float)

    # Ubah profile_vector jadi Series (karena cuma 1 baris, sum axis=1)
    profile_vector_series = profile_vector.sum(axis=1)

    scores = profile_product.dot(profile_vector_series)

    return scores.sort_values(ascending=False).head(5)


# %%
# Contoh penggunaan testing
sample_profile = {
    "age": 35,
    "profession": "Freelancer",
    "business_type": "Education"
}

print("Top 5 rekomendasi produk:")
print(recommend_by_profile(sample_profile))

# %%
