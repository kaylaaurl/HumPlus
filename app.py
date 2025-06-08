from flask import Flask, render_template, request, redirect, session
import pickle, json
import pandas as pd

app = Flask(__name__)
app.secret_key = 'humplus-secret'

# Load model CB
with open("models/model_cb.pkl", "rb") as f:
    profile_product = pickle.load(f)
with open("models/model_cb_columns.pkl", "rb") as f:
    expected_columns = pickle.load(f)

# Load model Apriori
with open("models/rules_apriori.pkl", "rb") as f:
    rules = pickle.load(f)

# Simpan user sementara di memory
user_data = {}
user_purchases = {}

@app.route('/')
def home():
    return redirect('/register')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        data = request.form.to_dict()
        email = data['email']
        user_data[email] = data
        session['email'] = email
        return redirect('/dashboard')
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        if email in user_data:
            session['email'] = email
            return redirect('/dashboard')
    return render_template('login.html')

@app.route('/dashboard')
def dashboard():
    email = session.get('email')
    if not email:
        return redirect('/login')
    
    profile = user_data[email]
    profile_vector = pd.get_dummies(pd.Series(profile)).reindex(expected_columns, axis=0).fillna(0).astype(int)
    profile_vector_series = profile_vector.sum(axis=1)
    scores = profile_product.dot(profile_vector_series).sort_values(ascending=False).head(5)
    return render_template('dashboard.html', products=scores.index.tolist())

@app.route('/etalase', methods=['GET', 'POST'])
def etalase():
    email = session.get('email')
    if not email:
        return redirect('/login')
    
    if request.method == 'POST':
        product = request.form['product']
        user_purchases.setdefault(email, []).append(product)

    purchased = user_purchases.get(email, [])
    recommended = []

    for product in purchased:
        matched = rules[rules['antecedents'].apply(lambda x: product in x)]
        for _, row in matched.iterrows():
            for cons in row['consequents']:
                if cons not in purchased:
                    recommended.append(cons)
    
    return render_template('etalase.html', recommended=list(set(recommended)))

if __name__ == '__main__':
    app.run(debug=True)

@app.route('/logout')
def logout():
    session.clear()
    return redirect('/login')

