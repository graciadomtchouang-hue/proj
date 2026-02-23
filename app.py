import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Détection de Fraude Bancaire",
    page_icon="🔍",
    layout="wide"
)

@st.cache_resource
def train_model():
    url = "https://raw.githubusercontent.com/graciadomtchouang-hue/proj/main/creditcard_sample.csv"
    df = pd.read_csv(url, nrows=50000)
    cols_order = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']
    X = df[cols_order].copy()
    y = df['Class']

    X['Amount'] = (X['Amount'] - 88.35) / 250.12
    X['Time'] = (X['Time'] - 94813.0) / 47488.0

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    fraud_idx = np.where(y_train == 1)[0]
    legit_idx = np.where(y_train == 0)[0]
    np.random.seed(42)
    oversampled = np.random.choice(fraud_idx, size=len(legit_idx), replace=True)
    idx = np.concatenate([legit_idx, oversampled])
    np.random.shuffle(idx)

    rf = RandomForestClassifier(n_estimators=50, random_state=42,
                                 class_weight='balanced', n_jobs=-1)
    rf.fit(X_train.iloc[idx], y_train.iloc[idx])
    return rf

st.title("🔍 Détection de Fraude à la Carte de Crédit")
st.markdown("Système intelligent de détection basé sur le Machine Learning.")
st.markdown("---")

with st.spinner("⏳ Chargement du modèle... (1-2 minutes)"):
    model = train_model()

st.success("✅ Modèle prêt !")
st.markdown("---")

mode = st.sidebar.selectbox(
    "📌 Navigation",
    ["🏠 Accueil", "📁 Analyse CSV", "✍️ Saisie manuelle"]
)

if mode == "🏠 Accueil":
    st.subheader("📊 Tableau de bord général")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("📦 Dataset", "284 807 transactions")
    col2.metric("🚨 Fraudes", "492 (0.17%)")
    col3.metric("✅ Légitimes", "284 315 (99.83%)")
    col4.metric("🤖 Modèle", "Random Forest")

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📈 Distribution des classes")
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.bar(['Légitimes', 'Fraudes'], [284315, 492],
               color=['#0f3460', '#e94560'], edgecolor='black')
        ax.set_ylabel("Nombre de transactions")
        ax.set_title("Répartition des transactions")
        st.pyplot(fig)
        plt.close()

    with col2:
        st.subheader("🥧 Proportion des classes")
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.pie([284315, 492], labels=['Légitimes', 'Fraudes'],
               autopct='%1.3f%%', colors=['#0f3460', '#e94560'], startangle=90)
        st.pyplot(fig)
        plt.close()

elif mode == "📁 Analyse CSV":
    st.subheader("📁 Analyse de fichier CSV")
    st.markdown("Le fichier doit contenir : **Time, V1 à V28, Amount**")

    uploaded_file = st.file_uploader("Choisissez votre fichier CSV", type=['csv'])

    if uploaded_file is not None:
        df_upload = pd.read_csv(uploaded_file)
        st.dataframe(df_upload.head(5), use_container_width=True)

        required_cols = ['Time', 'Amount'] + [f'V{i}' for i in range(1, 29)]
        missing = [c for c in required_cols if c not in df_upload.columns]

        if missing:
            st.error(f"❌ Colonnes manquantes : {missing}")
        else:
            if st.button("🔎 Lancer l'analyse", type="primary"):
                cols_order = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']
                X_upload = df_upload[cols_order].copy()
                X_upload['Amount'] = (X_upload['Amount'] - 88.35) / 250.12
                X_upload['Time'] = (X_upload['Time'] - 94813.0) / 47488.0

                predictions = model.predict(X_upload)
                probas = model.predict_proba(X_upload)[:, 1]

                df_result = df_upload.copy()
                df_result['Prédiction'] = ['⚠️ Fraude' if p == 1 else '✅ Légitime' for p in predictions]
                df_result['Probabilité Fraude (%)'] = (probas * 100).round(2)

                nb_fraudes = int(sum(predictions))
                nb_total = len(predictions)

                col1, col2, col3 = st.columns(3)
                col1.metric("📦 Total", nb_total)
                col2.metric("⚠️ Fraudes", nb_fraudes)
                col3.metric("✅ Légitimes", nb_total - nb_fraudes)

                st.dataframe(df_result[['Time', 'Amount', 'Prédiction', 'Probabilité Fraude (%)']],
                             use_container_width=True)

                csv_result = df_result.to_csv(index=False).encode('utf-8')
                st.download_button("⬇️ Télécharger les résultats",
                                   data=csv_result,
                                   file_name='resultats_fraude.csv',
                                   mime='text/csv')

elif mode == "✍️ Saisie manuelle":
    st.subheader("✍️ Analyse d'une transaction")

    col1, col2 = st.columns(2)
    with col1:
        amount = st.number_input("💰 Montant (€)", min_value=0.0, value=100.0)
    with col2:
        time = st.number_input("⏱️ Temps (s)", min_value=0.0, value=50000.0)

    st.markdown("**Variables V1 à V28 :**")
    cols = st.columns(4)
    v_values = []
    for i in range(1, 29):
        with cols[(i-1) % 4]:
            v = st.number_input(f"V{i}", value=0.0, step=0.1,
                                min_value=-20.0, max_value=20.0, key=f"v{i}")
            v_values.append(v)

    if st.button("🔎 Analyser", type="primary"):
        amount_scaled = (amount - 88.35) / 250.12
        time_scaled = (time - 94813.0) / 47488.0
        features = np.array([[time_scaled] + v_values + [amount_scaled]])

        prediction = model.predict(features)[0]
        proba = model.predict_proba(features)[0]

        col1, col2 = st.columns(2)
        col1.metric("✅ Probabilité Légitime", f"{proba[0]*100:.2f}%")
        col2.metric("⚠️ Probabilité Fraude", f"{proba[1]*100:.2f}%")

        if prediction == 1:
            st.error("⚠️ TRANSACTION FRAUDULEUSE DÉTECTÉE !")
        else:
            st.success("✅ Transaction légitime")

st.markdown("---")
st.markdown("<center><small>Random Forest | Credit Card Fraud Detection (Kaggle) | Université Saint Jean 2025-2026</small></center>",
            unsafe_allow_html=True)