import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Configuration de la page
st.set_page_config(
    page_title="Détection de Fraude Bancaire",
    page_icon="🔍",
    layout="wide"
)

# Style CSS personnalisé
st.markdown("""
<style>
    .main { background-color: #f5f7fa; }
    .stMetric {
        background-color: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .title-box {
        background: linear-gradient(135deg, #1a1a2e, #16213e, #0f3460);
        padding: 30px;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 20px;
    }
    .title-text {
        color: white;
        font-size: 2.2em;
        font-weight: bold;
        margin: 0;
    }
    .subtitle-text {
        color: #a0c4ff;
        font-size: 1em;
        margin-top: 8px;
    }
    .fraud-box {
        background-color: #ffe0e0;
        border-left: 5px solid #e94560;
        padding: 15px;
        border-radius: 8px;
        font-size: 1.2em;
        font-weight: bold;
        color: #c0392b;
    }
    .legit-box {
        background-color: #e0f7e9;
        border-left: 5px solid #27ae60;
        padding: 15px;
        border-radius: 8px;
        font-size: 1.2em;
        font-weight: bold;
        color: #1e8449;
    }
</style>
""", unsafe_allow_html=True)

# Entraînement du modèle
@st.cache_resource
def train_model():
    df = pd.read_csv('creditcard.csv', nrows=50000)
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

# ============================================================
# EN-TÊTE
# ============================================================
st.markdown("""
<div class="title-box">
    <p class="title-text">🔍 Détection de Fraude à la Carte de Crédit</p>
    <p class="subtitle-text">Système intelligent de détection basé sur le Machine Learning (Random Forest)</p>
</div>
""", unsafe_allow_html=True)

# Chargement du modèle
with st.spinner("⏳ Initialisation du modèle en cours..."):
    model = train_model()

st.success("✅ Modèle opérationnel !")
st.markdown("---")

# ============================================================
# NAVIGATION
# ============================================================
mode = st.sidebar.selectbox(
    "📌 Navigation",
    ["🏠 Accueil", "📁 Analyse CSV", "✍️ Saisie manuelle"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### ℹ️ À propos")
st.sidebar.info(
    "Cette application utilise un modèle **Random Forest** "
    "entraîné sur le dataset Credit Card Fraud Detection (Kaggle) "
    "pour détecter les transactions frauduleuses."
)

# ============================================================
# PAGE ACCUEIL
# ============================================================
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
        for i, v in enumerate([284315, 492]):
            ax.text(i, v + 1000, str(v), ha='center', fontweight='bold')
        st.pyplot(fig)
        plt.close()

    with col2:
        st.subheader("🥧 Proportion des classes")
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.pie([284315, 492], labels=['Légitimes', 'Fraudes'],
               autopct='%1.3f%%', colors=['#0f3460', '#e94560'],
               startangle=90)
        ax.set_title("Proportion fraudes vs légitimes")
        st.pyplot(fig)
        plt.close()

    st.markdown("---")
    st.subheader("🚀 Comment utiliser l'application ?")
    col1, col2 = st.columns(2)
    with col1:
        st.info("📁 **Analyse CSV** : Uploadez un fichier CSV contenant vos transactions pour les analyser en masse.")
    with col2:
        st.info("✍️ **Saisie manuelle** : Entrez les paramètres d'une transaction manuellement pour obtenir une prédiction.")

# ============================================================
# PAGE ANALYSE CSV
# ============================================================
elif mode == "📁 Analyse CSV":
    st.subheader("📁 Analyse de fichier CSV")
    st.markdown("Uploadez un fichier CSV avec les colonnes : **Time, V1 à V28, Amount**")

    uploaded_file = st.file_uploader("Choisissez votre fichier CSV", type=['csv'])

    if uploaded_file is not None:
        df_upload = pd.read_csv(uploaded_file)

        st.write("**Aperçu des données uploadées :**")
        st.dataframe(df_upload.head(5), use_container_width=True)

        required_cols = ['Time', 'Amount'] + [f'V{i}' for i in range(1, 29)]
        missing = [c for c in required_cols if c not in df_upload.columns]

        if missing:
            st.error(f"❌ Colonnes manquantes : {missing}")
        else:
            if st.button("🔎 Lancer l'analyse", type="primary"):
                with st.spinner("Analyse en cours..."):
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
                nb_legit = nb_total - nb_fraudes

                st.markdown("---")
                st.subheader("📊 Résultats de l'analyse")

                col1, col2, col3 = st.columns(3)
                col1.metric("📦 Total analysé", nb_total)
                col2.metric("⚠️ Fraudes détectées", nb_fraudes,
                            delta=f"{nb_fraudes/nb_total*100:.2f}%",
                            delta_color="inverse")
                col3.metric("✅ Légitimes", nb_legit)

                st.markdown("---")
                col1, col2 = st.columns(2)

                with col1:
                    fig, ax = plt.subplots(figsize=(5, 3))
                    ax.bar(['Légitimes', 'Fraudes'], [nb_legit, nb_fraudes],
                           color=['#0f3460', '#e94560'], edgecolor='black')
                    ax.set_title("Résultats de la détection")
                    ax.set_ylabel("Nombre")
                    st.pyplot(fig)
                    plt.close()

                with col2:
                    fig, ax = plt.subplots(figsize=(5, 3))
                    ax.hist(probas, bins=30, color='#0f3460', edgecolor='black')
                    ax.axvline(0.5, color='#e94560', linestyle='--', label='Seuil 50%')
                    ax.set_title("Distribution des probabilités de fraude")
                    ax.set_xlabel("Probabilité de fraude")
                    ax.set_ylabel("Nombre de transactions")
                    ax.legend()
                    st.pyplot(fig)
                    plt.close()

                st.markdown("---")
                st.subheader("📋 Détail des transactions")
                st.dataframe(
                    df_result[['Time', 'Amount', 'Prédiction', 'Probabilité Fraude (%)']],
                    use_container_width=True
                )

                csv_result = df_result.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="⬇️ Télécharger les résultats CSV",
                    data=csv_result,
                    file_name='resultats_fraude.csv',
                    mime='text/csv'
                )

# ============================================================
# PAGE SAISIE MANUELLE
# ============================================================
elif mode == "✍️ Saisie manuelle":
    st.subheader("✍️ Analyse d'une transaction")

    col1, col2 = st.columns(2)
    with col1:
        amount = st.number_input("💰 Montant de la transaction (€)",
                                  min_value=0.0, value=100.0, step=1.0)
    with col2:
        time = st.number_input("⏱️ Temps depuis 1ère transaction (s)",
                                min_value=0.0, value=50000.0)

    st.markdown("**Variables V1 à V28 (composantes PCA) :**")
    cols = st.columns(4)
    v_values = []
    for i in range(1, 29):
        with cols[(i-1) % 4]:
            v = st.number_input(f"V{i}", value=0.0, step=0.1,
                                min_value=-20.0, max_value=20.0,
                                key=f"v{i}")
            v_values.append(v)

    st.markdown("---")
    if st.button("🔎 Analyser la transaction", type="primary"):
        amount_scaled = (amount - 88.35) / 250.12
        time_scaled = (time - 94813.0) / 47488.0
        features = np.array([[time_scaled] + v_values + [amount_scaled]])

        prediction = model.predict(features)[0]
        proba = model.predict_proba(features)[0]

        st.markdown("---")
        st.subheader("📊 Résultat de l'analyse")

        col1, col2, col3 = st.columns(3)
        col1.metric("✅ Probabilité Légitime", f"{proba[0]*100:.2f}%")
        col2.metric("⚠️ Probabilité Fraude", f"{proba[1]*100:.2f}%")
        col3.metric("🎯 Décision", "FRAUDE" if prediction == 1 else "LÉGITIME")

        if prediction == 1:
            st.markdown('<div class="fraud-box">⚠️ TRANSACTION FRAUDULEUSE DÉTECTÉE !</div>',
                       unsafe_allow_html=True)
        else:
            st.markdown('<div class="legit-box">✅ Transaction légitime - Aucune anomalie détectée</div>',
                       unsafe_allow_html=True)

        # Jauge de probabilité
        st.markdown("---")
        fig, ax = plt.subplots(figsize=(8, 1.5))
        ax.barh([''], [proba[0]*100], color='#0f3460', label='Légitime')
        ax.barh([''], [proba[1]*100], left=[proba[0]*100],
                color='#e94560', label='Fraude')
        ax.set_xlim(0, 100)
        ax.set_xlabel("Probabilité (%)")
        ax.set_title("Répartition des probabilités")
        ax.legend(loc='upper right')
        st.pyplot(fig)
        plt.close()

st.markdown("---")
st.markdown(
    "<center><small>Modèle : Random Forest | "
    "Dataset : Credit Card Fraud Detection (Kaggle) | "
    "Université Saint Jean 2025-2026</small></center>",
    unsafe_allow_html=True
)