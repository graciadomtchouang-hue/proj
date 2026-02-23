import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, auc, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(
    page_title="Détection de Fraude Bancaire",
    page_icon="🔍",
    layout="wide"
)

@st.cache_resource
def train_model():
    url = "https://raw.githubusercontent.com/graciadomtchouang-hue/proj/main/creditcard_sample.csv"
    df = pd.read_csv(url)
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

    X_bal = X_train.iloc[idx]
    y_bal = y_train.iloc[idx]

    rf = RandomForestClassifier(n_estimators=50, random_state=42,
                                 class_weight='balanced', n_jobs=-1)
    rf.fit(X_bal, y_bal)
    return rf, X_test, y_test, X

# ============================================================
# CHARGEMENT
# ============================================================
st.title("🔍 Détection de Fraude à la Carte de Crédit")
st.markdown("Système intelligent de détection basé sur le Machine Learning.")
st.markdown("---")

with st.spinner("⏳ Chargement du modèle..."):
    model_rf, X_test_ref, y_test_ref, X_ref = train_model()

st.success("✅ Modèle prêt !")
st.markdown("---")

mode = st.sidebar.selectbox(
    "📌 Navigation",
    ["🏠 Accueil", "📁 Analyse CSV", "✍️ Saisie manuelle"]
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
        st.pyplot(fig)
        plt.close()

    with col2:
        st.subheader("🥧 Proportion des classes")
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.pie([284315, 492], labels=['Légitimes', 'Fraudes'],
               autopct='%1.3f%%', colors=['#0f3460', '#e94560'], startangle=90)
        st.pyplot(fig)
        plt.close()

    st.markdown("---")
    st.subheader("🔑 Importance des variables (Random Forest)")
    fig, ax = plt.subplots(figsize=(12, 4))
    feat_imp = pd.Series(model_rf.feature_importances_,
                          index=X_ref.columns).sort_values(ascending=False)
    feat_imp[:15].plot(kind='bar', ax=ax, color='#0f3460', edgecolor='black')
    ax.set_title("Top 15 variables les plus importantes")
    ax.set_ylabel("Importance")
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# ============================================================
# PAGE ANALYSE CSV
# ============================================================
elif mode == "📁 Analyse CSV":
    st.subheader("📁 Analyse de fichier CSV")
    st.markdown("Le fichier doit contenir : **Time, V1 à V28, Amount**")

    uploaded_file = st.file_uploader("Choisissez votre fichier CSV", type=['csv'])

    if uploaded_file is not None:
        df_upload = pd.read_csv(uploaded_file)
        st.write("**Aperçu des données :**")
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

                    predictions = model_rf.predict(X_upload)
                    probas = model_rf.predict_proba(X_upload)[:, 1]

                df_result = df_upload.copy()
                df_result['Prédiction'] = ['⚠️ Fraude' if p == 1 else '✅ Légitime' for p in predictions]
                df_result['Probabilité Fraude (%)'] = (probas * 100).round(2)

                nb_fraudes = int(sum(predictions))
                nb_total = len(predictions)
                nb_legit = nb_total - nb_fraudes

                # Métriques
                st.markdown("---")
                st.subheader("📊 Résultats de l'analyse")
                col1, col2, col3 = st.columns(3)
                col1.metric("📦 Total analysé", nb_total)
                col2.metric("⚠️ Fraudes détectées", nb_fraudes)
                col3.metric("✅ Légitimes", nb_legit)
                st.markdown("---")

                # Graphique 1 : Distribution des prédictions
                st.subheader("📈 Graphique 1 : Distribution des prédictions")
                col1, col2 = st.columns(2)
                with col1:
                    fig, ax = plt.subplots(figsize=(5, 3))
                    ax.bar(['Légitimes', 'Fraudes'], [nb_legit, nb_fraudes],
                           color=['#0f3460', '#e94560'], edgecolor='black')
                    ax.set_title("Transactions détectées")
                    ax.set_ylabel("Nombre")
                    for i, v in enumerate([nb_legit, nb_fraudes]):
                        ax.text(i, v + 0.1, str(v), ha='center', fontweight='bold')
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()

                with col2:
                    fig, ax = plt.subplots(figsize=(5, 3))
                    ax.pie([nb_legit, nb_fraudes],
                           labels=['Légitimes', 'Fraudes'],
                           autopct='%1.1f%%',
                           colors=['#0f3460', '#e94560'])
                    ax.set_title("Proportion des prédictions")
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()

                st.markdown("---")

                # Graphique 2 : Histogramme des probabilités
                st.subheader("📊 Graphique 2 : Histogramme des probabilités de fraude")
                fig, ax = plt.subplots(figsize=(10, 3))
                ax.hist(probas, bins=30, color='#0f3460', edgecolor='black')
                ax.axvline(0.5, color='#e94560', linestyle='--',
                           linewidth=2, label='Seuil de décision (50%)')
                ax.set_title("Distribution des probabilités de fraude")
                ax.set_xlabel("Probabilité de fraude")
                ax.set_ylabel("Nombre de transactions")
                ax.legend()
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

                st.markdown("---")

                # Graphique 3 : Distribution du montant
                st.subheader("💰 Graphique 3 : Distribution du montant par classe prédite")
                fig, ax = plt.subplots(figsize=(10, 3))
                legitimes_amount = df_upload['Amount'][predictions == 0]
                fraudes_amount = df_upload['Amount'][predictions == 1]
                ax.hist(legitimes_amount, bins=30, alpha=0.6,
                        color='#0f3460', label='Légitime', edgecolor='black')
                if len(fraudes_amount) > 0:
                    ax.hist(fraudes_amount, bins=30, alpha=0.6,
                            color='#e94560', label='Fraude', edgecolor='black')
                ax.set_title("Montant des transactions par classe prédite")
                ax.set_xlabel("Montant (€)")
                ax.set_ylabel("Fréquence")
                ax.legend()
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

                st.markdown("---")

                # Graphique 4 : Corrélations
                st.subheader("🔗 Graphique 4 : Corrélations des variables avec la prédiction")
                fig, ax = plt.subplots(figsize=(12, 4))
                pred_series = pd.Series(predictions, index=df_upload.index)
                corr = df_upload[cols_order].corrwith(pred_series).sort_values()
                colors = ['#e94560' if c < 0 else '#0f3460' for c in corr]
                corr.plot(kind='bar', ax=ax, color=colors, edgecolor='black')
                ax.set_title("Corrélation des variables avec la prédiction de fraude")
                ax.set_ylabel("Corrélation")
                ax.axhline(0, color='black', linewidth=0.8)
                plt.xticks(rotation=45, fontsize=8)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

                st.markdown("---")

                # Matrice de confusion et ROC si Class existe
                if 'Class' in df_upload.columns:
                    st.subheader("🎯 Graphique 5 : Matrice de Confusion")
                    col1, col2 = st.columns(2)
                    with col1:
                        cm = confusion_matrix(df_upload['Class'], predictions)
                        fig, ax = plt.subplots(figsize=(5, 4))
                        sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap='Blues',
                                    xticklabels=['Légitime', 'Fraude'],
                                    yticklabels=['Légitime', 'Fraude'])
                        ax.set_title("Matrice de Confusion")
                        ax.set_ylabel("Réel")
                        ax.set_xlabel("Prédit")
                        st.pyplot(fig)
                        plt.close()

                    with col2:
                        st.subheader("📉 Graphique 6 : Courbe ROC")
                        fig, ax = plt.subplots(figsize=(5, 4))
                        fpr, tpr, _ = roc_curve(df_upload['Class'], probas)
                        roc_auc = auc(fpr, tpr)
                        ax.plot(fpr, tpr, color='#0f3460', linewidth=2,
                                label=f'AUC = {roc_auc:.4f}')
                        ax.plot([0,1], [0,1], 'k--', linewidth=1, label='Aléatoire')
                        ax.set_xlabel("FPR")
                        ax.set_ylabel("TPR")
                        ax.set_title("Courbe ROC")
                        ax.legend()
                        st.pyplot(fig)
                        plt.close()
                        f1 = f1_score(df_upload['Class'], predictions)
                        st.metric("F1-score (fraude)", f"{f1:.4f}")

                    st.markdown("---")

                # Tableau résultats
                st.subheader("📋 Tableau détaillé des transactions")
                st.dataframe(
                    df_result[['Time', 'Amount', 'Prédiction', 'Probabilité Fraude (%)']],
                    use_container_width=True
                )

                csv_result = df_result.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "⬇️ Télécharger les résultats",
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

        prediction = model_rf.predict(features)[0]
        proba = model_rf.predict_proba(features)[0]

        col1, col2 = st.columns(2)
        col1.metric("✅ Probabilité Légitime", f"{proba[0]*100:.2f}%")
        col2.metric("⚠️ Probabilité Fraude", f"{proba[1]*100:.2f}%")

        fig, ax = plt.subplots(figsize=(8, 1.5))
        ax.barh([''], [proba[0]*100], color='#0f3460', label='Légitime')
        ax.barh([''], [proba[1]*100], left=[proba[0]*100],
                color='#e94560', label='Fraude')
        ax.set_xlim(0, 100)
        ax.set_xlabel("Probabilité (%)")
        ax.set_title("Répartition des probabilités")
        ax.legend()
        st.pyplot(fig)
        plt.close()

        if prediction == 1:
            st.error("⚠️ TRANSACTION FRAUDULEUSE DÉTECTÉE !")
        else:
            st.success("✅ Transaction légitime")

st.markdown("---")
st.markdown(
    "<center><small>Random Forest | Credit Card Fraud Detection | Université Saint Jean 2025-2026</small></center>",
    unsafe_allow_html=True
)
