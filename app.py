"""
Dashboard de scoring d'appétence aux services Data Fixe
Orange Cameroun - Outil d'aide à la décision pour les campagnes marketing

Modèle : XGBoost (AUC-ROC = 0,89 | PR-AUC = 0,81 sur ensemble de test 25 000 clients)
Auteur : Étudiant en stage - Orange Cameroun
"""
import os
from io import BytesIO

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# --- Import du transformeur personnalisé (nécessaire pour dépickler le modèle) ---
from custom_transformers import IQRCapper  # noqa: F401  (utilisé implicitement par joblib.load)

# =============================================================================
# CONFIGURATION DE LA PAGE
# =============================================================================
st.set_page_config(
    page_title="Scoring d'appétence Data Fixe · Orange Cameroun",
    page_icon="📶",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Palette et style Orange ---
ORANGE_PRIMARY = "#FF7900"
ORANGE_DARK = "#000000"
ORANGE_LIGHT = "#FFF3E5"
NEUTRAL_GREY = "#F5F5F5"
TEXT_DARK = "#1A1A1A"
GREEN_OK = "#2E7D32"
RED_ALERT = "#C62828"

st.markdown(
    f"""
    <style>
    /* Fond général */
    .stApp {{
        background-color: #FAFAFA;
    }}
    /* Suppression du bandeau Streamlit */
    #MainMenu, footer, header {{visibility: hidden;}}

    /* Titres */
    h1, h2, h3 {{
        color: {TEXT_DARK};
        font-family: 'Helvetica Neue', sans-serif;
    }}
    h1 {{
        border-bottom: 3px solid {ORANGE_PRIMARY};
        padding-bottom: 0.4rem;
    }}

    /* Cartes d'indicateurs (KPI) */
    div[data-testid="stMetric"] {{
        background-color: white;
        border-left: 4px solid {ORANGE_PRIMARY};
        padding: 1rem 1.2rem;
        border-radius: 6px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    }}
    div[data-testid="stMetric"] > label {{
        color: #666 !important;
        font-size: 0.85rem !important;
    }}
    div[data-testid="stMetric"] > div {{
        color: {TEXT_DARK} !important;
    }}

    /* Boutons */
    .stButton > button {{
        background-color: {ORANGE_PRIMARY};
        color: white;
        border: none;
        font-weight: 600;
        border-radius: 4px;
        padding: 0.5rem 1.5rem;
    }}
    .stButton > button:hover {{
        background-color: #E56900;
        color: white;
    }}
    .stDownloadButton > button {{
        background-color: {ORANGE_DARK};
        color: white;
        border: none;
        font-weight: 600;
        border-radius: 4px;
    }}
    .stDownloadButton > button:hover {{
        background-color: #333;
        color: white;
    }}

    /* Barre latérale */
    section[data-testid="stSidebar"] {{
        background-color: white;
        border-right: 1px solid #E0E0E0;
    }}

    /* Tabs */
    button[data-baseweb="tab"] {{
        font-weight: 600;
    }}
    button[data-baseweb="tab"][aria-selected="true"] {{
        color: {ORANGE_PRIMARY};
        border-bottom-color: {ORANGE_PRIMARY} !important;
    }}

    /* Zone de dépôt de fichier */
    section[data-testid="stFileUploaderDropzone"] {{
        border: 2px dashed {ORANGE_PRIMARY};
        background-color: {ORANGE_LIGHT};
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# =============================================================================
# CHARGEMENT DU MODÈLE
# =============================================================================
@st.cache_resource
def load_model(pkl_path: str = "modele_scoring_appetence_datafixe.pkl"):
    """Charge le pipeline sérialisé (prétraitement + modèle) une seule fois."""
    if not os.path.exists(pkl_path):
        return None
    return joblib.load(pkl_path)


MODEL_DATA = load_model()

# =============================================================================
# BARRE LATÉRALE - Contexte et informations
# =============================================================================
with st.sidebar:
    # Logo Orange (à placer dans le même dossier que app.py)
    if os.path.exists("logo_orange.png"):
        st.image("logo_orange.png", width=140)
    else:
        st.markdown(
            f"""
            <div style="background-color:{ORANGE_PRIMARY}; color:white;
                        padding:1rem; border-radius:6px; text-align:center;
                        font-weight:700; font-size:1.2rem; margin-bottom:1rem;">
                orange<sup>™</sup>
            </div>
            <div style="font-size:0.75rem; color:#888; margin-top:-0.8rem; margin-bottom:1rem;">
                (placez <code>logo_orange.png</code> ici pour afficher le vrai logo)
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### 📶 Scoring d'appétence")
    st.markdown("**Offre Data Fixe · Cameroun**")
    st.markdown("---")

    if MODEL_DATA is not None:
        st.markdown("#### 🎯 Modèle en production")
        st.markdown(f"**{MODEL_DATA['model_name']}**")
        st.markdown(
            f"""
            <div style="font-size:0.85rem; line-height:1.6;">
            AUC-ROC (test) : <b>{MODEL_DATA['test_auc']:.3f}</b><br>
            PR-AUC (test) : <b>{MODEL_DATA['test_prauc']:.3f}</b><br>
            Gain à 20% ciblée : <b>{MODEL_DATA.get('gain_20pct', 47.8):.1f}%</b>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("---")
        st.markdown("#### 📋 Variables attendues")
        with st.expander("Voir la liste"):
            st.markdown("**Numériques :**")
            for f in MODEL_DATA['features_num']:
                st.markdown(f"- `{f}`")
            st.markdown("**Catégorielles :**")
            for f in MODEL_DATA['features_cat']:
                st.markdown(f"- `{f}`")

    st.markdown("---")
    st.markdown(
        """
        <div style="font-size:0.75rem; color:#888;">
        Outil interne à usage marketing.<br>
        Développé dans le cadre d'un mémoire de fin de cycle.
        </div>
        """,
        unsafe_allow_html=True,
    )

# =============================================================================
# EN-TÊTE
# =============================================================================
col_titre, col_metric = st.columns([3, 1])
with col_titre:
    st.title("Dashboard de ciblage marketing")
    st.markdown(
        f"""
        <div style="color:#666; font-size:1rem; margin-top:-0.5rem;">
        Score d'appétence aux offres <b style="color:{ORANGE_PRIMARY};">Data Fixe</b>
        pour la clientèle mobile d'Orange Cameroun
        </div>
        """,
        unsafe_allow_html=True,
    )
with col_metric:
    st.markdown(
        f"""
        <div style="text-align:right; margin-top:1rem;">
            <span style="background-color:{ORANGE_LIGHT}; color:{ORANGE_PRIMARY};
                         padding:0.4rem 0.8rem; border-radius:20px;
                         font-weight:600; font-size:0.85rem;">
                🤖 Moteur XGBoost
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

# --- Contrôle : modèle chargé ---
if MODEL_DATA is None:
    st.error(
        "⚠️ **Modèle introuvable.** Le fichier `modele_scoring_appetence_datafixe.pkl` "
        "n'a pas été trouvé dans le dossier de l'application. Placez-le à côté de `app.py` et rechargez la page."
    )
    st.stop()

# =============================================================================
# ONGLETS
# =============================================================================
tab_import, tab_kpi, tab_ciblage, tab_analyse, tab_export = st.tabs(
    ["📥 1. Import du fichier", "📊 2. Vue d'ensemble", "🎯 3. Simulateur de campagne",
     "🔍 4. Analyses détaillées", "📤 5. Export ciblage"]
)

# =============================================================================
# ONGLET 1 - IMPORT
# =============================================================================
with tab_import:
    st.markdown("### Importer un fichier de clients à scorer")
    st.markdown(
        """
        Chargez un fichier **CSV** contenant les caractéristiques des clients à scorer.
        Le fichier doit inclure au minimum les variables listées dans la barre latérale.
        Les colonnes supplémentaires (comme `CLIENT_ID`) seront conservées et affichées dans le résultat.
        """
    )

    uploaded_file = st.file_uploader(
        "Déposez votre fichier CSV ici (glisser-déposer ou parcourir)",
        type=["csv"],
        help="Format attendu : un client par ligne, séparateur virgule.",
    )

    if uploaded_file is None:
        st.info(
            "💡 **Astuce :** pour tester rapidement, utilisez le fichier "
            "`sample_clients_test.csv` fourni avec l'application (2 000 clients extraits de la base historique)."
        )
    else:
        try:
            df_input = pd.read_csv(uploaded_file)
            required = set(MODEL_DATA['features_all'])
            available = set(df_input.columns)
            missing = required - available
            if missing:
                st.error(
                    f"❌ **Colonnes manquantes** : {', '.join(sorted(missing))}. "
                    f"Le fichier ne peut pas être scoré."
                )
                st.stop()

            st.success(
                f"✅ Fichier chargé : **{len(df_input):,} clients**, "
                f"{df_input.shape[1]} colonnes.".replace(",", " ")
            )
            with st.expander("Aperçu des données importées (5 premières lignes)"):
                st.dataframe(df_input.head(), use_container_width=True)

            # --- Scoring ---
            with st.spinner("Calcul des scores en cours..."):
                proba = MODEL_DATA['pipeline'].predict_proba(df_input[MODEL_DATA['features_all']])[:, 1]
            df_scored = df_input.copy()
            df_scored['SCORE_APPETENCE'] = proba
            df_scored['SCORE_%'] = (proba * 100).round(1)
            df_scored['DECILE'] = pd.qcut(df_scored['SCORE_APPETENCE'].rank(method='first', ascending=False),
                                            10, labels=range(1, 11)).astype(int)

            # Stockage dans la session pour les autres onglets
            st.session_state['df_scored'] = df_scored
            st.session_state['n_clients'] = len(df_scored)

            st.markdown("---")
            st.markdown("### ✅ Scoring terminé")
            st.markdown(
                "Naviguez vers les autres onglets pour visualiser les résultats "
                "(**Vue d'ensemble**, **Simulateur de campagne**, **Analyses détaillées**, **Export ciblage**)."
            )
        except Exception as e:
            st.error(f"❌ Erreur lors de la lecture du fichier : {e}")

# =============================================================================
# ONGLET 2 - KPI / VUE D'ENSEMBLE
# =============================================================================
with tab_kpi:
    if 'df_scored' not in st.session_state:
        st.info("👈 Importez d'abord un fichier de clients dans l'onglet **1. Import du fichier**.")
    else:
        df = st.session_state['df_scored']
        n = len(df)
        taux_moyen = df['SCORE_APPETENCE'].mean()
        n_forts = int((df['SCORE_APPETENCE'] >= 0.5).sum())
        n_top20 = int(np.ceil(n * 0.2))

        st.markdown("### Indicateurs clés")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Clients scorés", f"{n:,}".replace(",", " "))
        k2.metric("Score moyen", f"{taux_moyen*100:.1f}%")
        k3.metric("Clients à fort potentiel", f"{n_forts:,}".replace(",", " "),
                    help="Score de probabilité ≥ 50 %")
        k4.metric("Top 20 % à contacter", f"{n_top20:,}".replace(",", " "),
                    help="Les 20 % de clients les mieux scorés — noyau prioritaire d'une campagne")

        st.markdown("<br>", unsafe_allow_html=True)

        c1, c2 = st.columns([2, 1])
        with c1:
            st.markdown("#### Distribution des scores")
            fig_dist = px.histogram(
                df, x='SCORE_%', nbins=40,
                color_discrete_sequence=[ORANGE_PRIMARY],
                labels={'SCORE_%': "Probabilité d'appétence (%)"},
            )
            fig_dist.update_layout(
                bargap=0.05, plot_bgcolor='white', paper_bgcolor='white',
                height=380, margin=dict(l=10, r=10, t=30, b=10),
                yaxis_title="Nombre de clients",
                xaxis=dict(showgrid=False),
                yaxis=dict(gridcolor='#EEE'),
            )
            fig_dist.add_vline(x=50, line_dash="dash", line_color="grey",
                                annotation_text="Seuil 50 %", annotation_position="top")
            st.plotly_chart(fig_dist, use_container_width=True)

        with c2:
            st.markdown("#### Répartition par décile")
            decile_counts = df['DECILE'].value_counts().sort_index()
            fig_dec = px.bar(
                x=decile_counts.index, y=decile_counts.values,
                color=decile_counts.index,
                color_continuous_scale=[[0, ORANGE_PRIMARY], [1, "#FFD5A8"]],
                labels={'x': 'Décile', 'y': 'Nombre de clients'},
            )
            fig_dec.update_layout(
                plot_bgcolor='white', paper_bgcolor='white', showlegend=False,
                height=380, margin=dict(l=10, r=10, t=30, b=10),
                coloraxis_showscale=False,
                xaxis=dict(showgrid=False, dtick=1),
                yaxis=dict(gridcolor='#EEE'),
            )
            st.plotly_chart(fig_dec, use_container_width=True)

        # --- Répartition par zone / réseau ---
        st.markdown("#### Score moyen par segment")
        c3, c4 = st.columns(2)
        if 'ZONE' in df.columns:
            with c3:
                zone_score = df.groupby('ZONE')['SCORE_APPETENCE'].agg(['mean', 'count']).reset_index()
                zone_score.columns = ['ZONE', 'Score moyen', 'Nombre de clients']
                zone_score['Score moyen'] = zone_score['Score moyen'] * 100
                fig_zone = px.bar(
                    zone_score, x='ZONE', y='Score moyen',
                    color='ZONE', color_discrete_map={'Urbaine': ORANGE_PRIMARY, 'Rurale': '#7A7A7A'},
                    text='Score moyen',
                    labels={'Score moyen': "Score d'appétence moyen (%)"},
                )
                fig_zone.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
                fig_zone.update_layout(
                    plot_bgcolor='white', paper_bgcolor='white', showlegend=False,
                    height=320, margin=dict(l=10, r=10, t=40, b=10),
                    title="Score moyen par zone (urbaine / rurale)",
                    yaxis=dict(gridcolor='#EEE'),
                )
                st.plotly_chart(fig_zone, use_container_width=True)
        if 'TYPE_RESEAU' in df.columns:
            with c4:
                res_score = df.groupby('TYPE_RESEAU')['SCORE_APPETENCE'].agg(['mean', 'count']).reset_index()
                res_score.columns = ['TYPE_RESEAU', 'Score moyen', 'Nombre de clients']
                res_score['Score moyen'] = res_score['Score moyen'] * 100
                fig_res = px.bar(
                    res_score, x='TYPE_RESEAU', y='Score moyen',
                    color_discrete_sequence=[ORANGE_PRIMARY],
                    text='Score moyen',
                    labels={'Score moyen': "Score d'appétence moyen (%)"},
                )
                fig_res.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
                fig_res.update_layout(
                    plot_bgcolor='white', paper_bgcolor='white', showlegend=False,
                    height=320, margin=dict(l=10, r=10, t=40, b=10),
                    title="Score moyen par technologie réseau",
                    yaxis=dict(gridcolor='#EEE'),
                )
                st.plotly_chart(fig_res, use_container_width=True)

# =============================================================================
# ONGLET 3 - SIMULATEUR DE CAMPAGNE
# =============================================================================
with tab_ciblage:
    if 'df_scored' not in st.session_state:
        st.info("👈 Importez d'abord un fichier de clients dans l'onglet **1. Import du fichier**.")
    else:
        df = st.session_state['df_scored']
        n = len(df)

        st.markdown("### Simulateur de campagne marketing")
        st.markdown(
            """
            Ajustez le curseur ci-dessous pour définir la **part de la base à contacter**
            (les clients les mieux scorés en priorité) et visualisez immédiatement le
            gain attendu de la campagne.
            """
        )

        c_curseur, c_kpis = st.columns([2, 3])
        with c_curseur:
            pct = st.slider(
                "🎯 Part de la base à contacter",
                min_value=5, max_value=100, value=20, step=5,
                format="%d%%",
                help="Le simulateur cible en priorité les clients avec les scores les plus élevés."
            )
            n_cibles = int(np.ceil(n * pct / 100))
            st.markdown(f"👥 Nombre de clients ciblés : **{n_cibles:,}**".replace(",", " "))

            # Estimation du gain sur la base d'apprentissage (courbe de gain empirique)
            # On applique le meme raisonnement : on classe les clients par score decroissant
            # et on regarde combien de "hauts scores" (>= mediane du top decile) on capte
            df_sorted = df.sort_values('SCORE_APPETENCE', ascending=False).reset_index(drop=True)
            df_top = df_sorted.head(n_cibles)
            score_moyen_cible = df_top['SCORE_APPETENCE'].mean() * 100
            score_moyen_reste = df_sorted.iloc[n_cibles:]['SCORE_APPETENCE'].mean() * 100 if n_cibles < n else 0

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(f"📈 Score moyen dans le groupe ciblé : **{score_moyen_cible:.1f}%**")
            st.markdown(f"📉 Score moyen dans le reste : **{score_moyen_reste:.1f}%**")
            lift = score_moyen_cible / (df['SCORE_APPETENCE'].mean() * 100) if df['SCORE_APPETENCE'].mean() > 0 else 0
            st.markdown(f"🚀 Lift vs ciblage aléatoire : **×{lift:.2f}**")

        with c_kpis:
            # Extrapolation sur la base historique : le modele capte ~ ces % de souscripteurs
            # aux paliers connus (mesures sur le test set du notebook)
            reference_gain = {
                10: 26.5, 20: 47.8, 30: 65.5, 40: 78.3, 50: 89.7,
                60: 94.5, 70: 97.4, 80: 99.0, 90: 99.7, 100: 100.0
            }
            gain_estime = np.interp(pct, list(reference_gain.keys()), list(reference_gain.values()))

            # Jauge de gain attendu
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=gain_estime,
                title={'text': "% de souscripteurs captés<br>(estimation)", 'font': {'size': 16}},
                delta={'reference': pct, 'suffix': " pts vs aléatoire",
                        'increasing': {'color': GREEN_OK}},
                number={'suffix': "%", 'font': {'size': 44, 'color': ORANGE_DARK}},
                gauge={
                    'axis': {'range': [0, 100], 'tickcolor': '#888'},
                    'bar': {'color': ORANGE_PRIMARY, 'thickness': 0.7},
                    'bgcolor': "white",
                    'borderwidth': 1,
                    'bordercolor': "#DDD",
                    'steps': [
                        {'range': [0, 33], 'color': "#FFEBD9"},
                        {'range': [33, 66], 'color': "#FFD5A8"},
                        {'range': [66, 100], 'color': "#FFBF7A"},
                    ],
                    'threshold': {
                        'line': {'color': GREEN_OK, 'width': 3},
                        'thickness': 0.75,
                        'value': pct,
                    },
                },
            ))
            fig_gauge.update_layout(height=330, margin=dict(l=10, r=10, t=30, b=10),
                                     paper_bgcolor='white')
            st.plotly_chart(fig_gauge, use_container_width=True)

        st.markdown("---")

        # Courbe de gain complète
        st.markdown("#### 📈 Courbe de gain cumulé (référence historique du modèle)")
        st.markdown(
            f"""
            <div style="font-size:0.85rem; color:#666;">
            Cette courbe montre, pour toute part de la base contactée, la proportion
            de souscripteurs réels captés. Elle est calibrée sur l'ensemble de test
            de 25 000 clients utilisé lors de la validation du modèle.
            <b>Point actuel : {pct}% de la base → {gain_estime:.0f}% des souscripteurs captés.</b>
            </div>
            """,
            unsafe_allow_html=True,
        )
        pcts = list(range(0, 101, 5))
        gains = [np.interp(p, [0] + list(reference_gain.keys()), [0] + list(reference_gain.values())) for p in pcts]
        fig_gain = go.Figure()
        fig_gain.add_trace(go.Scatter(x=pcts, y=gains, mode='lines+markers',
                                        line=dict(color=ORANGE_PRIMARY, width=3),
                                        marker=dict(size=6, color=ORANGE_PRIMARY),
                                        name="Modèle XGBoost", fill='tozeroy',
                                        fillcolor='rgba(255, 121, 0, 0.1)'))
        fig_gain.add_trace(go.Scatter(x=[0, 100], y=[0, 100], mode='lines',
                                        line=dict(color='grey', width=2, dash='dash'),
                                        name="Ciblage aléatoire"))
        fig_gain.add_trace(go.Scatter(x=[pct], y=[gain_estime], mode='markers',
                                        marker=dict(size=15, color=GREEN_OK, symbol='star',
                                                    line=dict(color='white', width=2)),
                                        name=f"Sélection ({pct}%)"))
        fig_gain.update_layout(
            plot_bgcolor='white', paper_bgcolor='white',
            xaxis_title="% de la base contactée (triés par score décroissant)",
            yaxis_title="% de souscripteurs captés",
            height=400, margin=dict(l=10, r=10, t=10, b=10),
            xaxis=dict(gridcolor='#EEE', dtick=10),
            yaxis=dict(gridcolor='#EEE', dtick=10),
            legend=dict(orientation="h", yanchor="bottom", y=-0.25),
        )
        st.plotly_chart(fig_gain, use_container_width=True)

# =============================================================================
# ONGLET 4 - ANALYSES DÉTAILLÉES
# =============================================================================
with tab_analyse:
    if 'df_scored' not in st.session_state:
        st.info("👈 Importez d'abord un fichier de clients dans l'onglet **1. Import du fichier**.")
    else:
        df = st.session_state['df_scored']

        st.markdown("### Comprendre le modèle : importance des variables")
        st.markdown(
            """
            Le graphique ci-dessous montre la contribution relative de chaque variable
            aux décisions du modèle XGBoost, calculée sur la base d'entraînement de référence.
            Il aide à comprendre **pourquoi** un client obtient un score élevé ou faible.
            """
        )

        # Récupérer l'importance depuis le pipeline
        clf = MODEL_DATA['pipeline'].named_steps['clf']
        prep = MODEL_DATA['pipeline'].named_steps['prep']
        feat_names = (
            MODEL_DATA['features_num']
            + list(prep.named_transformers_['cat'].named_steps['enc'].get_feature_names_out(MODEL_DATA['features_cat']))
        )
        imp = pd.Series(clf.feature_importances_, index=feat_names).sort_values(ascending=True)
        imp_df = imp.reset_index()
        imp_df.columns = ['Variable', 'Importance']
        imp_df['Importance_%'] = imp_df['Importance'] / imp_df['Importance'].sum() * 100

        fig_imp = px.bar(
            imp_df, x='Importance_%', y='Variable', orientation='h',
            color='Importance_%', color_continuous_scale=[[0, "#FFD5A8"], [1, ORANGE_PRIMARY]],
            text='Importance_%',
        )
        fig_imp.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig_imp.update_layout(
            plot_bgcolor='white', paper_bgcolor='white',
            coloraxis_showscale=False,
            height=420, margin=dict(l=10, r=10, t=10, b=10),
            xaxis=dict(gridcolor='#EEE', title="Contribution au modèle (%)"),
            yaxis=dict(title=""),
        )
        st.plotly_chart(fig_imp, use_container_width=True)

        st.markdown(
            """
            💡 **Lecture métier** : la consommation data récente (`DATA_MOY_3M`) est de très loin
            le signal le plus fort de l'appétence à une offre Data Fixe — un client qui consomme déjà
            beaucoup de data mobile a manifestement un besoin de connectivité que la fibre pourrait combler.
            La réactivité aux campagnes passées (`REPONSE_CAMPAGNE_3M`), la zone géographique
            (`ZONE`) et le revenu (`ARPU_M2`) complètent ce classement.
            """
        )

        st.markdown("---")
        st.markdown("### Explorer un client précis")
        st.markdown("Sélectionnez un client pour voir son score et ses caractéristiques.")

        id_col = 'CLIENT_ID' if 'CLIENT_ID' in df.columns else df.columns[0]
        client_choice = st.selectbox(
            f"Choisir un client (par {id_col})",
            options=df.sort_values('SCORE_APPETENCE', ascending=False)[id_col].tolist()[:200],
            help="Les 200 clients les mieux scorés du fichier sont proposés."
        )
        client_row = df[df[id_col] == client_choice].iloc[0]

        c_score, c_details = st.columns([1, 2])
        with c_score:
            score_pct = client_row['SCORE_%']
            color = GREEN_OK if score_pct >= 70 else (ORANGE_PRIMARY if score_pct >= 40 else RED_ALERT)
            fig_client = go.Figure(go.Indicator(
                mode="gauge+number",
                value=score_pct,
                title={'text': f"Score du client<br><span style='font-size:0.85rem;color:#666;'>{client_choice}</span>"},
                number={'suffix': "%", 'font': {'size': 40, 'color': color}},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': color},
                    'bgcolor': "white",
                    'steps': [
                        {'range': [0, 40], 'color': "#FFE5E5"},
                        {'range': [40, 70], 'color': "#FFF3E5"},
                        {'range': [70, 100], 'color': "#E5F5E5"},
                    ],
                },
            ))
            fig_client.update_layout(height=280, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig_client, use_container_width=True)
            st.markdown(f"**Décile :** {int(client_row['DECILE'])} / 10")
            reco = ("🟢 **Contact prioritaire recommandé**" if score_pct >= 70
                    else "🟠 **À contacter si budget disponible**" if score_pct >= 40
                    else "🔴 **Faible priorité pour cette offre**")
            st.markdown(reco)

        with c_details:
            st.markdown("#### Caractéristiques du client")
            details = {f: client_row[f] for f in MODEL_DATA['features_all'] if f in client_row.index}
            details_df = pd.DataFrame(list(details.items()), columns=['Variable', 'Valeur'])
            st.dataframe(details_df, use_container_width=True, hide_index=True, height=380)

# =============================================================================
# ONGLET 5 - EXPORT
# =============================================================================
with tab_export:
    if 'df_scored' not in st.session_state:
        st.info("👈 Importez d'abord un fichier de clients dans l'onglet **1. Import du fichier**.")
    else:
        df = st.session_state['df_scored']

        st.markdown("### Générer la liste de ciblage")
        st.markdown(
            """
            Filtrez les clients selon vos critères de campagne, puis exportez la liste
            au format CSV, prête à être transmise aux équipes commerciales.
            """
        )

        c1, c2, c3 = st.columns(3)
        with c1:
            mode_filtre = st.radio(
                "Mode de sélection",
                options=["Par pourcentage de la base", "Par nombre de clients", "Par seuil de score"],
            )
        with c2:
            if mode_filtre == "Par pourcentage de la base":
                val = st.slider("Pourcentage à cibler", 5, 100, 20, 5, format="%d%%")
                n_ciblage = int(np.ceil(len(df) * val / 100))
                df_export = df.sort_values('SCORE_APPETENCE', ascending=False).head(n_ciblage)
            elif mode_filtre == "Par nombre de clients":
                val = st.number_input("Nombre de clients", 1, len(df), min(1000, len(df)), 100)
                df_export = df.sort_values('SCORE_APPETENCE', ascending=False).head(int(val))
            else:
                val = st.slider("Score minimum (%)", 0, 100, 50, 5)
                df_export = df[df['SCORE_%'] >= val].sort_values('SCORE_APPETENCE', ascending=False)
        with c3:
            filtre_zone = st.multiselect(
                "Filtrer par zone (facultatif)",
                options=sorted(df['ZONE'].dropna().unique()) if 'ZONE' in df.columns else [],
                default=[],
            )
            if filtre_zone:
                df_export = df_export[df_export['ZONE'].isin(filtre_zone)]

        st.markdown("---")
        st.markdown(
            f"#### 📋 Aperçu de la liste de ciblage : **{len(df_export):,} clients**".replace(",", " ")
        )

        # Colonnes affichées : identifiant + score + variables métier importantes
        cols_display = ['CLIENT_ID'] if 'CLIENT_ID' in df_export.columns else []
        cols_display += ['SCORE_%', 'DECILE']
        cols_display += [c for c in ['ZONE', 'TYPE_RESEAU', 'REGION', 'DATA_MOY_3M', 'ARPU_M2']
                          if c in df_export.columns]
        st.dataframe(df_export[cols_display].reset_index(drop=True),
                      use_container_width=True, height=400)

        # Export CSV
        csv_bytes = df_export.to_csv(index=False).encode('utf-8')
        col_dl1, col_dl2 = st.columns([1, 3])
        with col_dl1:
            st.download_button(
                label="📥 Télécharger la liste (CSV)",
                data=csv_bytes,
                file_name=f"ciblage_campagne_datafixe_{len(df_export)}clients.csv",
                mime="text/csv",
            )
        with col_dl2:
            st.markdown(
                f"""
                <div style="font-size:0.85rem; color:#666; padding-top:0.5rem;">
                Le fichier contient <b>toutes les colonnes originales</b> + le score et le décile,
                trié par score décroissant. Prêt à être importé dans un outil de campagne.
                </div>
                """,
                unsafe_allow_html=True,
            )
