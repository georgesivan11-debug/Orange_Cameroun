# Dashboard de scoring d'appétence Data Fixe — Orange Cameroun

Application interactive développée avec **Streamlit** pour le ciblage marketing
des offres Data Fixe, basée sur un modèle **XGBoost** entraîné sur 100 000 clients.

**Performance du modèle (mesurée sur ensemble de test) :**
- AUC-ROC : **0,89**
- PR-AUC : **0,81**
- Gain à 20 % de la base ciblée : **48 %** des souscripteurs captés
- Gain à 50 % de la base ciblée : **90 %** des souscripteurs captés

---

## 📁 Contenu du dossier

```
app/
├── app.py                                    # Application Streamlit principale
├── custom_transformers.py                    # Transformeur personnalisé (IQRCapper)
├── modele_scoring_appetence_datafixe.pkl     # Pipeline sérialisé (prétraitement + modèle)
├── sample_clients_test.csv                   # Fichier de test (2 000 clients)
├── requirements.txt                          # Dépendances Python
└── README.md                                 # Ce fichier
```

**À ajouter avant déploiement :**
- `logo_orange.png` : le logo officiel Orange (que vous placerez vous-même dans ce dossier).
  L'application détecte automatiquement sa présence. Si absent, un placeholder est affiché.

---

## 🚀 Lancer l'application localement (test)

1. Installer les dépendances :
```bash
pip install -r requirements.txt
```

2. Lancer l'application :
```bash
streamlit run orange_cameroun.py
```

3. L'application s'ouvre dans le navigateur à l'adresse `http://localhost:8501`

4. Pour tester : dans l'onglet **1. Import du fichier**, chargez `sample_clients_test.csv`.

---

## ☁️ Déployer sur Streamlit Community Cloud (gratuit)

1. Créer un compte sur [github.com](https://github.com) si ce n'est pas déjà fait.

2. Créer un nouveau dépôt GitHub (par exemple `scoring-datafixe-orange`), puis
   y déposer **tous les fichiers du dossier `app/`** (glisser-déposer sur l'interface web GitHub).

3. Aller sur [streamlit.io/cloud](https://streamlit.io/cloud), se connecter
   avec le compte GitHub.

4. Cliquer sur **New app**, sélectionner le dépôt, la branche `main`, et le fichier `app.py`.

5. Cliquer sur **Deploy**. L'application est en ligne en 2 à 3 minutes,
   avec une URL publique du type `https://votre-app.streamlit.app`.

---

## 📋 Format attendu du fichier CSV à scorer

Le fichier CSV doit contenir au minimum les colonnes suivantes (les colonnes
supplémentaires comme `CLIENT_ID` sont conservées) :

**Numériques :**
`DATA_MOY_3M`, `ARPU_M2`, `DATA_M0`, `NB_RECHARGES_3M`, `NB_TX_OM_3M`,
`MONTANT_RECHARGE_MOY_3M`, `VOIX_M2`, `REPONSE_CAMPAGNE_3M`

**Catégorielles :**
`ZONE` (valeurs attendues : `Urbaine`, `Rurale`),
`TYPE_RESEAU` (valeurs attendues : `3G`, `4G`, `5G`)

Le fichier `sample_clients_test.csv` fourni est un exemple valide.

---

## 🔧 Structure de l'application

L'application est organisée en **5 onglets** :

1. **📥 Import du fichier** — Charger un CSV de clients à scorer
2. **📊 Vue d'ensemble** — Indicateurs clés et distribution des scores
3. **🎯 Simulateur de campagne** — Ajuster le budget et voir le gain attendu en temps réel
4. **🔍 Analyses détaillées** — Importance des variables + exploration client par client
5. **📤 Export ciblage** — Générer et télécharger la liste finale de ciblage

---

*Développé dans le cadre d'un mémoire de fin de cycle — Orange Cameroun*
