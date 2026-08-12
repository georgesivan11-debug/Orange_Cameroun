"""
Transformeurs scikit-learn personnalises utilises dans le pipeline de scoring
d'appetence Data Fixe. Ce module doit etre importable a la fois lors de
l'entrainement (pour construire le pipeline) et lors du chargement du modele
serialise (.pkl) par l'application Streamlit, car pickle/joblib ont besoin de
retrouver la definition de la classe au moment du chargement.
"""
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd


class IQRCapper(BaseEstimator, TransformerMixin):
    """Ecretement (winsorisation) base sur l'IQR.

    Les bornes (Q1 - k*IQR, Q3 + k*IQR) sont apprises via fit() sur les
    donnees d'apprentissage uniquement, puis appliquees telles quelles via
    transform(). Cela evite toute fuite de donnees : les memes bornes,
    figees a l'entrainement, sont reutilisees pour n'importe quel nouveau
    client score ensuite (test, ou en production dans l'application).
    """

    def __init__(self, k=3.0):
        self.k = k

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        q1, q3 = X.quantile(0.25), X.quantile(0.75)
        iqr = q3 - q1
        self.lower_ = (q1 - self.k * iqr).values
        self.upper_ = (q3 + self.k * iqr).values
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        return X.clip(lower=self.lower_, upper=self.upper_, axis=1).values
