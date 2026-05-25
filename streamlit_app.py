import streamlit as st
import pandas as pd
from langchain_ollama.chat_models import ChatOllama
from langchain.messages import AIMessage, SystemMessage, HumanMessage
import os
import requests
from pathlib import Path
from io import StringIO
import traceback
from typing import List, Dict, Any

# Configuration Streamlit
st.set_page_config(page_title="Imora - Streamlit", page_icon="🧠", layout="wide")

# 🔗 Tunnel ngrok pour Ollama
OLLAMA_TUNNEL_URL = "https://hilma-unvaluable-cade.ngrok-free.dev"

# Initialiser session state
if "dataset" not in st.session_state:
    st.session_state.dataset = None
if "dataset_name" not in st.session_state:
    st.session_state.dataset_name = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "chat_metadata" not in st.session_state:
    st.session_state.chat_metadata = []  # Stocke les métadonnées de chaque réponse
if "llm_initialized" not in st.session_state:
    st.session_state.llm_initialized = False
if "llm" not in st.session_state:
    st.session_state.llm = None

# SYSTEM PROMPT Imora
SYSTEM_PROMPT = """
# 🧠 IMORA — Agent d'Intelligence Artificielle

Tu es **Imora**, un spécialiste en analyse de données et intelligence artificielle.

## 🎯 Compétences
- Analyse exploratoire de données
- Détection de valeurs manquantes et anomalies
- Nettoyage et préparation de données
- Conseils sur le machine learning

## ⚠️ Principe Fondamental
**Aucune modification sans demande explicite de l'utilisateur.**

## 📊 Fonctionnement
1. Analyse d'abord
2. Justifie tes recommandations
3. Propose une action si approprié
4. Attends la confirmation avant d'agir

## 🔧 Outils disponibles
- Analyse de types de colonnes
- Diagnostic des valeurs manquantes
- Détection d'outliers
- Suggestions de nettoyage

Tu es rigoureux, justificatif et prudent. Toujours traçable.
"""

# Registres globaux pour outils
models_registry: Dict[str, Any] = {}
_snapshots: Dict[str, pd.DataFrame] = {}

def get_dataset_description(df: pd.DataFrame) -> str:
    """Fournit une description complète du dataset"""
    try:
        output = StringIO()
        df.info(buf=output)
        info_str = output.getvalue()
        
        description = f"""
# 📊 Description du Dataset

## Dimensions
- Lignes: {df.shape[0]}
- Colonnes: {df.shape[1]}

## Types de données
{df.dtypes.to_string()}

## Informations détaillées
{info_str}

## Statistiques descriptives
{df.describe().to_string()}
"""
        return description
    except Exception as e:
        return f"Erreur lors de l'analyse: {str(e)}"

def detect_missing_values(df: pd.DataFrame) -> str:
    """Détecte et rapporte les valeurs manquantes"""
    missing = df.isna().sum()
    missing_percent = (df.isna().sum() / len(df)) * 100
    
    report = "## 🔍 Diagnostic Valeurs Manquantes\n\n"
    report += f"Total valeurs manquantes: {missing.sum()} / {df.size}\n\n"
    
    if missing.sum() == 0:
        report += "✅ Aucune valeur manquante détectée!"
    else:
        report += "### Colonnes affectées:\n"
        for col in missing[missing > 0].sort_values(ascending=False).index:
            report += f"- **{col}**: {missing[col]} ({missing_percent[col]:.2f}%)\n"
    
    return report

def detect_outliers_simple(df: pd.DataFrame) -> str:
    """Détecte les outliers dans les colonnes numériques"""
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    
    report = "## 📈 Détection d'Outliers\n\n"
    
    if len(numeric_cols) == 0:
        return report + "Aucune colonne numérique trouvée."
    
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        if len(outliers) > 0:
            report += f"- **{col}**: {len(outliers)} outliers détectés\n"
    
    return report

# ---------- Outils complémentaires du notebook (adaptés) ----------

def open_df(file_path: str, is_csv: bool, is_excel: bool, csv_sep: str = ",") -> pd.DataFrame | None:
    """Ouvre un DataFrame à partir d'un fichier local"""
    try:
        if is_csv:
            return pd.read_csv(file_path, sep=csv_sep)
        if is_excel:
            return pd.read_excel(file_path)
    except Exception as e:
        return None

def download_from_url(url: str) -> str:
    """Télécharge un fichier depuis une URL et retourne le chemin local"""
    filename = "/tmp/" + url.split("/")[-1]
    resp = requests.get(url)
    resp.raise_for_status()
    with open(filename, "wb") as f:
        f.write(resp.content)
    return filename

def get_file_type(file_path: str) -> str:
    """Détecte le type d'un fichier via son mime"""
    try:
        import magic
        mime = magic.from_file(file_path, mime=True)
        return "csv" if "csv" in mime else "excel" if "sheet" in mime else "unknown"
    except Exception:
        return "unknown"

def read_non_binary_file(file_path: str) -> str:
    """Retourne les premières lignes d'un fichier texte"""
    import os
    if os.path.exists(file_path) and os.path.isfile(file_path):
        with open(file_path, "r", errors="ignore") as f:
            return "\n".join(f.readlines()[:10])
    return "Fichier introuvable ou non lisible"

def normalize_date(columns_list: List[str]) -> str:
    df = st.session_state.dataset
    try:
        formats = ["%d/%m/%Y","%d-%m-%Y","%Y-%m-%d","%Y/%m/%d"]
        for col in columns_list:
            for fmt in formats:
                dt = pd.to_datetime(df[col], format=fmt, errors="coerce")
                if not dt.isna().all():
                    df[col] = dt
                    break
            if df[col].isna().all():
                return f"Date invalide dans la colonne: {col}"
    except Exception as e:
        return f"Erreur lors de la normalisation: {e}"
    return f"Les colonnes {', '.join(columns_list)} ont été normalisées au format ISO 8601."

def missing_data_diagnostic() -> str:
    df = st.session_state.dataset
    if df is None or df.empty:
        return "Le DataFrame est vide"
    missing_cols = [
        (col, float((df[col].isnull().sum() / len(df)) * 100))
        for col in df.columns[df.isnull().any()].to_list()
    ]
    return "\n".join([f"{c}: {p}% ({risk_type(p)})" for c, p in missing_cols])

def query_database(query: str) -> str:
    import duckdb
    df = st.session_state.dataset
    try:
        duckdb.register("df", df)
        result_df = duckdb.query(query).to_df()
        return result_df.to_string(index=False)
    except Exception as e:
        return f"Erreur lors de l'exécution de la requête : {e}"

def search_web(query: str) -> str:
    try:
        from dotenv import load_dotenv
        load_dotenv()
        url = "https://google.serper.dev/search"
        payload = {"q": query}
        headers = {"X-API-KEY": os.getenv("SERPER_API_KEY"), "Content-Type": "application/json"}
        response = requests.post(url, headers=headers, json=payload, timeout=10)
        return response.text
    except Exception as e:
        return f"Erreur recherche web: {e}"

def delete_rows_with_missing_values(column_name: str) -> str:
    df = st.session_state.dataset
    if df is None or df.empty:
        return "Le DataFrame est vide"
    if column_name not in df.columns:
        return f"La colonne '{column_name}' n'existe pas"
    missing_count = df[column_name].isnull().sum()
    df.dropna(subset=[column_name], inplace=True)
    return f"Suppression de {missing_count} lignes contenant des valeurs manquantes dans la colonne '{column_name}'."

def impute_missing_values(column_name: str, method: str) -> str:
    df = st.session_state.dataset
    if df is None or df.empty:
        return "Le DataFrame est vide"
    if column_name not in df.columns:
        return f"La colonne '{column_name}' n'existe pas"
    missing_count = df[column_name].isnull().sum()
    if method == "mean":
        imputed_value = df[column_name].mean()
    elif method == "median":
        imputed_value = df[column_name].median()
    elif method == "mode":
        imputed_value = df[column_name].mode()[0]
    else:
        return f"Méthode d'imputation '{method}' non reconnue."
    df[column_name].fillna(imputed_value, inplace=True)
    return f"Imputation de {missing_count} valeurs manquantes dans la colonne '{column_name}' en utilisant la méthode '{method}' avec la valeur imputée : {imputed_value}."

def replace_missing_values(column_name: str, value: Any) -> str:
    df = st.session_state.dataset
    if df is None or df.empty:
        return "Le DataFrame est vide"
    if column_name not in df.columns:
        return f"La colonne '{column_name}' n'existe pas"
    missing_count = df[column_name].isnull().sum()
    df[column_name].fillna(value, inplace=True)
    return f"Imputation de {missing_count} valeurs manquantes dans la colonne '{column_name}' en utilisant la valeur personnalisée : {value}."

def normalize_int(columns_list: List[str]) -> str:
    df = st.session_state.dataset
    try:
        for col in columns_list:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
        return f"Les colonnes {', '.join(columns_list)} ont été normalisées en entiers."
    except Exception as e:
        return f"Erreur lors de la normalisation: {e}"

def normalize_float(columns_list: List[str]) -> str:
    df = st.session_state.dataset
    try:
        for col in columns_list:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(float)
        return f"Les colonnes {', '.join(columns_list)} ont été normalisées en float."
    except Exception as e:
        return f"Erreur lors de la normalisation: {e}"

def analyze_numeric_columns(columns_list: list[str]) -> str:
    df = st.session_state.dataset
    dirty_cols = {}
    for col in columns_list:
        mask = ~df[col].astype(str).str.match(r"^\d+$", na=False)
        invalid_rows = df[mask]
        dirty_cols[col] = invalid_rows[col].to_dict()
    return f"Rapport d'analyse des colonnes numériques corrompues: \n{dirty_cols}"

def replace_specific_value(column: str, value_to_replace: Any, new_value: Any) -> str:
    df = st.session_state.dataset
    if df is None or df.empty:
        return "Le DataFrame est vide"
    if column not in df.columns:
        return f"La colonne '{column}' n'existe pas"
    replace_count = (df[column] == value_to_replace).sum()
    df[column] = df[column].replace(value_to_replace, new_value)
    return f"Remplacement de {replace_count} occurrences de la valeur '{value_to_replace}' par '{new_value}' dans la colonne '{column}'."

def remove_specific_characters(characters: str, columns: List[str]) -> str:
    df = st.session_state.dataset
    if df is None or df.empty:
        return "Le DataFrame est vide"
    missing_cols = [col for col in columns if col not in df.columns]
    if missing_cols:
        return f"Colonnes manquantes: {', '.join(missing_cols)}"
    replace_count = 0
    for col in columns:
        count = df[col].astype(str).str.count(characters).sum()
        replace_count += count
        df[col] = df[col].astype(str).str.replace(characters, "", regex=False)
    return f"Retrait de {replace_count} occurrences des caractères '{characters}' dans les colonnes {', '.join(columns)}."

def rename_columns(columns_mapping: Dict[str, str]) -> str:
    df = st.session_state.dataset
    if df is None or df.empty:
        return "Le DataFrame est vide"
    missing_cols = [col for col in columns_mapping.keys() if col not in df.columns]
    if missing_cols:
        return f"Les colonnes suivantes n'existent pas: {', '.join(missing_cols)}"
    df.rename(columns=columns_mapping, inplace=True)
    renamed_cols = ", ".join([f"'{old}' -> '{new}'" for old, new in columns_mapping.items()])
    return f"Renommage des colonnes : {renamed_cols}."

def knn_imputation(column_name: str, n_neighbors: int = 5) -> str:
    df = st.session_state.dataset
    if df is None or df.empty:
        return "Le DataFrame est vide"
    if column_name not in df.columns:
        return f"La colonne '{column_name}' n'existe pas"
    from sklearn.impute import KNNImputer
    imputer = KNNImputer(n_neighbors=n_neighbors)
    imputed_values = imputer.fit_transform(df[[column_name]])
    missing_count = df[column_name].isnull().sum()
    df[column_name] = imputed_values
    return f"Imputation de {missing_count} valeurs manquantes dans la colonne '{column_name}' en utilisant KNN ({n_neighbors} voisins)."

def save_df(file_path: str, is_csv: bool = False, is_excel: bool = False, csv_sep: str = ",") -> str:
    df = st.session_state.dataset
    if df is None or df.empty:
        return "Le DataFrame est vide"
    try:
        if is_csv:
            df.to_csv(file_path, sep=csv_sep, index=False)
            return f"DataFrame enregistré au format CSV : {file_path}"
        if is_excel:
            df.to_excel(file_path, index=False)
            return f"DataFrame enregistré au format Excel : {file_path}"
        return "Type de fichier non supporté"
    except Exception as e:
        return f"Erreur sauvegarde: {e}"

def create_snapshot(name: str) -> str:
    df = st.session_state.dataset
    if df is None:
        return "Aucun DataFrame chargé."
    _snapshots[name] = df.copy(deep=True)
    return f"Snapshot '{name}' créé. Total: {len(_snapshots)}"

def restore_snapshot(name: str) -> str:
    if name not in _snapshots:
        return f"Snapshot '{name}' introuvable."
    st.session_state.dataset = _snapshots[name].copy(deep=True)
    return f"Snapshot '{name}' restauré."

def list_snapshots() -> list:
    return list(_snapshots.keys())

def delete_snapshot(name: str) -> str:
    if name not in _snapshots:
        return f"Snapshot '{name}' introuvable."
    del _snapshots[name]
    return f"Snapshot '{name}' supprimé."

def encode_categorical_columns(columns_list: List[str]) -> str:
    df = st.session_state.dataset
    if df is None or df.empty:
        return "Le DataFrame est vide"
    missing_cols = [col for col in columns_list if col not in df.columns]
    if missing_cols:
        return f"Les colonnes suivantes n'existent pas: {', '.join(missing_cols)}"
    df_encoded = pd.get_dummies(df[columns_list], prefix=columns_list)
    df.drop(columns=columns_list, inplace=True)
    df[df_encoded.columns] = df_encoded
    return f"Les colonnes {', '.join(columns_list)} ont été encodées en one-hot."

def normalize_categorical_columns(columns_list: List[str]) -> str:
    df = st.session_state.dataset
    try:
        for col in columns_list:
            if col in df.columns:
                df[col] = df[col].astype(str).str.lower().str.replace(" ", "_", regex=False)
            else:
                return f"La colonne '{col}' n'existe pas"
    except Exception as e:
        return f"Erreur lors de la normalisation: {e}"
    return f"Les colonnes {', '.join(columns_list)} ont été normalisées."

def detect_outliers(column_name: str, method: str = "IQR") -> str:
    df = st.session_state.dataset
    if df is None or df.empty:
        return "Le DataFrame est vide"
    if column_name not in df.columns:
        return f"La colonne '{column_name}' n'existe pas"
    if method == "IQR":
        Q1 = df[column_name].quantile(0.25)
        Q3 = df[column_name].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df[(df[column_name] < lower_bound) | (df[column_name] > upper_bound)]
    elif method == "Z-score":
        mean = df[column_name].mean()
        std = df[column_name].std()
        z_scores = (df[column_name] - mean) / std
        outliers = df[(z_scores < -3) | (z_scores > 3)]
    else:
        return f"Méthode '{method}' non reconnue."
    outlier_count = outliers.shape[0]
    outlier_indices = outliers.index.tolist()
    return f"Détection de {outlier_count} valeurs aberrantes dans la colonne '{column_name}' ({outlier_indices})."

def handle_outliers(column_name: str, method: str = "IQR", action: str = "remove") -> str:
    df = st.session_state.dataset
    if df is None or df.empty:
        return "Le DataFrame est vide"
    if column_name not in df.columns:
        return f"La colonne '{column_name}' n'existe pas"
    if method == "IQR":
        Q1 = df[column_name].quantile(0.25)
        Q3 = df[column_name].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df[(df[column_name] < lower_bound) | (df[column_name] > upper_bound)]
    elif method == "Z-score":
        mean = df[column_name].mean()
        std = df[column_name].std()
        z_scores = (df[column_name] - mean) / std
        outliers = df[(z_scores < -3) | (z_scores > 3)]
    else:
        return f"Méthode '{method}' non reconnue."
    outlier_count = outliers.shape[0]
    if action == "remove":
        df.drop(outliers.index, inplace=True)
        return f"Suppression de {outlier_count} valeurs aberrantes dans la colonne '{column_name}'."
    elif action == "impute":
        median = df[column_name].median()
        df.loc[outliers.index, column_name] = median
        return f"Imputation de {outlier_count} valeurs aberrantes dans la colonne '{column_name}' avec la médiane: {median}."
    else:
        return f"Action '{action}' non reconnue."

def ml_regression_imputer(target_column: str, feature_columns: List[str], model_type: str = "linear") -> str:
    df = st.session_state.dataset
    global models_registry
    if df is None or df.empty:
        return "Le DataFrame est vide"
    if target_column not in df.columns:
        return f"La colonne cible '{target_column}' n'existe pas"
    missing_count = df[target_column].isnull().sum()
    if missing_count == 0:
        return f"Aucune valeur manquante à imputer dans la colonne '{target_column}'"
    if model_type == "linear":
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
    elif model_type == "random_forest":
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    else:
        return f"Type de modèle '{model_type}' non reconnu."
    train_data = df[df[target_column].notnull()]
    test_data = df[df[target_column].isnull()]
    X_train = train_data[feature_columns]
    y_train = train_data[target_column]
    X_test = test_data[feature_columns]
    model.fit(X_train, y_train)
    imputed_values = model.predict(X_test)
    df.loc[df[target_column].isnull(), target_column] = imputed_values
    models_registry[f"{target_column}_imputer"] = model
    return f"Imputation de {missing_count} valeurs manquantes dans la colonne '{target_column}' avec '{model_type}'."

def ml_classifier_imputer(target_column: str, feature_columns: List[str], model_type: str = "random_forest") -> str:
    df = st.session_state.dataset
    global models_registry
    if df is None or df.empty:
        return "Le DataFrame est vide"
    if target_column not in df.columns:
        return f"La colonne cible '{target_column}' n'existe pas"
    missing_count = df[target_column].isnull().sum()
    if missing_count == 0:
        return f"Aucune valeur manquante à imputer dans la colonne '{target_column}'"
    if model_type == "random_forest":
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    elif model_type == "logistic_regression":
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(max_iter=1000, random_state=42)
    else:
        return f"Type de modèle '{model_type}' non reconnu."
    train_data = df[df[target_column].notnull()]
    test_data = df[df[target_column].isnull()]
    X_train = train_data[feature_columns]
    y_train = train_data[target_column]
    X_test = test_data[feature_columns]
    model.fit(X_train, y_train)
    imputed_values = model.predict(X_test)
    df.loc[df[target_column].isnull(), target_column] = imputed_values
    models_registry[f"{target_column}_classifier_imputer"] = model
    return f"Imputation de {missing_count} valeurs manquantes dans la colonne '{target_column}' avec '{model_type}'."

def linear_regression_model(target_column: str, feature_columns: List[str]) -> str:
    df = st.session_state.dataset
    global models_registry
    if df is None or df.empty:
        return "Le DataFrame est vide"
    if target_column not in df.columns:
        return f"La colonne cible '{target_column}' n'existe pas"
    missing_cols = [col for col in feature_columns if col not in df.columns]
    if missing_cols:
        return f"Colonnes manquantes pour l'entraînement: {', '.join(missing_cols)}"
    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    X = df[feature_columns]
    y = df[target_column]
    model.fit(X, y)
    models_registry[f"{target_column}_linear_regression"] = model
    return f"Modèle de régression linéaire entraîné pour '{target_column}'."

def random_forest_model(target_column: str, feature_columns: List[str]) -> str:
    df = st.session_state.dataset
    global models_registry
    if df is None or df.empty:
        return "Le DataFrame est vide"
    if target_column not in df.columns:
        return f"La colonne cible '{target_column}' n'existe pas"
    missing_cols = [col for col in feature_columns if col not in df.columns]
    if missing_cols:
        return f"Colonnes manquantes pour l'entraînement: {', '.join(missing_cols)}"
    from sklearn.ensemble import RandomForestRegressor
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    X = df[feature_columns]
    y = df[target_column]
    model.fit(X, y)
    models_registry[f"{target_column}_random_forest"] = model
    return f"Modèle random forest entraîné pour '{target_column}'."

def gradient_boosting_regressor_model(target_column: str, feature_columns: List[str]) -> str:
    df = st.session_state.dataset
    global models_registry
    if df is None or df.empty:
        return "Le DataFrame est vide"
    if target_column not in df.columns:
        return f"La colonne cible '{target_column}' n'existe pas"
    missing_cols = [col for col in feature_columns if col not in df.columns]
    if missing_cols:
        return f"Colonnes manquantes pour l'entraînement: {', '.join(missing_cols)}"
    from sklearn.ensemble import GradientBoostingRegressor
    model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    X = df[feature_columns]
    y = df[target_column]
    model.fit(X, y)
    models_registry[f"{target_column}_gradient_boosting"] = model
    return f"Gradient boosting regressor entraîné pour '{target_column}'."

def random_forest_classifier_model(target_column: str, feature_columns: List[str]) -> str:
    df = st.session_state.dataset
    global models_registry
    if df is None or df.empty:
        return "Le DataFrame est vide"
    if target_column not in df.columns:
        return f"La colonne cible '{target_column}' n'existe pas"
    missing_cols = [col for col in feature_columns if col not in df.columns]
    if missing_cols:
        return f"Colonnes manquantes pour l'entraînement: {', '.join(missing_cols)}"
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    X = df[feature_columns]
    y = df[target_column]
    model.fit(X, y)
    models_registry[f"{target_column}_random_forest_classifier"] = model
    return f"Random forest classifier entraîné pour '{target_column}'."

def gradient_boosting_classifier_model(target_column: str, feature_columns: List[str]) -> str:
    df = st.session_state.dataset
    global models_registry
    if df is None or df.empty:
        return "Le DataFrame est vide"
    if target_column not in df.columns:
        return f"La colonne cible '{target_column}' n'existe pas"
    missing_cols = [col for col in feature_columns if col not in df.columns]
    if missing_cols:
        return f"Colonnes manquantes pour l'entraînement: {', '.join(missing_cols)}"
    from sklearn.ensemble import GradientBoostingClassifier
    model = GradientBoostingClassifier(n_estimators=100, random_state=42)
    X = df[feature_columns]
    y = df[target_column]
    model.fit(X, y)
    models_registry[f"{target_column}_gradient_boosting_classifier"] = model
    return f"Gradient boosting classifier entraîné pour '{target_column}'."

def kmeans_clustering_model(feature_columns: List[str], n_clusters: int = 3) -> str:
    df = st.session_state.dataset
    global models_registry
    if df is None or df.empty:
        return "Le DataFrame est vide"
    missing_cols = [col for col in feature_columns if col not in df.columns]
    if missing_cols:
        return f"Colonnes manquantes pour clustering: {', '.join(missing_cols)}"
    from sklearn.cluster import KMeans
    model = KMeans(n_clusters=n_clusters, random_state=42)
    X = df[feature_columns]
    model.fit(X)
    df["cluster"] = model.labels_
    models_registry[f"kmeans_clustering_{n_clusters}_clusters"] = model
    return f"KMeans entraîné ({n_clusters} clusters)."

def dbscan_clustering_model(feature_columns: List[str], eps: float = 0.5, min_samples: int = 5) -> str:
    df = st.session_state.dataset
    global models_registry
    if df is None or df.empty:
        return "Le DataFrame est vide"
    missing_cols = [col for col in feature_columns if col not in df.columns]
    if missing_cols:
        return f"Colonnes manquantes pour clustering: {', '.join(missing_cols)}"
    from sklearn.cluster import DBSCAN
    model = DBSCAN(eps=eps, min_samples=min_samples)
    X = df[feature_columns]
    model.fit(X)
    df["cluster"] = model.labels_
    models_registry[f"dbscan_clustering_eps_{eps}_min_samples_{min_samples}"] = model
    return f"DBSCAN entraîné (eps={eps}, min_samples={min_samples})."

def get_models_list() -> list:
    return list(models_registry.keys())

def make_prediction(model_name: str, input_data: Dict[str, Any]) -> str:
    if model_name not in models_registry:
        return f"Modèle '{model_name}' introuvable"
    model = models_registry[model_name]
    try:
        input_df = pd.DataFrame([input_data])
        prediction = model.predict(input_df)
        return f"Prédiction: {prediction[0]}"
    except Exception as e:
        return f"Erreur prediction: {e}"

def save_model(model_name: str, file_path: str) -> str:
    if model_name not in models_registry:
        return f"Modèle '{model_name}' introuvable"
    try:
        import joblib
        joblib.dump(models_registry[model_name], file_path)
        return f"Modèle '{model_name}' enregistré: {file_path}"
    except Exception as e:
        return f"Erreur save_model: {e}"

def load_model(model_name: str, file_path: str) -> str:
    try:
        import joblib
        model = joblib.load(file_path)
        models_registry[model_name] = model
        return f"Modèle '{model_name}' chargé depuis {file_path}"
    except Exception as e:
        return f"Erreur load_model: {e}"

def initialize_llm():
    """Initialise le modèle LLM via le tunnel ngrok"""
    if not st.session_state.llm_initialized:
        try:
            model_name = st.session_state.get("selected_model", "qwen3:4b")
            st.session_state.llm = ChatOllama(
                model=model_name,
                base_url=OLLAMA_TUNNEL_URL,
                temperature=0.7
            )
            st.session_state.llm_initialized = True
            st.session_state.current_model = model_name
            return True
        except Exception as e:
            st.error(f"❌ Erreur connexion au tunnel Ollama: {str(e)}")
            return False
    return True

def get_available_models():
    """Récupère la liste des modèles disponibles sur le serveur Ollama"""
    try:
        response = requests.get(f"{OLLAMA_TUNNEL_URL}/api/tags", timeout=5)
        if response.status_code == 200:
            data = response.json()
            models = [m["name"] for m in data.get("models", [])]
            return models
        return []
    except Exception as e:
        st.warning(f"⚠️ Impossible de récupérer la liste des modèles: {str(e)}")
        return []

def load_dataset(uploaded_file):
    if uploaded_file is None:
        return None
    try:
        if uploaded_file.name.endswith(".csv"):
            return pd.read_csv(uploaded_file)
        else:
            return pd.read_excel(uploaded_file)
    except Exception as exc:
        st.error(f"❌ Erreur de lecture du fichier : {exc}")
        return None

def summarize_dataset(df: pd.DataFrame):
    return {
        "shape": df.shape,
        "columns": df.columns.tolist(),
        "missing_values": df.isna().sum().sort_values(ascending=False).head(10),
        "types": df.dtypes.astype(str).to_dict(),
    }

def ask_imora_agent(question: str, df: pd.DataFrame = None) -> tuple[str, dict]:
    """Pose une question à Imora via LangChain + Ollama
    Retourne (réponse, métadonnées)
    """
    if not initialize_llm():
        return "❌ Impossible de se connecter au tunnel Ollama.", {"error": True}
    
    try:
        import time
        start_time = time.time()
        
        # Construire le contexte avec le dataset
        context = f"Utilisateur : {question}"
        
        dataset_info = {}
        if df is not None:
            dataset_info = {
                "rows": df.shape[0],
                "columns": df.shape[1],
                "column_names": df.columns.tolist()[:10]
            }
            context += f"\n\nContext Dataset: {df.shape[0]} lignes, {df.shape[1]} colonnes"
            context += f"\nColonnes: {', '.join(df.columns.tolist()[:10])}"
            if df.shape[1] > 10:
                context += f"... (+{df.shape[1] - 10} autres)"
        
        # Appeler le LLM
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=context)
        ]
        
        response = st.session_state.llm.invoke(messages)
        elapsed_time = time.time() - start_time
        
        # Métadonnées pour la réflexion
        metadata = {
            "model": st.session_state.current_model,
            "time_seconds": round(elapsed_time, 2),
            "dataset_rows": dataset_info.get("rows", 0),
            "dataset_columns": dataset_info.get("columns", 0),
            "tunnel_url": OLLAMA_TUNNEL_URL,
        }
        
        return response.content, metadata
    except Exception as e:
        error_msg = f"❌ Erreur lors de la requête: {str(e)}\n{traceback.format_exc()}"
        return error_msg, {"error": True, "error_message": str(e)}

# ============ SIDEBAR ============
with st.sidebar:
    st.title("🧠 Imora")
    st.markdown("*Data Intelligence & Chat avec Agent IA*")
    
    st.markdown("### 📁 Dataset")
    uploaded_file = st.file_uploader("Importer un dataset", type=["csv", "xlsx", "xls"])
    
    if uploaded_file is not None:
        df = load_dataset(uploaded_file)
        if df is not None:
            st.session_state.dataset = df
            st.session_state.dataset_name = uploaded_file.name
            st.success(f"✅ {uploaded_file.name} chargé")
    
    if st.session_state.dataset is not None:
        st.markdown("### 📊 Statut")
        st.metric("Colonnes", st.session_state.dataset.shape[1])
        st.metric("Lignes", st.session_state.dataset.shape[0])
        st.metric("Valeurs manquantes", st.session_state.dataset.isna().sum().sum())
    else:
        st.info("📥 Chargez un dataset pour commencer")
    
    st.markdown("---")
    
    # Gestion des modèles
    st.markdown("### 🤖 Modèle Ollama")
    available_models = get_available_models()
    
    if available_models:
        st.info(f"✅ Tunnel connecté ({len(available_models)} modèle(s))")
        selected_model = st.selectbox(
            "Choisir un modèle",
            available_models,
            index=0 if available_models else None,
            key="model_selector"
        )
        st.session_state.selected_model = selected_model
        st.session_state.llm_initialized = False  # Réinitialiser pour charger le nouveau modèle
    else:
        st.error(f"❌ Impossible de se connecter à {OLLAMA_TUNNEL_URL}")
        st.session_state.selected_model = "qwen3:4b"  # Défaut

# ============ MAIN CONTENT ============
st.header("🧠 Imora - Interface Streamlit")

if st.session_state.dataset is None:
    st.info("📥 Importez un fichier CSV ou Excel depuis la barre latérale pour commencer")
    st.stop()

# Afficher le résumé du dataset
summary = summarize_dataset(st.session_state.dataset)

col1, col2 = st.columns([2, 1])
with col1:
    st.subheader("📋 Aperçu")
    st.dataframe(st.session_state.dataset.head(10), use_container_width=True)

with col2:
    st.subheader("📊 Résumé")
    st.metric("Colonnes", summary["shape"][1])
    st.metric("Lignes", summary["shape"][0])

# Section diagnostics rapides
st.markdown("---")
col_diag1, col_diag2 = st.columns(2)

with col_diag1:
    if st.button("🔍 Diagnostic complet"):
        with st.spinner("Analyse en cours..."):
            desc = get_dataset_description(st.session_state.dataset)
            st.markdown(desc)

with col_diag2:
    if st.button("❌ Valeurs manquantes"):
        with st.spinner("Détection..."):
            missing = detect_missing_values(st.session_state.dataset)
            st.markdown(missing)

if st.button("📈 Détecter outliers"):
    with st.spinner("Analyse outliers..."):
        outliers = detect_outliers_simple(st.session_state.dataset)
        st.markdown(outliers)

# Section Chat avec Imora
st.markdown("---")
st.subheader("💬 Conversation avec Imora")

# Afficher l'historique avec métadonnées
for idx, (role, text) in enumerate(st.session_state.chat_history):
    if role == "user":
        st.info(f"👤 **Vous**: {text}")
    else:
        # Réponse Imora avec métadonnées en expander collapsible
        st.success(f"🤖 **Imora**: {text}")
        
        # Afficher les métadonnées dans un expander grisâtre et fin
        if idx < len(st.session_state.chat_metadata):
            metadata = st.session_state.chat_metadata[idx]
            with st.expander("💭 Réflexion et détails"):
                # CSS pour style fin et grisâtre
                st.markdown("""
                <style>
                .reflection-text {
                    font-size: 0.85rem;
                    color: #707070;
                    line-height: 1.4;
                    font-weight: 300;
                }
                </style>
                """, unsafe_allow_html=True)
                
                reflection_html = "<div class='reflection-text'>"
                reflection_html += f"<strong>Modèle:</strong> {metadata.get('model', 'N/A')}<br>"
                reflection_html += f"<strong>Temps de réponse:</strong> {metadata.get('time_seconds', 'N/A')}s<br>"
                
                if metadata.get("dataset_rows", 0) > 0:
                    reflection_html += f"<strong>Dataset analysé:</strong> {metadata.get('dataset_rows')} lignes × {metadata.get('dataset_columns')} colonnes<br>"
                
                reflection_html += f"<strong>Tunnel:</strong> <code>{metadata.get('tunnel_url', 'N/A')}</code><br>"
                reflection_html += f"<strong>Système prompt:</strong> Agent spécialisé en analyse de données"
                reflection_html += "</div>"
                
                st.markdown(reflection_html, unsafe_allow_html=True)

# Input pour poser une question
col_q1, col_q2 = st.columns([5, 1])
with col_q1:
    question = st.text_input("Posez une question à Imora", key="chat_input", placeholder="Ex: Analyse les valeurs manquantes...")
with col_q2:
    send_button = st.button("📤 Envoyer", key="send_btn")

if send_button and question:
    st.session_state.chat_history.append(("user", question))
    
    with st.spinner("Imora pense..."):
        response, metadata = ask_imora_agent(question, st.session_state.dataset)
    
    st.session_state.chat_history.append(("agent", response))
    st.session_state.chat_metadata.append(metadata)
    st.rerun()
