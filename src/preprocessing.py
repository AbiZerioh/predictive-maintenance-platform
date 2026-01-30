"""
Module de prétraitement des données pour la maintenance prédictive.
Nettoyage, normalisation et feature engineering.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, List
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

class DataPreprocessor:
    """
    Classe pour le prétraitement des données de maintenance.
    
    Attributes:
        scalers (dict): Dictionnaire des scalers entraînés
        imputers (dict): Dictionnaire des imputers
    """
    
    def __init__(self):
        """Initialise le préprocesseur."""
        self.scalers = {}
        self.imputers = {}
        print("✅ Préprocesseur initialisé")
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Nettoie les données brutes.
        
        Args:
            df: DataFrame à nettoyer
            
        Returns:
            DataFrame nettoyé
        """
        print("🧹 Nettoyage des données...")
        
        df_clean = df.copy()
        
        # 1. Supprimer les colonnes avec trop de valeurs manquantes
        missing_threshold = 0.5
        cols_to_drop = []
        
        for col in df_clean.columns:
            missing_ratio = df_clean[col].isnull().sum() / len(df_clean)
            if missing_ratio > missing_threshold:
                cols_to_drop.append(col)
        
        if cols_to_drop:
            print(f"  Suppression de {len(cols_to_drop)} colonnes avec >{missing_threshold*100}% de valeurs manquantes")
            df_clean = df_clean.drop(columns=cols_to_drop)
        
        # 2. Imputer les valeurs manquantes
        for col in df_clean.select_dtypes(include=[np.number]).columns:
            if df_clean[col].isnull().any():
                imputer = SimpleImputer(strategy='median')
                df_clean[col] = imputer.fit_transform(df_clean[[col]]).ravel()
                self.imputers[col] = imputer
        
        # 3. Supprimer les doublons
        initial_len = len(df_clean)
        df_clean = df_clean.drop_duplicates()
        duplicates_removed = initial_len - len(df_clean)
        
        if duplicates_removed > 0:
            print(f"  {duplicates_removed} doublons supprimés")
        
        print(f"  ✅ Données nettoyées: {df_clean.shape}")
        
        return df_clean
    
    def normalize_features(self, df: pd.DataFrame, method: str = 'standard') -> pd.DataFrame:
        """
        Normalise les caractéristiques numériques.
        
        Args:
            df: DataFrame à normaliser
            method: 'standard' (StandardScaler) ou 'minmax' (MinMaxScaler)
            
        Returns:
            DataFrame normalisé
        """
        print(f"📏 Normalisation des caractéristiques (méthode: {method})...")
        
        df_norm = df.copy()
        
        # Identifier les colonnes numériques à normaliser
        numeric_cols = df_norm.select_dtypes(include=[np.number]).columns.tolist()
        
        # Exclure certaines colonnes
        exclude_cols = ['unit_id', 'time_cycle', 'RUL']
        numeric_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        if not numeric_cols:
            print("  ⚠️  Aucune colonne à normaliser")
            return df_norm
        
        # Appliquer la normalisation
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        else:
            raise ValueError(f"Méthode de normalisation inconnue: {method}")
        
        # Normaliser les données
        df_norm[numeric_cols] = scaler.fit_transform(df_norm[numeric_cols])
        
        # Sauvegarder le scaler
        self.scalers[method] = scaler
        
        print(f"  ✅ {len(numeric_cols)} colonnes normalisées")
        
        return df_norm
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Crée de nouvelles caractéristiques à partir des données existantes.
        
        Args:
            df: DataFrame d'entrée
            
        Returns:
            DataFrame avec nouvelles caractéristiques
        """
        print("🔧 Création de caractéristiques...")
        
        df_features = df.copy()
        
        # Vérifier que les colonnes nécessaires existent
        if 'time_cycle' not in df_features.columns:
            print("  ⚠️  Colonne 'time_cycle' non trouvée")
            return df_features
        
        # Caractéristiques temporelles
        if 'time_cycle' in df_features.columns:
            # Cycle normalisé par unité
            df_features['cycle_norm'] = df_features.groupby('unit_id')['time_cycle'].transform(
                lambda x: (x - x.min()) / (x.max() - x.min()) if x.max() != x.min() else 0
            )
            
            # Différence avec le cycle précédent
            df_features['cycle_diff'] = df_features.groupby('unit_id')['time_cycle'].diff().fillna(0)
        
        # Caractéristiques statistiques par unité
        sensor_cols = [col for col in df_features.columns if 'sensor' in col]
        
        if sensor_cols:
            # Moyennes glissantes
            for sensor in sensor_cols[:5]:  # Limiter aux 5 premiers capteurs
                df_features[f'{sensor}_rolling_mean'] = df_features.groupby('unit_id')[sensor].transform(
                    lambda x: x.rolling(window=5, min_periods=1).mean()
                )
                
                df_features[f'{sensor}_rolling_std'] = df_features.groupby('unit_id')[sensor].transform(
                    lambda x: x.rolling(window=5, min_periods=1).std()
                )
        
        # Caractéristiques d'ingénierie
        if len(sensor_cols) >= 2:
            # Ratio entre capteurs
            df_features['sensor_ratio_1_2'] = df_features[sensor_cols[0]] / (df_features[sensor_cols[1]] + 1e-10)
        
        print(f"  ✅ Caractéristiques créées: {len(df_features.columns)} colonnes totales")
        
        return df_features
    
    def detect_outliers(self, df: pd.DataFrame, threshold: float = 3.0) -> Dict:
        """
        Détecte les outliers dans les données.
        
        Args:
            df: DataFrame à analyser
            threshold: Seuil en écarts-types
            
        Returns:
            Dict avec informations sur les outliers
        """
        print(f"🔍 Détection des outliers (seuil: {threshold}σ)...")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        outliers_info = {
            'total_outliers': 0,
            'outliers_per_column': {},
            'percentage_outliers': 0
        }
        
        for col in numeric_cols:
            if col in ['unit_id', 'time_cycle']:
                continue
            
            values = df[col]
            mean = values.mean()
            std = values.std()
            
            if std == 0:
                continue
            
            # Calculer les z-scores
            z_scores = np.abs((values - mean) / std)
            
            # Compter les outliers
            outliers = (z_scores > threshold).sum()
            
            if outliers > 0:
                outliers_info['outliers_per_column'][col] = {
                    'count': int(outliers),
                    'percentage': outliers / len(df) * 100,
                    'mean': float(mean),
                    'std': float(std)
                }
                outliers_info['total_outliers'] += outliers
        
        total_values = len(df) * len(numeric_cols)
        if total_values > 0:
            outliers_info['percentage_outliers'] = outliers_info['total_outliers'] / total_values * 100
        
        print(f"  Outliers détectés: {outliers_info['total_outliers']} ({outliers_info['percentage_outliers']:.2f}%)")
        
        return outliers_info
    
    def remove_outliers(self, df: pd.DataFrame, threshold: float = 3.0) -> pd.DataFrame:
        """
        Supprime les outliers des données.
        
        Args:
            df: DataFrame avec outliers
            threshold: Seuil en écarts-types
            
        Returns:
            DataFrame sans outliers
        """
        print(f"🗑️  Suppression des outliers (seuil: {threshold}σ)...")
        
        df_clean = df.copy()
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
        
        # Identifier les outliers
        mask = pd.Series([True] * len(df_clean))
        
        for col in numeric_cols:
            if col in ['unit_id', 'time_cycle', 'RUL']:
                continue
            
            values = df_clean[col]
            mean = values.mean()
            std = values.std()
            
            if std == 0:
                continue
            
            z_scores = np.abs((values - mean) / std)
            mask = mask & (z_scores <= threshold)
        
        outliers_removed = len(df_clean) - mask.sum()
        
        if outliers_removed > 0:
            print(f"  {outliers_removed} outliers supprimés ({outliers_removed/len(df_clean)*100:.1f}%)")
            df_clean = df_clean[mask].reset_index(drop=True)
        
        return df_clean
    
    def prepare_for_training(self, df: pd.DataFrame, target_col: str = 'RUL') -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prépare les données pour l'entraînement.
        
        Args:
            df: DataFrame complet
            target_col: Nom de la colonne cible
            
        Returns:
            Tuple (X, y) prêts pour l'entraînement
        """
        print("🎯 Préparation des données pour l'entraînement...")
        
        # Copier les données
        df_prep = df.copy()
        
        # Séparer les caractéristiques et la cible
        if target_col not in df_prep.columns:
            raise ValueError(f"Colonne cible '{target_col}' non trouvée")
        
        y = df_prep[target_col]
        X = df_prep.drop(columns=[target_col])
        
        # Exclure les colonnes d'identification
        exclude_cols = ['unit_id', 'time_cycle']
        X = X.drop(columns=[col for col in exclude_cols if col in X.columns])
        
        # Vérifier les valeurs manquantes
        missing_cols = X.columns[X.isnull().any()].tolist()
        if missing_cols:
            print(f"  Imputation des valeurs manquantes dans {len(missing_cols)} colonnes...")
            for col in missing_cols:
                imputer = SimpleImputer(strategy='median')
                X[col] = imputer.fit_transform(X[[col]]).ravel()
        
        print(f"  X shape: {X.shape}, y shape: {y.shape}")
        
        return X, y
    
    def run_pipeline(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Exécute le pipeline complet de prétraitement.
        
        Args:
            df: DataFrame brut
            
        Returns:
            DataFrame prétraité
        """
        print("\n" + "="*50)
        print("PIPELINE DE PRÉTRAITEMENT COMPLET")
        print("="*50)
        
        # 1. Nettoyage
        df_clean = self.clean_data(df)
        
        # 2. Détection des outliers
        outliers_info = self.detect_outliers(df_clean)
        
        # 3. Suppression des outliers
        df_no_outliers = self.remove_outliers(df_clean)
        
        # 4. Création de caractéristiques
        df_features = self.create_features(df_no_outliers)
        
        # 5. Normalisation
        df_normalized = self.normalize_features(df_features, method='standard')
        
        print("\n" + "="*50)
        print("✅ PRÉTRAITEMENT TERMINÉ")
        print("="*50)
        print(f"Shape initiale: {df.shape}")
        print(f"Shape finale: {df_normalized.shape}")
        print(f"Outliers traités: {outliers_info['total_outliers']}")
        
        return df_normalized

def main():
    """Fonction principale pour tester le module."""
    print("="*50)
    print("TEST DU MODULE DE PRÉTRAITEMENT")
    print("="*50)
    
    # Créer des données de test
    np.random.seed(42)
    n_samples = 1000
    
    test_data = pd.DataFrame({
        'unit_id': np.repeat(range(10), n_samples//10),
        'time_cycle': np.tile(range(n_samples//10), 10),
        'sensor_1': np.random.normal(100, 10, n_samples),
        'sensor_2': np.random.normal(50, 5, n_samples),
        'sensor_3': np.random.normal(20, 3, n_samples),
        'RUL': np.random.uniform(10, 200, n_samples)
    })
    
    # Ajouter quelques outliers
    test_data.loc[::100, 'sensor_1'] = 500
    test_data.loc[::50, 'sensor_2'] = 200
    
    # Initialiser le préprocesseur
    preprocessor = DataPreprocessor()
    
    # Exécuter le pipeline
    processed_data = preprocessor.run_pipeline(test_data)
    
    print(f"\n📋 DONNÉES TRAITÉES:")
    print(f"  Colonnes: {list(processed_data.columns)}")
    print(f"  Shape: {processed_data.shape}")
    
    return processed_data

if __name__ == "__main__":
    processed_data = main()
