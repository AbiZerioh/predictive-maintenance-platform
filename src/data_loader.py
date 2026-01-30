"""
Module de chargement des données pour la maintenance prédictive.
Charge le dataset NASA C-MAPSS et prépare les données pour l'analyse.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Optional
import requests
from io import StringIO
import os
import yaml

class DataLoader:
    """
    Classe pour charger et préparer les données de maintenance prédictive.
    
    Attributes:
        config (dict): Configuration du projet
        data (dict): Données chargées
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialise le chargeur de données.
        
        Args:
            config_path: Chemin vers le fichier de configuration
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.data = {}
        print("✅ Chargeur de données initialisé")
    
    def load_nasa_data(self, use_cache: bool = True) -> Dict[str, pd.DataFrame]:
        """
        Charge le dataset NASA C-MAPSS depuis GitHub ou cache local.
        
        Args:
            use_cache: Si True, utilise les données en cache si disponibles
            
        Returns:
            Dict avec les DataFrames train, test et truth
        """
        print("📥 Chargement des données NASA C-MAPSS...")
        
        # Chemins des fichiers
        cache_dir = self.config['paths']['data']
        raw_dir = os.path.join(cache_dir, 'raw')
        os.makedirs(raw_dir, exist_ok=True)
        
        train_path = os.path.join(raw_dir, 'train.csv')
        test_path = os.path.join(raw_dir, 'test.csv')
        truth_path = os.path.join(raw_dir, 'truth.csv')
        
        # Vérifier le cache
        if use_cache and all(os.path.exists(p) for p in [train_path, test_path, truth_path]):
            print("  Utilisation des données en cache...")
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            truth_df = pd.read_csv(truth_path)
        else:
            print("  Téléchargement depuis GitHub...")
            # URLs des données
            urls = {
                'train': 'https://raw.githubusercontent.com/ashishpatel26/Predictive-Maintenance-using-LSTM/master/PM_train.txt',
                'test': 'https://raw.githubusercontent.com/ashishpatel26/Predictive-Maintenance-using-LSTM/master/PM_test.txt',
                'truth': 'https://raw.githubusercontent.com/ashishpatel26/Predictive-Maintenance-using-LSTM/master/PM_truth.txt'
            }
            
            # Télécharger les données
            train_df = self._download_data(urls['train'])
            test_df = self._download_data(urls['test'])
            truth_df = self._download_data(urls['truth'])
            
            # Sauvegarder en cache
            train_df.to_csv(train_path, index=False)
            test_df.to_csv(test_path, index=False)
            truth_df.to_csv(truth_path, index=False)
            print("  Données sauvegardées en cache")
        
        # Nettoyer et nommer les colonnes
        train_df = self._clean_data(train_df)
        test_df = self._clean_data(test_df)
        
        columns = ['unit_id', 'time_cycle']
        columns += [f'operational_setting_{i}' for i in range(1, 4)]
        columns += [f'sensor_measurement_{i}' for i in range(1, 22)]
        
        train_df.columns = columns[:train_df.shape[1]]
        test_df.columns = columns[:test_df.shape[1]]
        
        # Préparer les données de test avec RUL
        truth_df.columns = ['RUL']
        test_df = pd.concat([test_df, truth_df], axis=1)
        
        self.data = {
            'train': train_df,
            'test': test_df,
            'truth': truth_df
        }
        
        print(f"✅ Données chargées: {len(train_df)} lignes d'entraînement")
        print(f"                     {len(test_df)} lignes de test")
        
        return self.data
    
    def _download_data(self, url: str) -> pd.DataFrame:
        """
        Télécharge des données depuis une URL.
        
        Args:
            url: URL des données
            
        Returns:
            DataFrame avec les données
        """
        response = requests.get(url)
        df = pd.read_csv(StringIO(response.text), sep=" ", header=None)
        df.dropna(axis=1, how='all', inplace=True)
        return df
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Nettoie les données brutes.
        
        Args:
            df: DataFrame à nettoyer
            
        Returns:
            DataFrame nettoyé
        """
        # Supprimer les colonnes vides
        df = df.dropna(axis=1, how='all')
        
        # Supprimer les doublons
        df = df.drop_duplicates()
        
        return df
    
    def calculate_rul(self, df: pd.DataFrame, label: str = 'train') -> pd.DataFrame:
        """
        Calcule le RUL (Remaining Useful Life) pour chaque observation.
        
        Args:
            df: DataFrame avec les données
            label: 'train' ou 'test'
            
        Returns:
            DataFrame avec colonne RUL ajoutée
        """
        print(f"📊 Calcul du RUL pour les données {label}...")
        
        df_rul = df.copy()
        
        if 'RUL' in df_rul.columns:
            print("  RUL déjà présent dans les données")
            return df_rul
        
        # Calculer le cycle maximum pour chaque unité
        max_cycle = df_rul.groupby('unit_id')['time_cycle'].max().reset_index()
        max_cycle.columns = ['unit_id', 'max_cycle']
        
        # Fusionner et calculer RUL
        df_rul = df_rul.merge(max_cycle, on='unit_id', how='left')
        df_rul['RUL'] = df_rul['max_cycle'] - df_rul['time_cycle']
        df_rul.drop('max_cycle', axis=1, inplace=True)
        
        print(f"  RUL calculé: {df_rul['RUL'].min():.0f} à {df_rul['RUL'].max():.0f} cycles")
        
        return df_rul
    
    def prepare_sequences(self, df: pd.DataFrame, sequence_length: int = 50) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prépare les séquences pour les modèles LSTM.
        
        Args:
            df: DataFrame avec les données
            sequence_length: Longueur des séquences
            
        Returns:
            Tuple (X, y) pour l'entraînement
        """
        print(f"🔄 Préparation des séquences (longueur: {sequence_length})...")
        
        sequences = []
        targets = []
        
        # Identifier les colonnes de capteurs
        sensor_cols = [col for col in df.columns if 'sensor' in col]
        
        for unit_id in df['unit_id'].unique():
            unit_data = df[df['unit_id'] == unit_id]
            
            # Trier par cycle temporel
            unit_data = unit_data.sort_values('time_cycle')
            
            # Créer des séquences glissantes
            for i in range(len(unit_data) - sequence_length):
                seq = unit_data.iloc[i:i + sequence_length][sensor_cols].values
                target = unit_data.iloc[i + sequence_length]['RUL']
                
                sequences.append(seq)
                targets.append(target)
        
        X = np.array(sequences)
        y = np.array(targets)
        
        print(f"  ✅ {len(sequences)} séquences créées")
        print(f"  X shape: {X.shape}, y shape: {y.shape}")
        
        return X, y
    
    def get_data_summary(self) -> Dict:
        """
        Retourne un résumé des données chargées.
        
        Returns:
            Dict avec les statistiques des données
        """
        if not self.data:
            return {"error": "Aucune donnée chargée"}
        
        train_df = self.data['train']
        test_df = self.data['test']
        
        summary = {
            'train': {
                'shape': train_df.shape,
                'units': train_df['unit_id'].nunique(),
                'cycles_max': train_df.groupby('unit_id')['time_cycle'].max().mean(),
                'sensors': len([col for col in train_df.columns if 'sensor' in col])
            },
            'test': {
                'shape': test_df.shape,
                'units': test_df['unit_id'].nunique(),
                'has_rul': 'RUL' in test_df.columns
            },
            'features': {
                'sensor_cols': [col for col in train_df.columns if 'sensor' in col][:5],
                'op_setting_cols': [col for col in train_df.columns if 'operational' in col]
            }
        }
        
        return summary

def main():
    """Fonction principale pour tester le module."""
    print("=" * 50)
    print("TEST DU MODULE DE CHARGEMENT DE DONNÉES")
    print("=" * 50)
    
    # Initialiser le chargeur
    loader = DataLoader()
    
    # Charger les données
    data = loader.load_nasa_data(use_cache=True)
    
    # Calculer RUL pour les données d'entraînement
    train_with_rul = loader.calculate_rul(data['train'], 'train')
    
    # Préparer les séquences
    X, y = loader.prepare_sequences(train_with_rul, sequence_length=50)
    
    # Afficher le résumé
    summary = loader.get_data_summary()
    print("\n📋 RÉSUMÉ DES DONNÉES:")
    print(f"  Données d'entraînement: {summary['train']['shape']}")
    print(f"  Unités d'entraînement: {summary['train']['units']}")
    print(f"  Capteurs: {summary['train']['sensors']}")
    print(f"  Forme des séquences: {X.shape}")
    
    return data

if __name__ == "__main__":
    data = main()
