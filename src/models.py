"""
Module de modèles de Machine Learning pour la maintenance prédictive.
LSTM, XGBoost, Random Forest pour la prédiction de RUL.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Any, List
import pickle
import joblib
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

# Deep Learning
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

class PredictiveModels:
    """
    Classe pour entraîner et gérer les modèles de prédiction.
    
    Attributes:
        models (dict): Dictionnaire des modèles entraînés
        histories (dict): Historique d'entraînement des modèles
        scaler (StandardScaler): Scaler pour les caractéristiques
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialise les modèles de prédiction.
        
        Args:
            config: Configuration des modèles
        """
        self.config = config or {}
        self.models = {}
        self.histories = {}
        self.scaler = StandardScaler()
        
        # Configuration TensorFlow
        tf.random.set_seed(42)
        
        print("✅ Modèles de prédiction initialisés")
    
    def create_lstm_model(self, input_shape: Tuple) -> Model:
        """
        Crée un modèle LSTM pour la prédiction de RUL.
        
        Args:
            input_shape: Forme des données d'entrée (timesteps, features)
            
        Returns:
            Modèle LSTM compilé
        """
        print("🤖 Création du modèle LSTM...")
        
        model = Sequential([
            Input(shape=input_shape),
            LSTM(128, return_sequences=True),
            BatchNormalization(),
            Dropout(0.2),
            
            LSTM(64, return_sequences=False),
            BatchNormalization(),
            Dropout(0.2),
            
            Dense(32, activation='relu'),
            BatchNormalization(),
            Dropout(0.1),
            
            Dense(16, activation='relu'),
            Dense(1)  # Prédiction RUL
        ])
        
        # Compiler le modèle
        optimizer = Adam(learning_rate=0.001)
        model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae', 'mse']
        )
        
        print(f"  ✅ Modèle LSTM créé: {model.summary()}")
        
        return model
    
    def create_xgboost_model(self) -> xgb.XGBRegressor:
        """
        Crée un modèle XGBoost.
        
        Returns:
            Modèle XGBoost configuré
        """
        print("🤖 Création du modèle XGBoost...")
        
        model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbosity=0
        )
        
        print("  ✅ Modèle XGBoost créé")
        
        return model
    
    def create_random_forest_model(self) -> RandomForestRegressor:
        """
        Crée un modèle Random Forest.
        
        Returns:
            Modèle Random Forest configuré
        """
        print("🤖 Création du modèle Random Forest...")
        
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42,
            n_jobs=-1
        )
        
        print("  ✅ Modèle Random Forest créé")
        
        return model
    
    def train_lstm(self, X_train: np.ndarray, y_train: np.ndarray, 
                   X_val: np.ndarray = None, y_val: np.ndarray = None,
                   epochs: int = 50, batch_size: int = 32) -> Tuple[Model, Dict]:
        """
        Entraîne un modèle LSTM.
        
        Args:
            X_train, y_train: Données d'entraînement
            X_val, y_val: Données de validation
            epochs: Nombre d'époques
            batch_size: Taille des lots
            
        Returns:
            Tuple (modèle, historique)
        """
        print("\n" + "="*50)
        print("ENTRAÎNEMENT DU MODÈLE LSTM")
        print("="*50)
        
        # Créer le modèle
        model = self.create_lstm_model(X_train.shape[1:])
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            ModelCheckpoint(
                'models/lstm_best.h5',
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1
            )
        ]
        
        # Entraîner le modèle
        if X_val is not None:
            validation_data = (X_val, y_val)
            validation_split = None
        else:
            validation_data = None
            validation_split = 0.2
        
        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=1
        )
        
        # Sauvegarder
        self.models['lstm'] = model
        self.histories['lstm'] = history.history
        
        print("✅ LSTM entraîné")
        
        return model, history.history
    
    def train_xgboost(self, X_train: np.ndarray, y_train: np.ndarray) -> xgb.XGBRegressor:
        """
        Entraîne un modèle XGBoost.
        
        Args:
            X_train, y_train: Données d'entraînement
            
        Returns:
            Modèle XGBoost entraîné
        """
        print("\n" + "="*50)
        print("ENTRAÎNEMENT DU MODÈLE XGBOOST")
        print("="*50)
        
        # Aplatir les données si nécessaire (pour séquences LSTM)
        if len(X_train.shape) == 3:
            X_train_flat = X_train.reshape(X_train.shape[0], -1)
        else:
            X_train_flat = X_train
        
        # Créer et entraîner le modèle
        model = self.create_xgboost_model()
        
        print("  Entraînement en cours...")
        model.fit(X_train_flat, y_train)
        
        # Validation croisée
        cv_scores = cross_val_score(model, X_train_flat, y_train, 
                                   cv=5, scoring='neg_mean_absolute_error')
        
        print(f"  Scores de validation croisée (MAE): {-cv_scores.mean():.2f} ± {cv_scores.std():.2f}")
        
        # Sauvegarder
        self.models['xgboost'] = model
        
        print("✅ XGBoost entraîné")
        
        return model
    
    def train_random_forest(self, X_train: np.ndarray, y_train: np.ndarray) -> RandomForestRegressor:
        """
        Entraîne un modèle Random Forest.
        
        Args:
            X_train, y_train: Données d'entraînement
            
        Returns:
            Modèle Random Forest entraîné
        """
        print("\n" + "="*50)
        print("ENTRAÎNEMENT DU MODÈLE RANDOM FOREST")
        print("="*50)
        
        # Aplatir les données si nécessaire
        if len(X_train.shape) == 3:
            X_train_flat = X_train.reshape(X_train.shape[0], -1)
        else:
            X_train_flat = X_train
        
        # Créer et entraîner le modèle
        model = self.create_random_forest_model()
        
        print("  Entraînement en cours...")
        model.fit(X_train_flat, y_train)
        
        # Importance des caractéristiques
        feature_importance = model.feature_importances_
        top_features = np.argsort(feature_importance)[-5:][::-1]
        
        print("  Top 5 caractéristiques importantes:")
        for i, idx in enumerate(top_features, 1):
            print(f"    {i}. Caractéristique {idx}: {feature_importance[idx]:.4f}")
        
        # Sauvegarder
        self.models['random_forest'] = model
        
        print("✅ Random Forest entraîné")
        
        return model
    
    def train_ensemble(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict:
        """
        Entraîne un modèle ensemble combinant les prédictions.
        
        Args:
            X_train, y_train: Données d'entraînement
            
        Returns:
            Modèle ensemble
        """
        print("\n" + "="*50)
        print("ENTRAÎNEMENT DU MODÈLE ENSEMBLE")
        print("="*50)
        
        # Vérifier que tous les modèles sont entraînés
        required_models = ['lstm', 'xgboost', 'random_forest']
        for model_name in required_models:
            if model_name not in self.models:
                print(f"  ⚠️  Modèle {model_name} non entraîné. Entraînement en cours...")
                if model_name == 'lstm':
                    self.train_lstm(X_train, y_train)
                elif model_name == 'xgboost':
                    self.train_xgboost(X_train, y_train)
                elif model_name == 'random_forest':
                    self.train_random_forest(X_train, y_train)
        
        # Créer le modèle ensemble
        ensemble_model = {
            'type': 'ensemble',
            'models': self.models,
            'weights': {'lstm': 0.4, 'xgboost': 0.4, 'random_forest': 0.2}
        }
        
        self.models['ensemble'] = ensemble_model
        
        print("✅ Modèle ensemble créé")
        
        return ensemble_model
    
    def evaluate_model(self, model_name: str, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """
        Évalue un modèle spécifique.
        
        Args:
            model_name: Nom du modèle
            X_test, y_test: Données de test
            
        Returns:
            Métriques d'évaluation
        """
        if model_name not in self.models:
            raise ValueError(f"Modèle {model_name} non disponible")
        
        print(f"\n📊 Évaluation du modèle {model_name}...")
        
        model = self.models[model_name]
        
        # Préparer les données selon le type de modèle
        if model_name == 'lstm':
            y_pred = model.predict(X_test).flatten()
        elif model_name == 'ensemble':
            # Moyenne pondérée des prédictions
            predictions = []
            for submodel_name, submodel in model['models'].items():
                if submodel_name == 'lstm':
                    pred = submodel.predict(X_test).flatten()
                else:
                    # Aplatir pour les modèles non-séquentiels
                    if len(X_test.shape) == 3:
                        X_test_flat = X_test.reshape(X_test.shape[0], -1)
                    else:
                        X_test_flat = X_test
                    pred = submodel.predict(X_test_flat)
                predictions.append(pred)
            
            # Moyenne pondérée
            weights = model['weights']
            y_pred = sum(p * weights[name] for name, p in zip(model['models'].keys(), predictions))
        else:
            # Aplatir pour les modèles non-séquentiels
            if len(X_test.shape) == 3:
                X_test_flat = X_test.reshape(X_test.shape[0], -1)
            else:
                X_test_flat = X_test
            y_pred = model.predict(X_test_flat)
        
        # Calculer les métriques
        metrics = self._calculate_metrics(y_test, y_pred)
        
        print(f"  MAE: {metrics['mae']:.2f}")
        print(f"  RMSE: {metrics['rmse']:.2f}")
        print(f"  R²: {metrics['r2']:.4f}")
        
        return metrics
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """
        Calcule les métriques d'évaluation.
        
        Args:
            y_true: Valeurs réelles
            y_pred: Valeurs prédites
            
        Returns:
            Dict avec les métriques
        """
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        
        # Erreur relative moyenne
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100
        
        return {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'mape': mape,
            'predictions': y_pred.tolist()
        }
    
    def train_all_models(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict:
        """
        Entraîne tous les modèles.
        
        Args:
            X_train, y_train: Données d'entraînement
            
        Returns:
            Dict avec tous les modèles entraînés
        """
        print("\n" + "="*50)
        print("🚀 ENTRAÎNEMENT DE TOUS LES MODÈLES")
        print("="*50)
        
        # Split validation
        X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42
        )
        
        # Entraîner LSTM
        lstm_model, lstm_history = self.train_lstm(
            X_train_split, y_train_split, 
            X_val_split, y_val_split
        )
        
        # Entraîner XGBoost
        xgb_model = self.train_xgboost(X_train_split, y_train_split)
        
        # Entraîner Random Forest
        rf_model = self.train_random_forest(X_train_split, y_train_split)
        
        # Créer ensemble
        ensemble_model = self.train_ensemble(X_train_split, y_train_split)
        
        print("\n" + "="*50)
        print("✅ TOUS LES MODÈLES ENTRAÎNÉS")
        print("="*50)
        
        return {
            'lstm': lstm_model,
            'xgboost': xgb_model,
            'random_forest': rf_model,
            'ensemble': ensemble_model,
            'lstm_history': lstm_history
        }
    
    def save_models(self, path: str = "models"):
        """
        Sauvegarde tous les modèles.
        
        Args:
            path: Chemin de sauvegarde
        """
        import os
        os.makedirs(path, exist_ok=True)
        
        print(f"\n💾 Sauvegarde des modèles dans {path}/")
        
        for name, model in self.models.items():
            if name == 'lstm':
                model.save(f"{path}/{name}_model.h5")
            elif name == 'ensemble':
                # Sauvegarder les modèles individuels de l'ensemble
                for submodel_name, submodel in model['models'].items():
                    if submodel_name == 'lstm':
                        submodel.save(f"{path}/ensemble_{submodel_name}_model.h5")
                    else:
                        joblib.dump(submodel, f"{path}/ensemble_{submodel_name}_model.pkl")
                # Sauvegarder la configuration de l'ensemble
                with open(f"{path}/ensemble_config.pkl", 'wb') as f:
                    pickle.dump({'weights': model['weights']}, f)
            else:
                joblib.dump(model, f"{path}/{name}_model.pkl")
            
            print(f"  ✅ {name} sauvegardé")
    
    def load_models(self, path: str = "models"):
        """
        Charge les modèles sauvegardés.
        
        Args:
            path: Chemin des modèles
        """
        import os
        
        print(f"\n📂 Chargement des modèles depuis {path}/")
        
        # Charger LSTM
        lstm_path = f"{path}/lstm_model.h5"
        if os.path.exists(lstm_path):
            self.models['lstm'] = tf.keras.models.load_model(lstm_path)
            print("  ✅ LSTM chargé")
        
        # Charger XGBoost
        xgb_path = f"{path}/xgboost_model.pkl"
        if os.path.exists(xgb_path):
            self.models['xgboost'] = joblib.load(xgb_path)
            print("  ✅ XGBoost chargé")
        
        # Charger Random Forest
        rf_path = f"{path}/random_forest_model.pkl"
        if os.path.exists(rf_path):
            self.models['random_forest'] = joblib.load(rf_path)
            print("  ✅ Random Forest chargé")
        
        # Charger Ensemble
        ensemble_config_path = f"{path}/ensemble_config.pkl"
        if os.path.exists(ensemble_config_path):
            with open(ensemble_config_path, 'rb') as f:
                ensemble_config = pickle.load(f)
            
            # Charger les modèles de l'ensemble
            ensemble_models = {}
            for model_name in ['lstm', 'xgboost', 'random_forest']:
                if model_name == 'lstm':
                    model_path = f"{path}/ensemble_{model_name}_model.h5"
                    if os.path.exists(model_path):
                        ensemble_models[model_name] = tf.keras.models.load_model(model_path)
                else:
                    model_path = f"{path}/ensemble_{model_name}_model.pkl"
                    if os.path.exists(model_path):
                        ensemble_models[model_name] = joblib.load(model_path)
            
            if ensemble_models:
                self.models['ensemble'] = {
                    'type': 'ensemble',
                    'models': ensemble_models,
                    'weights': ensemble_config['weights']
                }
                print("  ✅ Ensemble chargé")
    
    def predict(self, X: np.ndarray, model_name: str = 'ensemble') -> np.ndarray:
        """
        Fait une prédiction avec le modèle spécifié.
        
        Args:
            X: Données d'entrée
            model_name: Nom du modèle
            
        Returns:
            Prédictions
        """
        if model_name not in self.models:
            raise ValueError(f"Modèle {model_name} non disponible")
        
        model = self.models[model_name]
        
        if model_name == 'lstm':
            return model.predict(X).flatten()
        elif model_name == 'ensemble':
            # Combinaison des prédictions
            predictions = []
            weights = []
            
            for submodel_name, submodel in model['models'].items():
                if submodel_name == 'lstm':
                    pred = submodel.predict(X).flatten()
                else:
                    # Aplatir pour les modèles non-séquentiels
                    if len(X.shape) == 3:
                        X_flat = X.reshape(X.shape[0], -1)
                    else:
                        X_flat = X
                    pred = submodel.predict(X_flat)
                
                predictions.append(pred)
                weights.append(model['weights'][submodel_name])
            
            # Moyenne pondérée
            weighted_sum = sum(p * w for p, w in zip(predictions, weights))
            return weighted_sum
        else:
            # Aplatir pour les modèles non-séquentiels
            if len(X.shape) == 3:
                X_flat = X.reshape(X.shape[0], -1)
            else:
                X_flat = X
            return model.predict(X_flat)
    
    def plot_training_history(self, model_name: str = 'lstm'):
        """
        Affiche l'historique d'entraînement d'un modèle.
        
        Args:
            model_name: Nom du modèle
        """
        if model_name not in self.histories:
            print(f"Historique non disponible pour {model_name}")
            return
        
        history = self.histories[model_name]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss
        ax1.plot(history['loss'], label='Entraînement')
        ax1.plot(history['val_loss'], label='Validation')
        ax1.set_title('Loss du modèle')
        ax1.set_xlabel('Époque')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # MAE
        ax2.plot(history['mae'], label='Entraînement')
        ax2.plot(history['val_mae'], label='Validation')
        ax2.set_title('MAE du modèle')
        ax2.set_xlabel('Époque')
        ax2.set_ylabel('MAE')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

def main():
    """Fonction principale pour tester le module."""
    print("="*50)
    print("TEST DU MODULE DE MODÈLES")
    print("="*50)
    
    # Générer des données de test
    np.random.seed(42)
    n_samples = 1000
    sequence_length = 50
    n_features = 21
    
    # Données LSTM (3D)
    X_lstm = np.random.randn(n_samples, sequence_length, n_features)
    y_lstm = np.random.uniform(10, 200, n_samples)
    
    # Données pour modèles classiques (2D)
    X_flat = X_lstm.reshape(n_samples, -1)
    
    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X_lstm, y_lstm, test_size=0.2, random_state=42
    )
    
    # Initialiser les modèles
    pm = PredictiveModels()
    
    # Entraîner tous les modèles
    models = pm.train_all_models(X_train, y_train)
    
    # Évaluer les modèles
    print("\n" + "="*50)
    print("ÉVALUATION DES MODÈLES")
    print("="*50)
    
    results = {}
    for model_name in ['lstm', 'xgboost', 'random_forest', 'ensemble']:
        metrics = pm.evaluate_model(model_name, X_test, y_test)
        results[model_name] = metrics
    
    # Afficher les résultats
    print("\n📋 RÉSULTATS FINAUX:")
    print("-" * 40)
    print(f"{'Modèle':<20} {'MAE':<10} {'RMSE':<10} {'R²':<10}")
    print("-" * 40)
    
    for model_name, metrics in results.items():
        print(f"{model_name:<20} {metrics['mae']:<10.2f} {metrics['rmse']:<10.2f} {metrics['r2']:<10.4f}")
    
    # Sauvegarder les modèles
    pm.save_models('test_models')
    
    return results

if __name__ == "__main__":
    results = main()
