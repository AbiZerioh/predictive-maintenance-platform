"""
Utilitaires API pour la plateforme de maintenance prédictive.
Client pour interagir avec l'API FastAPI.
"""

import requests
import json
from typing import Dict, List, Optional, Any
from datetime import datetime
import time
import websockets
import asyncio

class APIClient:
    """
    Client pour l'API de maintenance prédictive.
    
    Attributes:
        base_url (str): URL de base de l'API
        timeout (int): Timeout des requêtes
        session (requests.Session): Session HTTP
    """
    
    def __init__(self, base_url: str = "http://localhost:8000", timeout: int = 30):
        """
        Initialise le client API.
        
        Args:
            base_url: URL de base de l'API
            timeout: Timeout des requêtes en secondes
        """
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.session = requests.Session()
        
        # Headers par défaut
        self.session.headers.update({
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        })
        
        print(f"✅ Client API initialisé: {self.base_url}")
    
    def health_check(self) -> Dict:
        """
        Vérifie l'état de l'API.
        
        Returns:
            Réponse de l'API
        """
        try:
            response = self.session.get(
                f"{self.base_url}/health",
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            return {
                "error": str(e),
                "status": "unhealthy",
                "timestamp": datetime.now().isoformat()
            }
    
    def predict_rul(self, features: List[float], 
                   equipment_id: str = "test_equipment",
                   model_type: str = "xgboost") -> Dict:
        """
        Envoie une requête de prédiction à l'API.
        
        Args:
            features: Liste des caractéristiques
            equipment_id: ID de l'équipement
            model_type: Type de modèle à utiliser
            
        Returns:
            Prédiction de l'API
        """
        payload = {
            "equipment_id": equipment_id,
            "features": features,
            "model_type": model_type
        }
        
        try:
            response = self.session.post(
                f"{self.base_url}/predict",
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            return {
                "error": str(e),
                "prediction": 0.0,
                "confidence": 0.0,
                "status": "error",
                "message": f"Erreur API: {e}"
            }
    
    def batch_predict(self, batch_data: List[Dict]) -> List[Dict]:
        """
        Envoie plusieurs prédictions en une seule requête.
        
        Args:
            batch_data: Liste de dictionnaires avec features
            
        Returns:
            Liste des prédictions
        """
        try:
            response = self.session.post(
                f"{self.base_url}/predict/batch",
                json=batch_data,
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            return [{
                "error": str(e),
                "prediction": 0.0,
                "status": "error"
            } for _ in batch_data]
    
    def get_models(self) -> List[Dict]:
        """
        Récupère la liste des modèles disponibles.
        
        Returns:
            Liste des modèles
        """
        try:
            response = self.session.get(
                f"{self.base_url}/models",
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            return [{"error": str(e), "name": "unknown", "status": "unavailable"}]
    
    def retrain_model(self, model_name: str) -> Dict:
        """
        Déclenche le réentraînement d'un modèle.
        
        Args:
            model_name: Nom du modèle
            
        Returns:
            Réponse de l'API
        """
        try:
            response = self.session.post(
                f"{self.base_url}/models/{model_name}/retrain",
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            return {"error": str(e), "status": "failed"}
    
    def get_metrics(self) -> Dict:
        """
        Récupère les métriques du système.
        
        Returns:
            Métriques système
        """
        try:
            response = self.session.get(
                f"{self.base_url}/monitor/metrics",
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            return {"error": str(e), "status": "unavailable"}
    
    def get_alerts(self, acknowledged: bool = False, 
                  severity: Optional[str] = None,
                  limit: int = 100) -> List[Dict]:
        """
        Récupère les alertes.
        
        Args:
            acknowledged: Filtre sur l'état d'acquittement
            severity: Filtre sur la sévérité
            limit: Nombre maximum d'alertes
            
        Returns:
            Liste des alertes
        """
        params = {
            "acknowledged": str(acknowledged).lower(),
            "limit": limit
        }
        
        if severity:
            params["severity"] = severity
        
        try:
            response = self.session.get(
                f"{self.base_url}/monitor/alerts",
                params=params,
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            return [{"error": str(e), "message": "Impossible de récupérer les alertes"}]
    
    def acknowledge_alert(self, alert_id: str) -> Dict:
        """
        Acquitte une alerte.
        
        Args:
            alert_id: ID de l'alerte
            
        Returns:
            Réponse de l'API
        """
        try:
            response = self.session.post(
                f"{self.base_url}/monitor/alerts/{alert_id}/acknowledge",
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            return {"error": str(e), "status": "failed"}
    
    async def connect_websocket(self, endpoint: str = "realtime"):
        """
        Se connecte au WebSocket pour les données temps réel.
        
        Args:
            endpoint: Endpoint WebSocket
            
        Returns:
            Connexion WebSocket
        """
        ws_url = self.base_url.replace("http", "ws") + f"/ws/{endpoint}"
        
        try:
            connection = await websockets.connect(ws_url)
            print(f"✅ Connecté au WebSocket: {ws_url}")
            return connection
        except Exception as e:
            print(f"❌ Erreur de connexion WebSocket: {e}")
            return None
    
    def wait_for_api(self, max_retries: int = 10, delay: int = 2) -> bool:
        """
        Attend que l'API soit disponible.
        
        Args:
            max_retries: Nombre maximum de tentatives
            delay: Délai entre les tentatives en secondes
            
        Returns:
            True si l'API est disponible, False sinon
        """
        print(f"⏳ Attente de l'API ({self.base_url})...")
        
        for attempt in range(max_retries):
            try:
                health = self.health_check()
                if "status" in health and health["status"] == "healthy":
                    print(f"✅ API disponible après {attempt + 1} tentatives")
                    return True
            except:
                pass
            
            if attempt < max_retries - 1:
                print(f"  Tentative {attempt + 1}/{max_retries} échouée, nouvelle tentative dans {delay}s...")
                time.sleep(delay)
        
        print(f"❌ API non disponible après {max_retries} tentatives")
        return False

class MockAPIClient(APIClient):
    """
    Client API mock pour le testing sans serveur réel.
    """
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        """Initialise le client mock."""
        super().__init__(base_url)
        print("⚠️  Utilisation du client API mock")
    
    def health_check(self) -> Dict:
        """Mock health check."""
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "version": "1.0.0",
            "models_loaded": 4
        }
    
    def predict_rul(self, features: List[float], 
                   equipment_id: str = "test_equipment",
                   model_type: str = "xgboost") -> Dict:
        """Mock prediction."""
        # Simulation basée sur la moyenne des features
        base_rul = 100.0
        feature_effect = sum(features) / len(features) * 50 if features else 0
        
        rul = base_rul + feature_effect + (random.random() * 20 - 10)  # Ajouter du bruit
        
        # Déterminer le statut
        if rul < 30:
            status = "critical"
            message = "Maintenance requise immédiatement"
        elif rul < 60:
            status = "warning"
            message = "Planifier la maintenance prochainement"
        else:
            status = "normal"
            message = "Aucune action requise"
        
        return {
            "prediction": round(rul, 2),
            "confidence": round(0.7 + random.random() * 0.25, 2),  # 0.7-0.95
            "status": status,
            "message": message,
            "model_used": model_type,
            "timestamp": datetime.now().isoformat()
        }
    
    def get_models(self) -> List[Dict]:
        """Mock models list."""
        return [
            {"name": "xgboost", "status": "available", "version": "1.0"},
            {"name": "lstm", "status": "available", "version": "1.0"},
            {"name": "random_forest", "status": "available", "version": "1.0"},
            {"name": "ensemble", "status": "available", "version": "1.0"}
        ]

def main():
    """Fonction principale pour tester le module."""
    print("="*50)
    print("TEST DU CLIENT API")
    print("="*50)
    
    # Utiliser le client mock pour le testing
    client = MockAPIClient()
    
    # Vérifier la santé
    print("\n🧪 Test health check:")
    health = client.health_check()
    print(f"  Statut: {health.get('status', 'unknown')}")
    print(f"  Version: {health.get('version', 'unknown')}")
    
    # Obtenir la liste des modèles
    print("\n🧪 Test liste des modèles:")
    models = client.get_models()
    for model in models:
        print(f"  - {model['name']}: {model['status']}")
    
    # Tester la prédiction
    print("\n🧪 Test prédiction:")
    features = [random.random() for _ in range(21)]  # 21 features comme le dataset NASA
    
    for model_type in ["xgboost", "lstm", "random_forest", "ensemble"]:
        prediction = client.predict_rul(features, model_type=model_type)
        print(f"  {model_type}: RUL={prediction['prediction']}, "
              f"Confiance={prediction['confidence']}, "
              f"Statut={prediction['status']}")
    
    # Tester le batch prediction
    print("\n🧪 Test batch prediction:")
    batch_data = [
        {"equipment_id": f"EQ_{i}", "features": [random.random() for _ in range(10)], 
         "model_type": "xgboost"}
        for i in range(3)
    ]
    
    batch_results = client.batch_predict(batch_data)
    for i, result in enumerate(batch_results):
        print(f"  Équipement {i}: RUL={result.get('prediction', 0):.1f}")
    
    print("\n✅ Tests terminés")
    
    return client

if __name__ == "__main__":
    import random  # Pour le mock
    client = main()
