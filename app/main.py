"""
Application FastAPI principale pour la plateforme de maintenance prédictive.
"""

from fastapi import FastAPI, HTTPException, Depends, status, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
import numpy as np
import json
import asyncio
import uuid

# Import des modules locaux
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models import PredictiveModels
from src.real_time import RealTimeSimulator
from src.api import APIClient

# Modèles Pydantic pour validation
class PredictionRequest(BaseModel):
    """Requête de prédiction."""
    equipment_id: str = Field(..., description="ID de l'équipement")
    features: List[float] = Field(..., min_items=1, max_items=100, 
                                 description="Caractéristiques de l'équipement")
    model_type: str = Field("xgboost", description="Type de modèle à utiliser")

class PredictionResponse(BaseModel):
    """Réponse de prédiction."""
    prediction_id: str
    equipment_id: str
    prediction: float = Field(..., ge=0, description="RUL prédit en cycles")
    confidence: float = Field(..., ge=0, le=1, description="Confiance de la prédiction")
    status: str = Field(..., description="Statut de l'équipement")
    message: str = Field(..., description="Message descriptif")
    model_used: str
    timestamp: datetime

class EquipmentStatus(BaseModel):
    """Statut d'un équipement."""
    equipment_id: str
    name: str
    type: str
    status: str
    last_update: datetime
    temperature: float
    pressure: float
    vibration: float
    rul: float

class Alert(BaseModel):
    """Alerte de maintenance."""
    id: str
    equipment_id: str
    type: str
    severity: str
    message: str
    timestamp: datetime
    acknowledged: bool = False

# Initialisation de l'application FastAPI
app = FastAPI(
    title="API de Maintenance Prédictive",
    description="API pour la prédiction de pannes et l'optimisation de la maintenance",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configuration CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # À restreindre en production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Variables globales
models_manager = PredictiveModels()
simulator = RealTimeSimulator()
connected_clients = []
alerts_db = []

# Routes API
@app.get("/")
async def root():
    """Route racine."""
    return {
        "message": "API de Maintenance Prédictive",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health")
async def health_check():
    """Vérification de l'état de l'API."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "models_loaded": len(models_manager.models),
        "connected_clients": len(connected_clients),
        "alerts_active": len([a for a in alerts_db if not a['acknowledged']])
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_rul(request: PredictionRequest):
    """
    Prédit le RUL (Remaining Useful Life) d'un équipement.
    
    Args:
        request: Requête de prédiction
        
    Returns:
        Prédiction du RUL
    """
    try:
        # Vérifier le type de modèle
        if request.model_type not in models_manager.models:
            if request.model_type == "ensemble" and 'ensemble' not in models_manager.models:
                # Créer un modèle ensemble si demandé
                features_array = np.array(request.features).reshape(1, -1)
                if len(features_array.shape) == 2 and features_array.shape[1] > 50:
                    # Reshape pour LSTM si nécessaire
                    seq_length = 50
                    n_features = features_array.shape[1] // seq_length
                    features_array = features_array.reshape(1, seq_length, n_features)
                
                models_manager.train_ensemble(features_array, np.array([100]))
            
            if request.model_type not in models_manager.models:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Modèle {request.model_type} non disponible"
                )
        
        # Préparer les features
        features_array = np.array(request.features)
        
        # Reshape selon le type de modèle
        if request.model_type == "lstm":
            # Pour LSTM, besoin de forme 3D (batch, timesteps, features)
            seq_length = 50
            n_features = len(request.features) // seq_length
            if len(request.features) != seq_length * n_features:
                # Padding si nécessaire
                padding = seq_length * n_features - len(request.features)
                features_array = np.pad(features_array, (0, padding), 'constant')
            features_array = features_array.reshape(1, seq_length, n_features)
        
        # Faire la prédiction
        prediction = models_manager.predict(features_array, request.model_type)
        rul = float(prediction[0] if isinstance(prediction, np.ndarray) else prediction)
        
        # Calculer la confiance (simulée pour l'exemple)
        confidence = min(0.95, max(0.7, 1 - abs(rul - 100) / 200))
        
        # Déterminer le statut
        if rul < 30:
            status_val = "critical"
            message = "Maintenance requise immédiatement"
        elif rul < 60:
            status_val = "warning"
            message = "Planifier la maintenance prochainement"
        else:
            status_val = "normal"
            message = "Aucune action requise"
        
        # Générer un ID de prédiction
        prediction_id = str(uuid.uuid4())
        
        # Créer la réponse
        response = PredictionResponse(
            prediction_id=prediction_id,
            equipment_id=request.equipment_id,
            prediction=rul,
            confidence=confidence,
            status=status_val,
            message=message,
            model_used=request.model_type,
            timestamp=datetime.now()
        )
        
        # Vérifier si une alerte doit être créée
        if rul < 30:
            alert = {
                "id": str(uuid.uuid4()),
                "equipment_id": request.equipment_id,
                "type": "critical_rul",
                "severity": "critical",
                "message": f"RUL critique: {rul:.1f} cycles restants",
                "timestamp": datetime.now().isoformat(),
                "acknowledged": False
            }
            alerts_db.append(alert)
            
            # Notifier les clients WebSocket
            await broadcast_alert(alert)
        
        return response
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erreur lors de la prédiction: {str(e)}"
        )

@app.post("/predict/batch")
async def batch_predict(requests: List[PredictionRequest]):
    """
    Prédictions par lots pour plusieurs équipements.
    
    Args:
        requests: Liste de requêtes de prédiction
        
    Returns:
        Liste de prédictions
    """
    responses = []
    
    for request in requests:
        try:
            # Utiliser la route de prédiction unique
            response = await predict_rul(request)
            responses.append(response.dict())
        except Exception as e:
            responses.append({
                "error": str(e),
                "equipment_id": request.equipment_id,
                "prediction": 0.0,
                "status": "error"
            })
    
    return responses

@app.get("/models")
async def list_models():
    """
    Liste les modèles disponibles.
    
    Returns:
        Liste des modèles avec leurs métadonnées
    """
    models_info = []
    
    for name, model in models_manager.models.items():
        if name == "ensemble":
            model_info = {
                "name": name,
                "type": "ensemble",
                "submodels": list(model['models'].keys()),
                "weights": model['weights']
            }
        else:
            model_info = {
                "name": name,
                "type": type(model).__name__,
                "parameters": getattr(model, 'n_estimators', getattr(model, 'count_params', lambda: 0)())
            }
        
        models_info.append(model_info)
    
    return models_info

@app.post("/models/{model_name}/retrain")
async def retrain_model(model_name: str):
    """
    Déclenche le réentraînement d'un modèle.
    
    Args:
        model_name: Nom du modèle à réentraîner
        
    Returns:
        Confirmation du réentraînement
    """
    # Note: Dans une application réelle, ceci lancerait un job d'entraînement asynchrone
    
    return {
        "status": "training_scheduled",
        "model": model_name,
        "job_id": str(uuid.uuid4()),
        "estimated_completion": (datetime.now().timestamp() + 3600),  # 1 heure
        "message": "Réentraînement planifié"
    }

@app.get("/monitor/metrics")
async def get_system_metrics():
    """
    Récupère les métriques du système.
    
    Returns:
        Métriques système
    """
    # Métriques simulées pour l'exemple
    return {
        "system": {
            "uptime": "24h",
            "memory_usage": "45%",
            "cpu_usage": "32%"
        },
        "predictions": {
            "total": 1500,
            "today": 42,
            "average_response_time": "125ms"
        },
        "models": {
            "loaded": len(models_manager.models),
            "last_training": "2024-01-15T10:30:00"
        },
        "alerts": {
            "active": len([a for a in alerts_db if not a['acknowledged']]),
            "critical": len([a for a in alerts_db if a['severity'] == 'critical']),
            "today": 5
        }
    }

@app.get("/monitor/alerts")
async def get_alerts(acknowledged: bool = False, severity: Optional[str] = None, limit: int = 100):
    """
    Récupère les alertes.
    
    Args:
        acknowledged: Filtre sur l'état d'acquittement
        severity: Filtre sur la sévérité
        limit: Nombre maximum d'alertes à retourner
        
    Returns:
        Liste des alertes
    """
    filtered_alerts = alerts_db.copy()
    
    # Filtrer par acquittement
    filtered_alerts = [a for a in filtered_alerts if a['acknowledged'] == acknowledged]
    
    # Filtrer par sévérité
    if severity:
        filtered_alerts = [a for a in filtered_alerts if a['severity'] == severity]
    
    # Limiter le nombre
    filtered_alerts = filtered_alerts[:limit]
    
    return filtered_alerts

@app.post("/monitor/alerts/{alert_id}/acknowledge")
async def acknowledge_alert(alert_id: str):
    """
    Acquitte une alerte.
    
    Args:
        alert_id: ID de l'alerte
        
    Returns:
        Confirmation de l'acquittement
    """
    for alert in alerts_db:
        if alert['id'] == alert_id:
            alert['acknowledged'] = True
            alert['acknowledged_at'] = datetime.now().isoformat()
            return {"status": "success", "alert_id": alert_id}
    
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Alerte {alert_id} non trouvée"
    )

@app.get("/equipment/status")
async def get_equipment_status():
    """
    Récupère le statut de tous les équipements.
    
    Returns:
        Liste des statuts des équipements
    """
    status_list = simulator.get_equipment_status()
    
    # Convertir en modèles Pydantic
    equipment_status = []
    for status in status_list:
        equipment_status.append(EquipmentStatus(
            equipment_id=status['id'],
            name=status['name'],
            type=status['type'],
            status=status['status'],
            last_update=datetime.fromisoformat(status['last_update']),
            temperature=status['temperature'],
            pressure=status['pressure'],
            vibration=status.get('vibration', 0.0),
            rul=status['rul']
        ))
    
    return equipment_status

# WebSocket pour données temps réel
@app.websocket("/ws/realtime")
async def websocket_realtime(websocket: WebSocket):
    """
    WebSocket pour les données temps réel.
    
    Args:
        websocket: Connexion WebSocket
    """
    await websocket.accept()
    connected_clients.append(websocket)
    
    try:
        # Envoyer des données périodiquement
        while True:
            # Générer des données simulées
            data = simulator.generate_sensor_data()
            
            # Ajouter des informations supplémentaires
            data['type'] = 'sensor_data'
            data['server_timestamp'] = datetime.now().isoformat()
            
            # Envoyer au client
            await websocket.send_json(data)
            
            # Attendre avant le prochain envoi
            await asyncio.sleep(1)
            
    except WebSocketDisconnect:
        connected_clients.remove(websocket)
    except Exception as e:
        print(f"Erreur WebSocket: {e}")
        connected_clients.remove(websocket)

@app.websocket("/ws/alerts")
async def websocket_alerts(websocket: WebSocket):
    """
    WebSocket pour les alertes en temps réel.
    
    Args:
        websocket: Connexion WebSocket
    """
    await websocket.accept()
    
    try:
        # Envoyer les alertes existantes
        for alert in alerts_db[-10:]:  # 10 dernières alertes
            if not alert['acknowledged']:
                await websocket.send_json({
                    'type': 'alert',
                    'data': alert,
                    'timestamp': datetime.now().isoformat()
                })
        
        # Maintenir la connexion ouverte
        while True:
            await asyncio.sleep(1)
            # Vérifier les nouvelles alertes serait fait par un système d'événements
            # Pour l'exemple, on reste simplement connecté
            
    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"Erreur WebSocket alertes: {e}")

async def broadcast_alert(alert: Dict):
    """
    Diffuse une alerte à tous les clients WebSocket connectés.
    
    Args:
        alert: Données de l'alerte
    """
    message = {
        'type': 'alert',
        'data': alert,
        'timestamp': datetime.now().isoformat()
    }
    
    disconnected_clients = []
    
    for client in connected_clients:
        try:
            await client.send_json(message)
        except:
            disconnected_clients.append(client)
    
    # Retirer les clients déconnectés
    for client in disconnected_clients:
        connected_clients.remove(client)

# Événements de démarrage et d'arrêt
@app.on_event("startup")
async def startup_event():
    """Événement de démarrage de l'application."""
    print("🚀 Démarrage de l'API de maintenance prédictive...")
    
    # Charger les modèles
    try:
        models_manager.load_models("models")
        print(f"✅ Modèles chargés: {list(models_manager.models.keys())}")
    except Exception as e:
        print(f"⚠️  Erreur lors du chargement des modèles: {e}")
        print("   Les modèles seront créés à la première prédiction.")
    
    # Initialiser le simulateur
    print(f"✅ Simulateur initialisé: {len(simulator.equipment_list)} équipements")
    
    # Charger quelques alertes de test
    alerts_db.extend([
        {
            "id": str(uuid.uuid4()),
            "equipment_id": "EQ_001",
            "type": "temperature_high",
            "severity": "warning",
            "message": "Température élevée: 118°C",
            "timestamp": (datetime.now()).isoformat(),
            "acknowledged": False
        },
        {
            "id": str(uuid.uuid4()),
            "equipment_id": "EQ_003",
            "type": "vibration_high",
            "severity": "critical",
            "message": "Vibration excessive: 4.2 mm/s",
            "timestamp": (datetime.now()).isoformat(),
            "acknowledged": True
        }
    ])
    
    print("✅ API prête")

@app.on_event("shutdown")
async def shutdown_event():
    """Événement d'arrêt de l'application."""
    print("🛑 Arrêt de l'API...")
    
    # Fermer toutes les connexions WebSocket
    for client in connected_clients:
        try:
            await client.close()
        except:
            pass
    
    print("✅ API arrêtée")

# Route pour servir la documentation
@app.get("/documentation")
async def documentation():
    """Page de documentation HTML."""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Documentation API - Maintenance Prédictive</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 40px;
                background-color: #f5f5f5;
            }
            .container {
                max-width: 1200px;
                margin: 0 auto;
                background-color: white;
                padding: 30px;
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }
            h1 {
                color: #333;
                border-bottom: 2px solid #3498db;
                padding-bottom: 10px;
            }
            .endpoint {
                background-color: #f8f9fa;
                border-left: 4px solid #3498db;
                padding: 15px;
                margin: 20px 0;
            }
            .method {
                display: inline-block;
                padding: 5px 10px;
                border-radius: 4px;
                color: white;
                font-weight: bold;
                margin-right: 10px;
            }
            .get { background-color: #28a745; }
            .post { background-color: #007bff; }
            .put { background-color: #ffc107; color: #333; }
            .delete { background-color: #dc3545; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🏭 API de Maintenance Prédictive</h1>
            
            <h2>Endpoints disponibles</h2>
            
            <div class="endpoint">
                <span class="method get">GET</span>
                <strong>/health</strong>
                <p>Vérifie l'état de l'API</p>
            </div>
            
            <div class="endpoint">
                <span class="method post">POST</span>
                <strong>/predict</strong>
                <p>Prédit le RUL d'un équipement</p>
                <p><em>Body:</em> {"equipment_id": "string", "features": [float], "model_type": "string"}</p>
            </div>
            
            <div class="endpoint">
                <span class="method get">GET</span>
                <strong>/models</strong>
                <p>Liste les modèles disponibles</p>
            </div>
            
            <div class="endpoint">
                <span class="method get">GET</span>
                <strong>/monitor/alerts</strong>
                <p>Récupère les alertes</p>
            </div>
            
            <div class="endpoint">
                <span class="method get">GET</span>
                <strong>/equipment/status</strong>
                <p>Statut de tous les équipements</p>
            </div>
            
            <h2>WebSockets</h2>
            <ul>
                <li><strong>ws://localhost:8000/ws/realtime</strong> - Données temps réel</li>
                <li><strong>ws://localhost:8000/ws/alerts</strong> - Alertes temps réel</li>
            </ul>
            
            <h2>Documentation interactive</h2>
            <p>Pour la documentation interactive avec Swagger UI, visitez: <a href="/docs">/docs</a></p>
            
            <p><em>Version: 1.0.0</em></p>
        </div>
    </body>
    </html>
    """
    
    return HTMLResponse(content=html_content)

# Point d'entrée pour l'exécution directe
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
