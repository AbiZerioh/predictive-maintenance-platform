"""
Module de simulation de données temps réel pour la maintenance prédictive.
Génère des données de capteurs simulées pour le testing.
"""

import asyncio
import json
import random
import time
from datetime import datetime
from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd

class RealTimeSimulator:
    """
    Simulateur de données temps réel pour les capteurs industriels.
    
    Attributes:
        equipment_list (list): Liste des équipements simulés
        sensor_history (dict): Historique des lectures de capteurs
        is_running (bool): État du simulateur
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialise le simulateur temps réel.
        
        Args:
            config: Configuration du simulateur
        """
        self.config = config or {}
        self.equipment_list = []
        self.sensor_history = {}
        self.is_running = False
        
        # Initialiser les équipements
        self._initialize_equipment()
        
        print("✅ Simulateur temps réel initialisé")
    
    def _initialize_equipment(self):
        """Initialise la liste des équipements simulés."""
        equipment_types = [
            {"type": "turbofan", "base_temp": 85, "base_pressure": 120},
            {"type": "centrifugal_pump", "base_temp": 65, "base_pressure": 80},
            {"type": "screw_compressor", "base_temp": 75, "base_pressure": 150},
            {"type": "generator", "base_temp": 70, "base_pressure": 100},
        ]
        
        for i in range(10):
            eq_type = random.choice(equipment_types)
            equipment = {
                "id": f"EQ_{i+1:03d}",
                "type": eq_type["type"],
                "name": f"{eq_type['type'].replace('_', ' ').title()} {i+1}",
                "base_temp": eq_type["base_temp"],
                "base_pressure": eq_type["base_pressure"],
                "health": 1.0,  # 1.0 = parfait, 0.0 = défaillant
                "hours_operation": random.randint(100, 10000),
                "status": "normal"
            }
            self.equipment_list.append(equipment)
        
        print(f"  {len(self.equipment_list)} équipements initialisés")
    
    def generate_sensor_data(self, equipment_id: str = None) -> Dict[str, Any]:
        """
        Génère des données de capteur simulées.
        
        Args:
            equipment_id: ID de l'équipement (si None, aléatoire)
            
        Returns:
            Dict avec les données des capteurs
        """
        if not equipment_id:
            equipment = random.choice(self.equipment_list)
        else:
            equipment = next((eq for eq in self.equipment_list if eq["id"] == equipment_id), 
                            self.equipment_list[0])
        
        # Calculer la dégradation basée sur les heures d'opération
        degradation = min(1.0, equipment["hours_operation"] / 20000)
        
        # Mettre à jour la santé
        health_decrease = random.uniform(0.0001, 0.001)
        equipment["health"] = max(0.0, equipment["health"] - health_decrease)
        
        # Générer les lectures des capteurs avec tendance et bruit
        base_temp = equipment["base_temp"]
        temp_increase = degradation * 40  # Augmentation max de 40°C
        temperature = base_temp + temp_increase + random.uniform(-5, 5)
        
        base_pressure = equipment["base_pressure"]
        pressure_variation = degradation * 30
        pressure = base_pressure + pressure_variation + random.uniform(-10, 10)
        
        # Vibration augmente avec la dégradation
        vibration = 0.5 + (degradation * 3.5) + random.uniform(-0.2, 0.2)
        
        # Courant électrique
        current = 25 + (degradation * 15) + random.uniform(-3, 3)
        
        # Calculer RUL approximatif
        rul = max(10, 200 - (equipment["hours_operation"] / 50))
        
        # Ajuster RUL basé sur la santé
        rul *= equipment["health"]
        
        # Déterminer le statut
        if temperature > 120 or vibration > 4.0 or equipment["health"] < 0.3:
            status = "critical"
            color = "red"
        elif temperature > 100 or vibration > 2.5 or equipment["health"] < 0.7:
            status = "warning"
            color = "orange"
        else:
            status = "normal"
            color = "green"
        
        equipment["status"] = status
        
        # Créer la réponse
        sensor_data = {
            "equipment_id": equipment["id"],
            "equipment_name": equipment["name"],
            "timestamp": datetime.now().isoformat(),
            "sensors": {
                "temperature": round(temperature, 2),
                "pressure": round(pressure, 2),
                "vibration": round(vibration, 3),
                "current": round(current, 2),
                "health": round(equipment["health"], 3)
            },
            "metadata": {
                "hours_operation": equipment["hours_operation"],
                "degradation": round(degradation, 3),
                "status": status,
                "status_color": color
            },
            "prediction": {
                "rul": round(rul, 1),
                "confidence": random.uniform(0.7, 0.95)
            }
        }
        
        # Stocker dans l'historique
        if equipment["id"] not in self.sensor_history:
            self.sensor_history[equipment["id"]] = []
        
        self.sensor_history[equipment["id"]].append(sensor_data)
        
        # Limiter l'historique
        if len(self.sensor_history[equipment["id"]]) > 100:
            self.sensor_history[equipment["id"]] = self.sensor_history[equipment["id"]][-100:]
        
        return sensor_data
    
    def generate_batch_data(self, n_samples: int = 10) -> List[Dict]:
        """
        Génère un lot de données de capteurs.
        
        Args:
            n_samples: Nombre d'échantillons
            
        Returns:
            Liste de données de capteurs
        """
        batch_data = []
        
        for _ in range(n_samples):
            equipment_id = random.choice(self.equipment_list)["id"]
            data = self.generate_sensor_data(equipment_id)
            batch_data.append(data)
        
        return batch_data
    
    async def start_streaming(self, callback = None, interval: float = 1.0):
        """
        Démarre le streaming de données temps réel.
        
        Args:
            callback: Fonction à appeler avec chaque nouvel échantillon
            interval: Intervalle entre les échantillons en secondes
        """
        print(f"📡 Démarrage du streaming temps réel (intervalle: {interval}s)")
        self.is_running = True
        
        try:
            while self.is_running:
                # Générer des données
                data = self.generate_sensor_data()
                
                # Appeler le callback si fourni
                if callback:
                    try:
                        callback(data)
                    except Exception as e:
                        print(f"Erreur dans le callback: {e}")
                
                # Attendre avant le prochain échantillon
                await asyncio.sleep(interval)
        
        except asyncio.CancelledError:
            print("Streaming annulé")
        finally:
            self.is_running = False
    
    def stop_streaming(self):
        """Arrête le streaming de données."""
        self.is_running = False
        print("⏹️  Streaming arrêté")
    
    def get_equipment_status(self) -> List[Dict]:
        """
        Retourne le statut de tous les équipements.
        
        Returns:
            Liste des statuts des équipements
        """
        status_list = []
        
        for equipment in self.equipment_list:
            # Générer les dernières données
            latest_data = self.generate_sensor_data(equipment["id"])
            
            status_list.append({
                "id": equipment["id"],
                "name": equipment["name"],
                "type": equipment["type"],
                "hours_operation": equipment["hours_operation"],
                "health": equipment["health"],
                "status": equipment["status"],
                "last_update": latest_data["timestamp"],
                "temperature": latest_data["sensors"]["temperature"],
                "pressure": latest_data["sensors"]["pressure"],
                "rul": latest_data["prediction"]["rul"]
            })
        
        return status_list
    
    def get_sensor_history(self, equipment_id: str, n_points: int = 50) -> List[Dict]:
        """
        Retourne l'historique des capteurs pour un équipement.
        
        Args:
            equipment_id: ID de l'équipement
            n_points: Nombre de points à retourner
            
        Returns:
            Historique des capteurs
        """
        if equipment_id not in self.sensor_history:
            return []
        
        history = self.sensor_history[equipment_id]
        return history[-n_points:] if n_points > 0 else history
    
    def reset_equipment(self, equipment_id: str):
        """
        Réinitialise un équipement (pour le testing).
        
        Args:
            equipment_id: ID de l'équipement
        """
        for equipment in self.equipment_list:
            if equipment["id"] == equipment_id:
                equipment["health"] = 1.0
                equipment["hours_operation"] = random.randint(100, 1000)
                equipment["status"] = "normal"
                print(f"✅ Équipement {equipment_id} réinitialisé")
                return
        
        print(f"⚠️  Équipement {equipment_id} non trouvé")
    
    def simulate_failure(self, equipment_id: str):
        """
        Simule une défaillance sur un équipement.
        
        Args:
            equipment_id: ID de l'équipement
        """
        for equipment in self.equipment_list:
            if equipment["id"] == equipment_id:
                equipment["health"] = 0.1
                equipment["status"] = "critical"
                print(f"⚠️  Défaillance simulée sur {equipment_id}")
                return
        
        print(f"⚠️  Équipement {equipment_id} non trouvé")

def print_sensor_data(data: Dict):
    """
    Fonction d'exemple pour afficher les données de capteurs.
    
    Args:
        data: Données des capteurs
    """
    print(f"\n📊 Données capteurs - {data['timestamp']}")
    print(f"Équipement: {data['equipment_name']} ({data['equipment_id']})")
    print(f"Température: {data['sensors']['temperature']}°C")
    print(f"Pression: {data['sensors']['pressure']} psi")
    print(f"Vibration: {data['sensors']['vibration']} mm/s")
    print(f"RUL: {data['prediction']['rul']} cycles")
    print(f"Statut: {data['metadata']['status'].upper()}")

async def main():
    """Fonction principale pour tester le module."""
    print("="*50)
    print("TEST DU SIMULATEUR TEMPS RÉEL")
    print("="*50)
    
    # Initialiser le simulateur
    simulator = RealTimeSimulator()
    
    # Tester la génération de données
    print("\n🧪 Test de génération de données:")
    
    # Générer un échantillon
    sample_data = simulator.generate_sensor_data()
    print_sensor_data(sample_data)
    
    # Générer un lot de données
    print("\n📦 Génération d'un lot de données:")
    batch_data = simulator.generate_batch_data(n_samples=3)
    for i, data in enumerate(batch_data, 1):
        print(f"{i}. {data['equipment_id']}: {data['sensors']['temperature']}°C, "
              f"RUL: {data['prediction']['rul']}")
    
    # Obtenir le statut des équipements
    print("\n📋 Statut des équipements:")
    status_list = simulator.get_equipment_status()
    for status in status_list[:3]:  # Afficher seulement 3
        print(f"{status['id']}: {status['status']}, "
              f"Health: {status['health']:.3f}, "
              f"RUL: {status['rul']}")
    
    # Tester le streaming (court)
    print("\n📡 Test du streaming (5 secondes):")
    
    async def streaming_callback(data):
        """Callback pour le streaming."""
        print(f"  → {data['equipment_id']}: {data['sensors']['temperature']}°C, "
              f"Statut: {data['metadata']['status']}")
    
    # Démarrer le streaming pendant 5 secondes
    streaming_task = asyncio.create_task(
        simulator.start_streaming(callback=streaming_callback, interval=0.5)
    )
    
    await asyncio.sleep(5)
    simulator.stop_streaming()
    
    # Attendre que la tâche se termine
    try:
        await streaming_task
    except asyncio.CancelledError:
        pass
    
    print("\n✅ Test terminé")
    
    return simulator

if __name__ == "__main__":
    # Pour tester sans asyncio
    simulator = RealTimeSimulator()
    
    print("Test simple:")
    for _ in range(3):
        data = simulator.generate_sensor_data()
        print(f"{data['equipment_id']}: {data['sensors']['temperature']}°C, "
              f"RUL: {data['prediction']['rul']}")
