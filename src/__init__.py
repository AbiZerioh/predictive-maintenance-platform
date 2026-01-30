"""
Package principal de la plateforme de maintenance prédictive.
"""

__version__ = "1.0.0"
__author__ = "Votre Nom"
__email__ = "votre.email@example.com"

from src.data_loader import DataLoader
from src.preprocessing import DataPreprocessor
from src.models import PredictiveModels
from src.real_time import RealTimeSimulator
from src.api import APIClient
from src.dashboard import DashboardUtils

__all__ = [
    "DataLoader",
    "DataPreprocessor", 
    "PredictiveModels",
    "RealTimeSimulator",
    "APIClient",
    "DashboardUtils"
]
