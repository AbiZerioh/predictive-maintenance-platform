"""
Utilitaires pour le dashboard de maintenance prédictive.
Fonctions de visualisation et d'analyse.
"""

import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta

class DashboardUtils:
    """
    Utilitaires pour créer des visualisations pour le dashboard.
    
    Attributes:
        color_palette (dict): Palette de couleurs pour les visualisations
        style_config (dict): Configuration du style des graphiques
    """
    
    def __init__(self, theme: str = "dark"):
        """
        Initialise les utilitaires du dashboard.
        
        Args:
            theme: Thème visuel ('dark' ou 'light')
        """
        self.theme = theme
        self.color_palette = self._get_color_palette()
        self.style_config = self._get_style_config()
        
        # Configuration matplotlib
        plt.style.use('seaborn-v0_8-darkgrid' if theme == "dark" else 'seaborn-v0_8-whitegrid')
        
        print(f"✅ Utilitaires dashboard initialisés (thème: {theme})")
    
    def _get_color_palette(self) -> Dict:
        """Retourne la palette de couleurs selon le thème."""
        if self.theme == "dark":
            return {
                "primary": "#3498db",      # Bleu
                "secondary": "#2ecc71",    # Vert
                "warning": "#f39c12",      # Orange
                "danger": "#e74c3c",       # Rouge
                "dark": "#2c3e50",         # Bleu foncé
                "light": "#ecf0f1",        # Gris clair
                "success": "#27ae60",      # Vert foncé
                "info": "#17a2b8",         # Cyan
                "background": "#121212",   # Fond sombre
                "paper": "#1e1e1e",        # Surface
                "text": "#ffffff"          # Texte blanc
            }
        else:
            return {
                "primary": "#1f77b4",      # Bleu
                "secondary": "#2ca02c",    # Vert
                "warning": "#ff7f0e",      # Orange
                "danger": "#d62728",       # Rouge
                "dark": "#7f7f7f",         # Gris
                "light": "#ffffff",        # Blanc
                "success": "#2ca02c",      # Vert
                "info": "#17becf",         # Cyan
                "background": "#ffffff",    # Fond clair
                "paper": "#f5f5f5",        # Surface
                "text": "#000000"          # Texte noir
            }
    
    def _get_style_config(self) -> Dict:
        """Retourne la configuration du style."""
        return {
            "font_family": "Arial, sans-serif",
            "title_font_size": 18,
            "axis_font_size": 12,
            "legend_font_size": 10,
            "grid_alpha": 0.1,
            "line_width": 2,
            "marker_size": 6
        }
    
    def create_sensor_timeseries(self, sensor_data: pd.DataFrame, 
                                sensor_names: List[str] = None) -> go.Figure:
        """
        Crée un graphique temporel des données des capteurs.
        
        Args:
            sensor_data: DataFrame avec colonnes timestamp et données des capteurs
            sensor_names: Liste des noms des capteurs à afficher
            
        Returns:
            Figure Plotly
        """
        if sensor_names is None:
            # Prendre toutes les colonnes numériques sauf timestamp
            numeric_cols = sensor_data.select_dtypes(include=[np.number]).columns.tolist()
            if 'timestamp' in sensor_data.columns:
                sensor_names = [col for col in numeric_cols if col != 'timestamp']
            else:
                sensor_names = numeric_cols[:5]  # Limiter à 5 capteurs
        
        fig = go.Figure()
        
        for sensor in sensor_names[:5]:  # Limiter à 5 capteurs pour la lisibilité
            if sensor in sensor_data.columns:
                fig.add_trace(go.Scatter(
                    x=sensor_data.index if 'timestamp' not in sensor_data.columns else sensor_data['timestamp'],
                    y=sensor_data[sensor],
                    mode='lines',
                    name=sensor,
                    line=dict(
                        width=self.style_config['line_width'],
                        color=self.color_palette['primary']
                    ),
                    hovertemplate='%{x}<br>%{y:.2f}<extra></extra>'
                ))
        
        # Mise en forme
        fig.update_layout(
            title="Données des capteurs - Évolution temporelle",
            xaxis_title="Temps",
            yaxis_title="Valeur",
            template="plotly_dark" if self.theme == "dark" else "plotly_white",
            hovermode='x unified',
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            ),
            margin=dict(l=50, r=50, t=50, b=50)
        )
        
        return fig
    
    def create_rul_prediction_chart(self, actual_rul: np.ndarray, 
                                   predicted_rul: np.ndarray,
                                   equipment_names: List[str] = None) -> go.Figure:
        """
        Crée un graphique de comparaison RUL réel vs prédit.
        
        Args:
            actual_rul: Valeurs RUL réelles
            predicted_rul: Valeurs RUL prédites
            equipment_names: Noms des équipements
            
        Returns:
            Figure Plotly
        """
        if equipment_names is None:
            equipment_names = [f"Équipement {i+1}" for i in range(len(actual_rul))]
        
        # Calculer l'erreur
        errors = predicted_rul - actual_rul
        mae = np.mean(np.abs(errors))
        rmse = np.sqrt(np.mean(errors**2))
        
        fig = go.Figure()
        
        # Nuage de points RUL réel vs prédit
        fig.add_trace(go.Scatter(
            x=actual_rul,
            y=predicted_rul,
            mode='markers',
            name='Prédictions',
            marker=dict(
                size=self.style_config['marker_size'],
                color=self.color_palette['primary'],
                opacity=0.7
            ),
            text=equipment_names,
            hovertemplate='%{text}<br>Réel: %{x:.1f}<br>Prédit: %{y:.1f}<extra></extra>'
        ))
        
        # Ligne parfaite (y = x)
        max_val = max(actual_rul.max(), predicted_rul.max())
        min_val = min(actual_rul.min(), predicted_rul.min())
        
        fig.add_trace(go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            name='Prédiction parfaite',
            line=dict(
                color=self.color_palette['danger'],
                width=1,
                dash='dash'
            )
        ))
        
        # Mise en forme
        fig.update_layout(
            title=f"RUL Réel vs Prédit (MAE: {mae:.1f}, RMSE: {rmse:.1f})",
            xaxis_title="RUL Réel (cycles)",
            yaxis_title="RUL Prédit (cycles)",
            template="plotly_dark" if self.theme == "dark" else "plotly_white",
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )
        
        return fig
    
    def create_equipment_status_chart(self, status_data: List[Dict]) -> go.Figure:
        """
        Crée un graphique de statut des équipements.
        
        Args:
            status_data: Liste des statuts des équipements
            
        Returns:
            Figure Plotly
        """
        # Compter les statuts
        status_counts = {}
        for equipment in status_data:
            status = equipment.get('status', 'unknown')
            status_counts[status] = status_counts.get(status, 0) + 1
        
        # Couleurs selon le statut
        status_colors = {
            'normal': self.color_palette['success'],
            'warning': self.color_palette['warning'],
            'critical': self.color_palette['danger'],
            'unknown': self.color_palette['dark']
        }
        
        fig = go.Figure(data=[go.Pie(
            labels=list(status_counts.keys()),
            values=list(status_counts.values()),
            hole=0.3,
            marker=dict(colors=[status_colors.get(s, self.color_palette['dark']) 
                              for s in status_counts.keys()]),
            hovertemplate='%{label}: %{value} équipements<br>%{percent}<extra></extra>'
        )])
        
        # Mise en forme
        fig.update_layout(
            title="Statut des équipements",
            template="plotly_dark" if self.theme == "dark" else "plotly_white",
            annotations=[dict(
                text=f"Total: {sum(status_counts.values())}",
                x=0.5, y=0.5,
                font_size=14,
                showarrow=False
            )]
        )
        
        return fig
    
    def create_alert_timeline(self, alerts: List[Dict]) -> go.Figure:
        """
        Crée une timeline des alertes.
        
        Args:
            alerts: Liste des alertes
            
        Returns:
            Figure Plotly
        """
        if not alerts:
            # Créer une figure vide
            fig = go.Figure()
            fig.update_layout(
                title="Aucune alerte récente",
                template="plotly_dark" if self.theme == "dark" else "plotly_white",
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                annotations=[dict(
                    text="Aucune alerte à afficher",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5,
                    showarrow=False,
                    font=dict(size=16)
                )]
            )
            return fig
        
        # Préparer les données
        alert_types = {}
        for alert in alerts:
            alert_type = alert.get('type', 'unknown')
            if alert_type not in alert_types:
                alert_types[alert_type] = []
            alert_types[alert_type].append(alert)
        
        fig = go.Figure()
        
        # Couleurs pour les types d'alerte
        type_colors = {
            'critical': self.color_palette['danger'],
            'warning': self.color_palette['warning'],
            'info': self.color_palette['primary'],
            'maintenance': self.color_palette['secondary'],
            'unknown': self.color_palette['dark']
        }
        
        y_pos = 0
        for alert_type, type_alerts in alert_types.items():
            for alert in type_alerts[:10]:  # Limiter à 10 par type
                timestamp = alert.get('timestamp', datetime.now().isoformat())
                
                fig.add_trace(go.Scatter(
                    x=[timestamp, timestamp],
                    y=[y_pos - 0.2, y_pos + 0.2],
                    mode='lines',
                    name=alert_type,
                    line=dict(
                        color=type_colors.get(alert_type, self.color_palette['dark']),
                        width=4
                    ),
                    showlegend=False,
                    hovertemplate=f"<b>{alert_type.upper()}</b><br>"
                                f"Équipement: {alert.get('equipment_id', 'N/A')}<br>"
                                f"Message: {alert.get('message', 'N/A')}<br>"
                                f"Heure: {timestamp}<extra></extra>"
                ))
                
                y_pos += 1
        
        # Mise en forme
        fig.update_layout(
            title="Timeline des alertes récentes",
            xaxis_title="Temps",
            yaxis=dict(
                title="Alertes",
                tickvals=[],
                showticklabels=False
            ),
            template="plotly_dark" if self.theme == "dark" else "plotly_white",
            height=400,
            hovermode='closest'
        )
        
        return fig
    
    def create_performance_metrics(self, metrics_history: pd.DataFrame) -> go.Figure:
        """
        Crée un graphique des métriques de performance.
        
        Args:
            metrics_history: DataFrame avec l'historique des métriques
            
        Returns:
            Figure Plotly
        """
        fig = go.Figure()
        
        # Métriques à afficher
        metric_cols = ['mae', 'rmse', 'r2']
        metric_names = {'mae': 'MAE', 'rmse': 'RMSE', 'r2': 'R²'}
        metric_colors = {
            'mae': self.color_palette['primary'],
            'rmse': self.color_palette['secondary'],
            'r2': self.color_palette['success']
        }
        
        for metric in metric_cols:
            if metric in metrics_history.columns:
                fig.add_trace(go.Scatter(
                    x=metrics_history.index,
                    y=metrics_history[metric],
                    mode='lines+markers',
                    name=metric_names[metric],
                    line=dict(
                        color=metric_colors[metric],
                        width=self.style_config['line_width']
                    ),
                    marker=dict(
                        size=self.style_config['marker_size']
                    ),
                    hovertemplate=f"{metric_names[metric]}: %{{y:.3f}}<br>Date: %{{x}}<extra></extra>"
                ))
        
        # Mise en forme
        fig.update_layout(
            title="Évolution des métriques de performance",
            xaxis_title="Date",
            yaxis_title="Valeur",
            template="plotly_dark" if self.theme == "dark" else "plotly_white",
            hovermode='x unified',
            showlegend=True
        )
        
        return fig
    
    def create_correlation_heatmap(self, sensor_data: pd.DataFrame) -> go.Figure:
        """
        Crée une heatmap de corrélation entre les capteurs.
        
        Args:
            sensor_data: DataFrame avec les données des capteurs
            
        Returns:
            Figure Plotly
        """
        # Sélectionner uniquement les colonnes de capteurs
        sensor_cols = [col for col in sensor_data.columns if 'sensor' in col]
        if len(sensor_cols) > 10:
            sensor_cols = sensor_cols[:10]  # Limiter à 10 capteurs
        
        sensor_subset = sensor_data[sensor_cols]
        
        # Calculer la matrice de corrélation
        corr_matrix = sensor_subset.corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.index,
            colorscale='RdBu' if self.theme == "light" else 'Viridis',
            zmin=-1,
            zmax=1,
            text=np.round(corr_matrix.values, 2),
            texttemplate='%{text}',
            textfont={"size": 10},
            hoverongaps=False
        ))
        
        # Mise en forme
        fig.update_layout(
            title="Matrice de corrélation entre capteurs",
            template="plotly_dark" if self.theme == "dark" else "plotly_white",
            xaxis_title="Capteurs",
            yaxis_title="Capteurs",
            height=500,
            width=600
        )
        
        return fig
    
    def create_maintenance_schedule(self, schedule_data: List[Dict]) -> go.Figure:
        """
        Crée un diagramme de Gantt pour le planning de maintenance.
        
        Args:
            schedule_data: Données du planning de maintenance
            
        Returns:
            Figure Plotly
        """
        if not schedule_data:
            # Créer une figure vide
            fig = go.Figure()
            fig.update_layout(
                title="Aucun planning de maintenance",
                template="plotly_dark" if self.theme == "dark" else "plotly_white",
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                annotations=[dict(
                    text="Aucun planning disponible",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5,
                    showarrow=False,
                    font=dict(size=16)
                )]
            )
            return fig
        
        # Préparer les données pour le diagramme de Gantt
        tasks = []
        for task in schedule_data:
            tasks.append(dict(
                Task=task.get('equipment_id', 'Unknown'),
                Start=task.get('start_date', datetime.now().isoformat()),
                Finish=task.get('end_date', (datetime.now() + timedelta(hours=2)).isoformat()),
                Type=task.get('type', 'maintenance'),
                Duration=task.get('duration_hours', 2),
                Status=task.get('status', 'planned')
            ))
        
        df = pd.DataFrame(tasks)
        
        # Couleurs selon le type
        type_colors = {
            'preventive': self.color_palette['primary'],
            'corrective': self.color_palette['danger'],
            'predictive': self.color_palette['secondary'],
            'maintenance': self.color_palette['warning']
        }
        
        fig = px.timeline(
            df, 
            x_start="Start", 
            x_end="Finish", 
            y="Task",
            color="Type",
            color_discrete_map=type_colors,
            hover_data=["Duration", "Status"]
        )
        
        # Mise en forme
        fig.update_layout(
            title="Planning de maintenance",
            xaxis_title="Date/Heure",
            yaxis_title="Équipement",
            template="plotly_dark" if self.theme == "dark" else "plotly_white",
            height=400,
            showlegend=True
        )
        
        return fig
    
    def create_statistics_summary(self, data: pd.DataFrame) -> Dict:
        """
        Calcule des statistiques sommaires pour l'affichage.
        
        Args:
            data: DataFrame avec les données
            
        Returns:
            Dict avec les statistiques
        """
        if data.empty:
            return {"error": "Aucune donnée disponible"}
        
        summary = {
            "general": {
                "total_samples": len(data),
                "total_equipment": data['unit_id'].nunique() if 'unit_id' in data.columns else 0,
                "date_range": {
                    "min": data['timestamp'].min() if 'timestamp' in data.columns else "N/A",
                    "max": data['timestamp'].max() if 'timestamp' in data.columns else "N/A"
                }
            },
            "sensors": {}
        }
        
        # Statistiques par capteur
        sensor_cols = [col for col in data.columns if 'sensor' in col]
        for sensor in sensor_cols[:5]:  # Limiter aux 5 premiers
            if sensor in data.columns:
                sensor_data = data[sensor].dropna()
                if len(sensor_data) > 0:
                    summary["sensors"][sensor] = {
                        "mean": float(sensor_data.mean()),
                        "std": float(sensor_data.std()),
                        "min": float(sensor_data.min()),
                        "max": float(sensor_data.max()),
                        "median": float(sensor_data.median())
                    }
        
        # Statistiques RUL si disponible
        if 'RUL' in data.columns:
            rul_data = data['RUL'].dropna()
            if len(rul_data) > 0:
                summary["rul"] = {
                    "mean": float(rul_data.mean()),
                    "std": float(rul_data.std()),
                    "min": float(rul_data.min()),
                    "max": float(rul_data.max()),
                    "critical_count": int((rul_data < 30).sum()),
                    "warning_count": int(((rul_data >= 30) & (rul_data < 60)).sum())
                }
        
        return summary

def main():
    """Fonction principale pour tester le module."""
    print("="*50)
    print("TEST DES UTILITAIRES DASHBOARD")
    print("="*50)
    
    # Initialiser les utilitaires
    utils = DashboardUtils(theme="dark")
    
    # Créer des données de test
    np.random.seed(42)
    n_samples = 100
    
    # Données de capteurs
    timestamps = pd.date_range(start='2024-01-01', periods=n_samples, freq='H')
    sensor_data = pd.DataFrame({
        'timestamp': timestamps,
        'sensor_1': np.random.normal(100, 10, n_samples),
        'sensor_2': np.random.normal(50, 5, n_samples),
        'sensor_3': np.random.normal(20, 3, n_samples),
        'sensor_4': np.random.normal(80, 8, n_samples),
        'sensor_5': np.random.normal(30, 4, n_samples)
    })
    
    # Données RUL
    actual_rul = np.random.uniform(10, 200, 50)
    predicted_rul = actual_rul + np.random.normal(0, 10, 50)
    
    # Données de statut
    status_data = [
        {"equipment_id": f"EQ_{i}", "status": np.random.choice(['normal', 'warning', 'critical'])}
        for i in range(20)
    ]
    
    # Données d'alerte
    alerts = [
        {
            "id": i,
            "equipment_id": f"EQ_{np.random.randint(1, 10)}",
            "type": np.random.choice(['critical', 'warning', 'info']),
            "timestamp": (datetime.now() - timedelta(hours=i)).isoformat(),
            "message": f"Alerte test {i}"
        }
        for i in range(10)
    ]
    
    # Données de performance
    dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
    metrics_history = pd.DataFrame({
        'date': dates,
        'mae': np.random.uniform(10, 20, 30),
        'rmse': np.random.uniform(15, 25, 30),
        'r2': np.random.uniform(0.8, 0.95, 30)
    }).set_index('date')
    
    # Tester les visualisations
    print("\n🧪 Test des visualisations:")
    
    # 1. Graphique temporel des capteurs
    fig1 = utils.create_sensor_timeseries(sensor_data)
    print("  ✅ Graphique temporel des capteurs créé")
    
    # 2. Graphique RUL
    fig2 = utils.create_rul_prediction_chart(actual_rul, predicted_rul)
    print("  ✅ Graphique RUL créé")
    
    # 3. Graphique de statut
    fig3 = utils.create_equipment_status_chart(status_data)
    print("  ✅ Graphique de statut créé")
    
    # 4. Timeline des alertes
    fig4 = utils.create_alert_timeline(alerts)
    print("  ✅ Timeline des alertes créée")
    
    # 5. Métriques de performance
    fig5 = utils.create_performance_metrics(metrics_history)
    print("  ✅ Graphique des métriques créé")
    
    # 6. Heatmap de corrélation
    fig6 = utils.create_correlation_heatmap(sensor_data)
    print("  ✅ Heatmap de corrélation créée")
    
    # 7. Statistiques sommaires
    summary = utils.create_statistics_summary(sensor_data)
    print("\n📋 Statistiques sommaires:")
    print(f"  Total échantillons: {summary['general']['total_samples']}")
    print(f"  Capteurs analysés: {len(summary['sensors'])}")
    
    print("\n✅ Tous les tests terminés")
    
    return utils

if __name__ == "__main__":
    utils = main()
