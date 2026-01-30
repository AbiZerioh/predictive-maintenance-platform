"""
Dashboard Streamlit pour la maintenance prédictive.
Interface utilisateur interactive pour la visualisation et les prédictions.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import requests
import json
import time

# Configuration de la page
st.set_page_config(
    page_title="Dashboard Maintenance Prédictive",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Titre principal
st.title("🏭 Plateforme de Maintenance Prédictive")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Sélection du modèle
    model_type = st.selectbox(
        "Modèle de prédiction",
        ["XGBoost", "LSTM", "Random Forest", "Ensemble"],
        index=0
    )
    
    # Simulation temps réel
    realtime_enabled = st.checkbox("Activer simulation temps réel", value=True)
    
    # Seuils d'alerte
    st.subheader("🔔 Seuils d'alerte")
    warning_threshold = st.slider("Seuil avertissement (cycles)", 30, 100, 60)
    critical_threshold = st.slider("Seuil critique (cycles)", 10, 50, 30)
    
    # Bouton de prédiction
    predict_button = st.button("🎯 Lancer la prédiction", type="primary")

# Fonction pour générer des données simulées
def generate_sensor_data():
    """Génère des données de capteur simulées."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Données de base avec variation aléatoire
    data = {
        "timestamp": timestamp,
        "equipment_id": f"MOTEUR_{np.random.randint(100, 999)}",
        "temperature": np.random.normal(85, 10),
        "pressure": np.random.normal(120, 15),
        "vibration": np.random.normal(2.5, 0.8),
        "current": np.random.normal(30, 5),
        "rul": np.random.uniform(20, 200)  # Cycles restants
    }
    
    # Déterminer le statut
    if data["rul"] < critical_threshold:
        data["status"] = "CRITIQUE"
        data["status_color"] = "red"
    elif data["rul"] < warning_threshold:
        data["status"] = "AVERTISSEMENT"
        data["status_color"] = "orange"
    else:
        data["status"] = "NORMAL"
        data["status_color"] = "green"
    
    return data

# Fonction pour faire une prédiction via l'API
def make_prediction(features, model_type="xgboost"):
    """Envoie une requête à l'API pour faire une prédiction."""
    try:
        response = requests.post(
            "http://localhost:8000/predict",
            json={
                "equipment_id": "test_equipment",
                "features": features,
                "model_type": model_type.lower().replace(" ", "_")
            },
            timeout=5
        )
        return response.json()
    except:
        # Simulation si l'API n'est pas disponible
        return {
            "prediction": np.random.uniform(50, 200),
            "confidence": np.random.uniform(0.7, 0.95),
            "status": "normal",
            "message": "Prédiction simulée (API non disponible)"
        }

# Layout principal
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        label="Équipements actifs",
        value="24",
        delta="+2"
    )

with col2:
    st.metric(
        label="Taux de disponibilité",
        value="96.5%",
        delta="+1.2%"
    )

with col3:
    st.metric(
        label="Alertes actives",
        value="3",
        delta="-1",
        delta_color="inverse"
    )

# Section des graphiques
st.markdown("## 📈 Visualisations temps réel")

# Créer des onglets
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Capteurs", 
    "🎯 Prédictions", 
    "📋 Historique", 
    "🚨 Alertes"
])

with tab1:
    # Graphique des capteurs
    st.subheader("Données des capteurs")
    
    # Générer des données historiques
    time_points = pd.date_range(end=datetime.now(), periods=50, freq='H')
    sensor_data = pd.DataFrame({
        'timestamp': time_points,
        'temperature': np.random.normal(85, 5, 50),
        'pressure': np.random.normal(120, 10, 50),
        'vibration': np.random.normal(2.5, 0.5, 50)
    })
    
    # Graphique Plotly
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=sensor_data['timestamp'],
        y=sensor_data['temperature'],
        name='Température (°C)',
        line=dict(color='red', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=sensor_data['timestamp'],
        y=sensor_data['pressure'],
        name='Pression (psi)',
        yaxis='y2',
        line=dict(color='blue', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=sensor_data['timestamp'],
        y=sensor_data['vibration'],
        name='Vibration (mm/s)',
        yaxis='y3',
        line=dict(color='green', width=2)
    ))
    
    fig.update_layout(
        title='Données des capteurs - Historique',
        xaxis=dict(title='Date/Heure'),
        yaxis=dict(title='Température (°C)', titlefont=dict(color='red')),
        yaxis2=dict(
            title='Pression (psi)',
            titlefont=dict(color='blue'),
            overlaying='y',
            side='right'
        ),
        yaxis3=dict(
            title='Vibration (mm/s)',
            titlefont=dict(color='green'),
            overlaying='y',
            side='right',
            position=0.95
        ),
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    # Interface de prédiction
    st.subheader("Prédiction de RUL")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Paramètres de simulation
        st.markdown("### Paramètres de simulation")
        
        equipment_type = st.selectbox(
            "Type d'équipement",
            ["Moteur Turbofan", "Pompe Centrifuge", "Compresseur", "Générateur"]
        )
        
        hours_operation = st.slider(
            "Heures d'opération",
            min_value=100,
            max_value=10000,
            value=5000,
            step=100
        )
        
        # Générer des features aléatoires basées sur les paramètres
        if st.button("Générer des features", type="secondary"):
            features = np.random.rand(21).tolist()  # 21 features comme le dataset NASA
            st.session_state['features'] = features
            st.success("Features générées !")
    
    with col2:
        # Prédiction
        st.markdown("### Résultat de prédiction")
        
        if 'features' not in st.session_state:
            st.session_state['features'] = np.random.rand(21).tolist()
        
        if predict_button:
            with st.spinner("Calcul de la prédiction..."):
                time.sleep(1)  # Simulation du temps de calcul
                result = make_prediction(
                    st.session_state['features'], 
                    model_type
                )
                
                # Afficher le résultat
                st.metric(
                    label="RUL Prédit",
                    value=f"{result['prediction']:.1f}",
                    help="Cycles restants avant maintenance"
                )
                
                # Indicateur de confiance
                st.progress(result['confidence'])
                st.caption(f"Confiance: {result['confidence']*100:.1f}%")
                
                # Statut
                if result['prediction'] < critical_threshold:
                    st.error("🚨 STATUT CRITIQUE - Maintenance requise immédiatement")
                elif result['prediction'] < warning_threshold:
                    st.warning("⚠️ STATUT AVERTISSEMENT - Planifier maintenance")
                else:
                    st.success("✅ STATUT NORMAL - Aucune action requise")
                
                # Message
                st.info(result['message'])

with tab3:
    # Historique des prédictions
    st.subheader("Historique des prédictions")
    
    # Données historiques simulées
    history_data = pd.DataFrame({
        'Date': pd.date_range(end=datetime.now(), periods=20, freq='D'),
        'Equipment': [f'EQ_{i}' for i in range(20)],
        'RUL_Predicted': np.random.uniform(20, 200, 20),
        'RUL_Actual': np.random.uniform(15, 210, 20),
        'Model': np.random.choice(['XGBoost', 'LSTM', 'RF'], 20),
        'Status': np.random.choice(['NORMAL', 'WARNING', 'CRITICAL'], 20)
    })
    
    # Calculer l'erreur
    history_data['Error'] = abs(history_data['RUL_Predicted'] - history_data['RUL_Actual'])
    
    # Afficher le tableau
    st.dataframe(
        history_data.sort_values('Date', ascending=False),
        use_container_width=True
    )
    
    # Graphique d'erreur
    fig = px.bar(
        history_data,
        x='Date',
        y='Error',
        color='Model',
        title='Erreur de prédiction par modèle'
    )
    st.plotly_chart(fig, use_container_width=True)

with tab4:
    # Gestion des alertes
    st.subheader("Alertes actives")
    
    # Données d'alerte simulées
    alerts = [
        {
            "id": 1,
            "equipment": "MOTEUR_123",
            "type": "Température élevée",
            "severity": "CRITIQUE",
            "timestamp": "2024-01-15 14:30",
            "value": "128°C",
            "threshold": "120°C"
        },
        {
            "id": 2,
            "equipment": "POMPE_456",
            "type": "Vibration excessive",
            "severity": "AVERTISSEMENT",
            "timestamp": "2024-01-15 13:45",
            "value": "3.8 mm/s",
            "threshold": "3.0 mm/s"
        },
        {
            "id": 3,
            "equipment": "COMP_789",
            "type": "RUL faible",
            "severity": "CRITIQUE",
            "timestamp": "2024-01-15 12:15",
            "value": "25 cycles",
            "threshold": "30 cycles"
        }
    ]
    
    # Afficher les alertes
    for alert in alerts:
        with st.container():
            col1, col2, col3 = st.columns([3, 2, 1])
            
            with col1:
                st.markdown(f"**{alert['equipment']}** - {alert['type']}")
                st.caption(f"📅 {alert['timestamp']}")
            
            with col2:
                if alert['severity'] == "CRITIQUE":
                    st.error(f"🔴 {alert['severity']}")
                else:
                    st.warning(f"🟡 {alert['severity']}")
                
                st.write(f"Valeur: {alert['value']} (Seuil: {alert['threshold']})")
            
            with col3:
                if st.button("Acquitter", key=f"ack_{alert['id']}"):
                    st.success("Alert acquittée")
                    time.sleep(1)
                    st.rerun()

# Section inférieure
st.markdown("---")
st.markdown("### 📊 Statistiques de performance")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Précision moyenne", "92.5%", "+2.1%")

with col2:
    st.metric("MAE", "16.8 cycles", "-1.2")

with col3:
    st.metric("RMSE", "22.3 cycles", "-0.8")

with col4:
    st.metric("Temps réponse", "125 ms", "-15 ms")

# Pied de page
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>🏭 Plateforme de Maintenance Prédictive v1.0.0</p>
        <p>📞 Contact: maintenance@entreprise.com | 📍 GitHub: ton-username</p>
    </div>
    """,
    unsafe_allow_html=True
)

# Simulation temps réel
if realtime_enabled:
    # Espace pour les mises à jour temps réel
    realtime_placeholder = st.empty()
    
    # Simuler des mises à jour (dans un vrai projet, utiliser WebSocket)
    if st.button("Actualiser données temps réel"):
        latest_data = generate_sensor_data()
        
        with realtime_placeholder.container():
            st.subheader("📡 Dernière lecture temps réel")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Équipement", latest_data["equipment_id"])
                st.metric("Température", f"{latest_data['temperature']:.1f}°C")
            
            with col2:
                st.metric("Pression", f"{latest_data['pressure']:.1f} psi")
                st.metric("Vibration", f"{latest_data['vibration']:.2f} mm/s")
            
            with col3:
                st.metric(
                    "RUL estimé",
                    f"{latest_data['rul']:.1f}",
                    help="Cycles restants"
                )
                
                # Indicateur de statut
                if latest_data["status"] == "CRITIQUE":
                    st.error("🚨 CRITIQUE")
                elif latest_data["status"] == "AVERTISSEMENT":
                    st.warning("⚠️ AVERTISSEMENT")
                else:
                    st.success("✅ NORMAL")
