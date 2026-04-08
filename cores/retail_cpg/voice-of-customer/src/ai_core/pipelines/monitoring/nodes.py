import numpy as np
import polars as pl
import logging
from typing import Dict, Any, List
from scipy.stats import chisquare
import requests
import json
from datetime import datetime
import mlflow

logger = logging.getLogger(__name__)


def check_sentiment_drift(
    current_summary: Dict[str, Any],
    monitoring_params: Dict[str, Any]
) -> Dict[str, Any]:
    """Check for drift in sentiment distribution"""
    logger.info("Checking sentiment drift...")
    
    # In a real scenario, we would load historical baseline
    # For now, we'll assume a uniform distribution as baseline if not provided
    baseline = {
        'positive': 0.33,
        'neutral': 0.33,
        'negative': 0.33
    }
    
    current_dist = current_summary['sentiment_percentages']
    
    drift_detected = False
    max_drift = 0
    
    for sentiment, pct in current_dist.items():
        baseline_pct = baseline.get(sentiment, 0) * 100
        drift = abs(pct - baseline_pct)
        if drift > max_drift:
            max_drift = drift
            
    threshold = monitoring_params['sentiment_drift']['threshold'] * 100
    
    if max_drift > threshold:
        drift_detected = True
        logger.warning(f"Sentiment drift detected! Max drift: {max_drift:.2f}%")

    # Log to MLflow if run is active
    if mlflow.active_run():
        mlflow.log_metric("drift_max_sentiment_shift", max_drift)
        mlflow.log_metric("drift_detected", int(drift_detected))
        logger.info("Logged sentiment drift metrics to MLflow")

    return {
        'drift_detected': drift_detected,
        'max_drift': max_drift,
        'current_distribution': current_dist,
        'timestamp': datetime.now().isoformat()
    }


def check_topic_drift(
    current_summary: Dict[str, Any],
    monitoring_params: Dict[str, Any]
) -> Dict[str, Any]:
    """Check for new or shifting topics"""
    logger.info("Checking topic drift...")
    
    # Simplified check: just look for new topics or major shifts
    # Real implementation would compare with historical topic distributions
    
    current_topics = current_summary['topic_distribution']
    
    return {
        'drift_detected': False,  # Placeholder
        'new_topics_count': 0,
        'timestamp': datetime.now().isoformat()
    }


def generate_alerts(
    sentiment_drift: Dict[str, Any],
    topic_drift: Dict[str, Any],
    monitoring_params: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """Generate alerts based on drift checks"""
    logger.info("Generating alerts...")
    
    alerts = []
    
    if sentiment_drift['drift_detected']:
        alert = {
            'type': 'sentiment_drift',
            'severity': 'high',
            'message': f"Sentiment drift detected. Max deviation: {sentiment_drift['max_drift']:.2f}%",
            'timestamp': datetime.now().isoformat()
        }
        alerts.append(alert)
        send_alert(alert, monitoring_params['alerts'])
        
    if topic_drift['drift_detected']:
        alert = {
            'type': 'topic_drift',
            'severity': 'medium',
            'message': "Topic distribution has shifted significantly.",
            'timestamp': datetime.now().isoformat()
        }
        alerts.append(alert)
        send_alert(alert, monitoring_params['alerts'])
        
    return alerts


def send_alert(alert: Dict[str, Any], alert_params: Dict[str, Any]):
    """Send alert to configured channels"""
    channels = alert_params.get('channels', [])
    
    if 'slack' in channels and alert_params.get('slack_webhook'):
        try:
            webhook_url = alert_params['slack_webhook']
            payload = {'text': f"🚨 *{alert['type'].upper()}*\n{alert['message']}"}
            # requests.post(webhook_url, json=payload) # Uncomment to enable
            logger.info(f"Slack alert sent: {alert['message']}")
        except Exception as e:
            logger.error(f"Failed to send Slack alert: {e}")
            
    if 'email' in channels:
        # Placeholder for email sending logic
        logger.info(f"Email alert sent to {alert_params.get('email')}: {alert['message']}")

def detect_high_urgency_alerts(
    unified_df: pl.DataFrame
) -> List[Dict[str, Any]]:
    """Detect specific interactions that require immediate human intervention"""
    logger.info("Detecting high urgency alerts...")
    
    high_urgency = unified_df.filter(pl.col('urgency') == 'High')
    
    alerts = []
    for row in high_urgency.to_dicts()[:50]: # Cap at 50 for reporting
        alerts.append({
            'interaction_id': row.get('Interaction_ID') or row.get('review_id'),
            'customer_id': row.get('customer_id'),
            'topic': row.get('topic_name'),
            'emotion': row.get('detected_emotion'),
            'action': row.get('recommended_action'),
            'text_preview': str(row.get('review_text'))[:100] + "...",
            'timestamp': datetime.now().isoformat()
        })
    
    logger.info(f"Detected {len(alerts)} high urgency alerts")
    return alerts
