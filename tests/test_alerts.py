"""Tests for sentiment spike alert generation."""

from src.alerts import generate_alerts


def test_generate_global_alert():
    spikes = [
        {
            "timestamp": "2026-01-01 05:00:00",
            "observed_value": 10.0,
            "baseline_value": 2.0,
            "z_score": 2.5,
            "threshold": 2.0,
        }
    ]

    alerts = generate_alerts(spikes)

    assert len(alerts) == 1

    alert = alerts[0]

    assert alert["alert_type"] == "NEGATIVE_SENTIMENT_SPIKE"
    assert alert["severity"] == "MEDIUM"
    assert "dimension" not in alert
    assert "entity" not in alert


def test_generate_entity_alert():
    spikes = [
        {
            "timestamp": "2026-01-01 05:00:00",
            "observed_value": 10.0,
            "baseline_value": 1.0,
            "z_score": 5.0,
            "threshold": 2.0,
            "dimension": "brand",
            "entity": "Samsung",
        }
    ]

    alerts = generate_alerts(spikes)

    assert len(alerts) == 1

    alert = alerts[0]

    assert alert["alert_type"] == "NEGATIVE_SENTIMENT_SPIKE"
    assert alert["severity"] == "CRITICAL"
    assert alert["dimension"] == "brand"
    assert alert["entity"] == "Samsung"
    assert "Samsung" in alert["message"]
    assert "Brand" in alert["message"]


def test_entity_alert_for_product():
    spikes = [
        {
            "timestamp": "2026-01-01 05:00:00",
            "observed_value": 8.0,
            "baseline_value": 1.0,
            "z_score": 3.5,
            "threshold": 2.0,
            "dimension": "product",
            "entity": "Galaxy S25",
        }
    ]

    alerts = generate_alerts(spikes)

    assert len(alerts) == 1
    assert alerts[0]["severity"] == "HIGH"
    assert alerts[0]["dimension"] == "product"
    assert alerts[0]["entity"] == "Galaxy S25"
    assert "Galaxy S25" in alerts[0]["message"]


def test_entity_alert_for_topic():
    spikes = [
        {
            "timestamp": "2026-01-01 05:00:00",
            "observed_value": 7.0,
            "baseline_value": 1.0,
            "z_score": 2.5,
            "threshold": 2.0,
            "dimension": "topic",
            "entity": "battery",
        }
    ]

    alerts = generate_alerts(spikes)

    assert len(alerts) == 1
    assert alerts[0]["severity"] == "MEDIUM"
    assert alerts[0]["dimension"] == "topic"
    assert alerts[0]["entity"] == "battery"
    assert "battery" in alerts[0]["message"]