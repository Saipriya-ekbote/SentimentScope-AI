from src.alerts import generate_alerts


def test_generate_global_alert():
    spikes = [
        {
            "timestamp": "2026-01-01 10:00:00",
            "observed_value": 10,
            "baseline_value": 5,
            "z_score": 3.5,
            "threshold": 2.0,
        }
    ]

    alerts = generate_alerts(spikes)

    assert len(alerts) == 1
    assert alerts[0]["alert_type"] == "NEGATIVE_SENTIMENT_SPIKE"
    assert alerts[0]["severity"] == "HIGH"
    assert alerts[0]["observed_value"] == 10.0
    assert alerts[0]["baseline_value"] == 5.0


def test_generate_entity_alert():
    spikes = [
        {
            "timestamp": "2026-01-01 10:00:00",
            "observed_value": 20,
            "baseline_value": 10,
            "z_score": 3.5,
            "threshold": 2.0,
            "dimension": "brand",
            "entity": "Apple",
        }
    ]

    alerts = generate_alerts(spikes)

    assert len(alerts) == 1
    assert alerts[0]["dimension"] == "brand"
    assert alerts[0]["entity"] == "Apple"
    assert "Apple" in alerts[0]["message"]


def test_entity_alert_for_product():
    spikes = [
        {
            "timestamp": "2026-01-01 10:00:00",
            "observed_value": 20,
            "baseline_value": 10,
            "z_score": 3.5,
            "threshold": 2.0,
            "dimension": "product",
            "entity": "iPhone",
        }
    ]

    alerts = generate_alerts(spikes)

    assert alerts[0]["dimension"] == "product"
    assert alerts[0]["entity"] == "iPhone"
    assert "Product" in alerts[0]["message"]


def test_entity_alert_for_topic():
    spikes = [
        {
            "timestamp": "2026-01-01 10:00:00",
            "observed_value": 20,
            "baseline_value": 10,
            "z_score": 3.5,
            "threshold": 2.0,
            "dimension": "topic",
            "entity": "Delivery",
        }
    ]

    alerts = generate_alerts(spikes)

    assert alerts[0]["dimension"] == "topic"
    assert alerts[0]["entity"] == "Delivery"
    assert "Topic" in alerts[0]["message"]


def test_alert_contains_priority():
    spikes = [
        {
            "timestamp": "2026-01-01 10:00:00",
            "observed_value": 20,
            "baseline_value": 10,
            "z_score": 4.5,
            "threshold": 2.0,
        }
    ]

    alerts = generate_alerts(spikes)

    assert alerts[0]["severity"] == "CRITICAL"
    assert alerts[0]["priority"] == 1


def test_alert_calculates_increase_percent():
    spikes = [
        {
            "timestamp": "2026-01-01 10:00:00",
            "observed_value": 25,
            "baseline_value": 10,
            "z_score": 3.5,
            "threshold": 2.0,
        }
    ]

    alerts = generate_alerts(spikes)

    assert alerts[0]["increase_percent"] == 150.0


def test_alert_handles_zero_baseline():
    spikes = [
        {
            "timestamp": "2026-01-01 10:00:00",
            "observed_value": 10,
            "baseline_value": 0,
            "z_score": 5.0,
            "threshold": 2.0,
        }
    ]

    alerts = generate_alerts(spikes)

    assert alerts[0]["increase_percent"] == 0.0


def test_alerts_are_sorted_by_priority():
    spikes = [
        {
            "timestamp": "2026-01-01 12:00:00",
            "observed_value": 20,
            "baseline_value": 10,
            "z_score": 2.5,
            "threshold": 2.0,
        },
        {
            "timestamp": "2026-01-01 10:00:00",
            "observed_value": 50,
            "baseline_value": 10,
            "z_score": 4.5,
            "threshold": 2.0,
        },
        {
            "timestamp": "2026-01-01 11:00:00",
            "observed_value": 30,
            "baseline_value": 10,
            "z_score": 3.5,
            "threshold": 2.0,
        },
    ]

    alerts = generate_alerts(spikes)

    assert alerts[0]["severity"] == "CRITICAL"
    assert alerts[1]["severity"] == "HIGH"
    assert alerts[2]["severity"] == "MEDIUM"


def test_duplicate_alerts_are_removed():
    spike = {
        "timestamp": "2026-01-01 10:00:00",
        "observed_value": 20,
        "baseline_value": 10,
        "z_score": 3.5,
        "threshold": 2.0,
        "dimension": "brand",
        "entity": "Apple",
    }

    alerts = generate_alerts([spike, spike.copy()])

    assert len(alerts) == 1
    assert alerts[0]["entity"] == "Apple"