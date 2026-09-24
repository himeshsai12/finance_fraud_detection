from fraud_detection.config import Settings


def test_default_settings_are_deterministic() -> None:
    settings = Settings()

    assert settings.random_seed == 42
    assert settings.neighbor_cap == 50
    assert settings.kafka_transactions_topic == "transactions"


def test_environment_prefix_is_supported(monkeypatch) -> None:
    monkeypatch.setenv("FRAUD_NEIGHBOR_CAP", "12")

    assert Settings().neighbor_cap == 12


def test_paysim_download_metadata_can_be_configured() -> None:
    settings = Settings()

    assert settings.paysim_url is None
    assert settings.paysim_sha256 is None
