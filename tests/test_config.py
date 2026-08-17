from src.config import APP_NAME, ENVIRONMENT, DEBUG


def test_configuration():
    assert APP_NAME == "AI Business Analytics Assistant"
    assert ENVIRONMENT == "development"
    assert DEBUG is True