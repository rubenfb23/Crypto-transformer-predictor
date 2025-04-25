import pytest
from src.main import main


def test_main_prints_startup_message(capsys):
    # When main() is called, it should print the startup message
    main()
    captured = capsys.readouterr()
    assert "Crypto-transformer-predictor started." in captured.out
