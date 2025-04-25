# Crypto-transformer-predictor

A simple application to predict cryptocurrency trends using transformer models.

## Project Structure

```
Crypto-transformer-predictor/
├── src/           # Application source code
│   ├── __init__.py
│   ├── main.py    # Entry point of the application
│   ├── data/      # Data ingestion modules
│   │   ├── ohlcv_data.py
│   │   ├── onchain_data.py
│   │   └── sentiment_data.py
│   ├── models/    # Model definitions
│   │   ├── lstm_model.py
│   │   └── transformer_model.py
│   ├── preprocessing/  # Data preparation
│   │   ├── data_sync.py
│   │   ├── dataset.py
│   │   └── feature_engineering.py
│   ├── training/  # Training and evaluation scripts
│   │   ├── train.py
│   │   └── evaluate.py
│   └── visualization/  # Dashboard interface
│       └── dashboard.py
├── tests/         # Unit tests
│   ├── test_data_sync.py
│   ├── test_dataset.py
│   ├── test_feature_engineering.py
│   ├── test_main.py
│   ├── test_ohlcv_data.py
│   ├── test_onchain_data.py
│   └── test_sentiment_data.py
├── docs/          # Documentation files
│   └── index.md
├── requirements.txt  # Python dependencies
├── LICENSE        # License information
└── .gitignore     # Files to ignore in git
```

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd Crypto-transformer-predictor
   ```
2. (Optional) Create and activate a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
3. Install dependencies (if any):
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the main application with a specific mode:

```bash
python -m src.main --mode <train|evaluate|dashboard>
```

Examples:
```bash
python -m src.main --mode train      # Train the model
python -m src.main --mode evaluate  # Evaluate the model
python -m src.main --mode dashboard # Start the interactive dashboard
```

## Running Tests

Execute tests with pytest:
```bash
pytest
```

## Documentation

Documentation is available in the `docs/` folder. Open `docs/index.md` for more information.