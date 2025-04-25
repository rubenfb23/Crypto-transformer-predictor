# Getting Started

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

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the application with a specific mode:
```bash
python -m src.main --mode <train|evaluate|dashboard>
```

Examples:
```bash
python -m src.main --mode train      # Train the model
python -m src.main --mode evaluate   # Evaluate the model
python -m src.main --mode dashboard  # Start the interactive dashboard
```