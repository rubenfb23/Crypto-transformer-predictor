"""
Secure configuration loader for API secrets.
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


def get_api_secret_key() -> str:
    """Retrieve the API secret key from environment variables."""
    # Changed variable name
    secret_key = os.getenv("BLOCKCHAIN_API_SECRET_KEY")
    if not secret_key:
        # Updated variable name in error message
        raise EnvironmentError(
            "Environment variable BLOCKCHAIN_API_SECRET_KEY is not set."
        )
    return secret_key
