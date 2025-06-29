import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'your-secret-key-here-change-in-production'
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or 'sqlite:///autoparts.db'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # Flutterwave Configuration
    FLUTTERWAVE_PUBLIC_KEY = os.environ.get('FLWPUBK-0c4347ddda625a4f63c27b43005889bc-X') or 'FLWPUBK_TEST-1234567890abcdef'
    FLUTTERWAVE_SECRET_KEY = os.environ.get('FLWSECK-ef7734afb2dca1415a831364155f13e4-19140fe2419vt-X') or 'FLWSECK_TEST-1234567890abcdef'
    # FLUTTERWAVE_WEBHOOK_SECRET = os.environ.get('FLUTTERWAVE_WEBHOOK_SECRET') or 'test_webhook_secret'
    
    # MoMoPay Configuration (for future integration)
    MOMOPAY_API_KEY = os.environ.get('MOMOPAY_API_KEY') or 'test_momopay_key'
    MOMOPAY_SECRET_KEY = os.environ.get('MOMOPAY_SECRET_KEY') or 'test_momopay_secret'
    
    # Payment Settings
    SHIPPING_COST = 10000  # RWF
    TAX_RATE = 0.08  # 8%
    
    # Environment
    DEBUG = os.environ.get('FLASK_DEBUG', 'True').lower() == 'true' 