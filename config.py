import os
from dotenv import load_dotenv

basedir = os.path.abspath(os.path.dirname(__file__))
load_dotenv(os.path.join(basedir, '.env'))

class Config:
    SECRET_KEY = os.getenv('SECRET_KEY') or 'you-will-never-guess'
    SQLALCHEMY_DATABASE_URI = os.getenv('SQLALCHEMY_DATABASE_URI') or \
        'sqlite:///' + os.path.join(basedir, 'instance/autoparts.db')
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # Flutterwave Configuration
    FLUTTERWAVE_PUBLIC_KEY = os.getenv('FLUTTERWAVE_PUBLIC_KEY')
    FLUTTERWAVE_SECRET_KEY = os.getenv('FLUTTERWAVE_SECRET_KEY')
    
    
    # Payment Settings
    SHIPPING_COST = 100  # RWF
    TAX_RATE = 0.08  # 8%
    
    # Environment
    DEBUG = os.getenv('FLASK_DEBUG', 'True').lower() == 'true' 