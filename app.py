from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify, send_file
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import os
import re
from datetime import datetime, timedelta
from functools import wraps
from sqlalchemy.orm import joinedload
from io import BytesIO
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

# ML imports for recommendations
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import NMF
from collections import defaultdict
import pickle
import json
import random

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here-change-in-production'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///autoparts.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Password validation function
def validate_password(password):
    """
    Validates password strength with the following requirements:
    - At least 8 characters long
    - Contains at least one uppercase letter
    - Contains at least one lowercase letter
    - Contains at least one digit
    - Contains at least one special character
    """
    if len(password) < 8:
        return False, "Password must be at least 8 characters long"
    
    if not re.search(r'[A-Z]', password):
        return False, "Password must contain at least one uppercase letter"
    
    if not re.search(r'[a-z]', password):
        return False, "Password must contain at least one lowercase letter"
    
    if not re.search(r'\d', password):
        return False, "Password must contain at least one digit"
    
    if not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
        return False, "Password must contain at least one special character (!@#$%^&*(),.?\":{}|<>)"
    
    return True, "Password is strong"

# Database Models
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)
    first_name = db.Column(db.String(50))
    last_name = db.Column(db.String(50))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    is_admin = db.Column(db.Boolean, default=False)
    orders = db.relationship('Order', backref='user', lazy=True)

class Product(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    description = db.Column(db.Text)
    price = db.Column(db.Float, nullable=False)
    category = db.Column(db.String(50))
    image_url = db.Column(db.String(200))
    stock = db.Column(db.Integer, default=0)
    is_featured = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Tire-specific fields
    tire_width = db.Column(db.String(10))
    tire_aspect_ratio = db.Column(db.String(10))
    tire_rim_size = db.Column(db.String(10))
    tire_brand = db.Column(db.String(50))

class Order(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    total_amount = db.Column(db.Float, nullable=False)
    status = db.Column(db.String(20), default='pending')
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    items = db.relationship('OrderItem', backref='order', lazy=True)

class OrderItem(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    order_id = db.Column(db.Integer, db.ForeignKey('order.id'), nullable=False)
    product_id = db.Column(db.Integer, db.ForeignKey('product.id'), nullable=False)
    quantity = db.Column(db.Integer, nullable=False)
    price = db.Column(db.Float, nullable=False)
    
    # Relationships
    product = db.relationship('Product')

class Wishlist(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    product_id = db.Column(db.Integer, db.ForeignKey('product.id'), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationships
    user = db.relationship('User', backref='wishlist_items')
    product = db.relationship('Product')

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# ============================================================================
# MACHINE LEARNING RECOMMENDATION SYSTEM
# ============================================================================

class RecommendationSystem:
    def __init__(self):
        self.tfidf_vectorizer = None
        self.product_similarity_matrix = None
        self.user_product_matrix = None
        self.nmf_model = None
        self.user_factors = None
        self.product_factors = None
        
    def build_content_based_recommendations(self):
        """Build content-based recommendations using TF-IDF and cosine similarity"""
        # Get all products
        products = Product.query.all()
        
        if len(products) < 2:
            return
        
        # Create product descriptions for TF-IDF
        product_descriptions = []
        product_ids = []
        
        for product in products:
            # Combine name, description, and category for better representation
            text = f"{product.name} {product.description} {product.category}"
            product_descriptions.append(text)
            product_ids.append(product.id)
        
        # Create TF-IDF vectors
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(product_descriptions)
        
        # Calculate cosine similarity
        self.product_similarity_matrix = cosine_similarity(tfidf_matrix)
        
        # Save the product IDs mapping
        self.product_ids = product_ids
        
    def build_collaborative_filtering(self):
        """Build collaborative filtering recommendations using NMF"""
        # Get user purchase history
        orders = Order.query.filter_by(status='completed').all()
        
        if len(orders) < 5:  # Need minimum data for collaborative filtering
            return
        
        # Create user-product matrix
        user_product_data = []
        
        for order in orders:
            for item in order.items:
                user_product_data.append({
                    'user_id': order.user_id,
                    'product_id': item.product_id,
                    'rating': min(item.quantity * 2, 5)  # Convert quantity to rating (1-5)
                })
        
        if len(user_product_data) < 10:
            return
        
        # Create DataFrame
        df = pd.DataFrame(user_product_data)
        
        # Create user-product matrix
        self.user_product_matrix = df.pivot_table(
            index='user_id', 
            columns='product_id', 
            values='rating', 
            fill_value=0
        )
        
        # Apply NMF for collaborative filtering
        n_components = min(10, min(self.user_product_matrix.shape) - 1)
        if n_components < 2:
            return
            
        self.nmf_model = NMF(n_components=n_components, random_state=42)
        self.user_factors = self.nmf_model.fit_transform(self.user_product_matrix)
        self.product_factors = self.nmf_model.components_
    
    def get_content_based_recommendations(self, product_id, n_recommendations=5):
        """Get content-based recommendations for a product"""
        if self.product_similarity_matrix is None:
            return []
        
        try:
            # Find product index
            product_idx = self.product_ids.index(product_id)
            
            # Get similarity scores
            similarity_scores = list(enumerate(self.product_similarity_matrix[product_idx]))
            
            # Sort by similarity
            similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
            
            # Get top recommendations (excluding the product itself)
            recommendations = []
            for idx, score in similarity_scores[1:n_recommendations+1]:
                if score > 0.1:  # Only recommend if similarity > 0.1
                    recommendations.append({
                        'product_id': self.product_ids[idx],
                        'similarity_score': score
                    })
            
            return recommendations
        except ValueError:
            return []
    
    def get_collaborative_recommendations(self, user_id, n_recommendations=5):
        """Get collaborative filtering recommendations for a user"""
        if self.user_factors is None or self.product_factors is None:
            return []
        
        try:
            # Find user index
            user_idx = self.user_product_matrix.index.get_loc(user_id)
            
            # Get user's latent factors
            user_vector = self.user_factors[user_idx].reshape(1, -1)
            
            # Predict ratings for all products
            predicted_ratings = np.dot(user_vector, self.product_factors).flatten()
            
            # Get products the user hasn't purchased
            user_products = set(self.user_product_matrix.columns[self.user_product_matrix.iloc[user_idx] > 0])
            all_products = set(self.user_product_matrix.columns)
            candidate_products = all_products - user_products
            
            # Get top recommendations
            recommendations = []
            for product_id in candidate_products:
                product_idx = self.user_product_matrix.columns.get_loc(product_id)
                predicted_rating = predicted_ratings[product_idx]
                
                if predicted_rating > 0.5:  # Only recommend if predicted rating > 0.5
                    recommendations.append({
                        'product_id': product_id,
                        'predicted_rating': predicted_rating
                    })
            
            # Sort by predicted rating and return top N
            recommendations.sort(key=lambda x: x['predicted_rating'], reverse=True)
            return recommendations[:n_recommendations]
            
        except (KeyError, ValueError):
            return []
    
    def get_hybrid_recommendations(self, user_id, product_id=None, n_recommendations=5):
        """Get hybrid recommendations combining content-based and collaborative filtering"""
        recommendations = []
        
        # Get collaborative filtering recommendations
        cf_recommendations = self.get_collaborative_recommendations(user_id, n_recommendations)
        
        # Get content-based recommendations if product_id is provided
        cb_recommendations = []
        if product_id:
            cb_recommendations = self.get_content_based_recommendations(product_id, n_recommendations)
        
        # Combine recommendations with weights
        product_scores = defaultdict(float)
        
        # Add collaborative filtering scores
        for rec in cf_recommendations:
            product_scores[rec['product_id']] += rec['predicted_rating'] * 0.6  # 60% weight
        
        # Add content-based scores
        for rec in cb_recommendations:
            product_scores[rec['product_id']] += rec['similarity_score'] * 0.4  # 40% weight
        
        # Convert to list and sort
        for product_id, score in product_scores.items():
            recommendations.append({
                'product_id': product_id,
                'score': score
            })
        
        recommendations.sort(key=lambda x: x['score'], reverse=True)
        return recommendations[:n_recommendations]
    
    def get_popular_products(self, n_recommendations=5):
        """Get popular products based on purchase frequency"""
        # Get product purchase counts
        product_counts = db.session.query(
            OrderItem.product_id,
            db.func.sum(OrderItem.quantity).label('total_quantity')
        ).join(Order).filter(Order.status == 'completed').group_by(OrderItem.product_id).order_by(
            db.func.sum(OrderItem.quantity).desc()
        ).limit(n_recommendations).all()
        
        return [{'product_id': pid, 'popularity_score': qty} for pid, qty in product_counts]
    
    def get_category_based_recommendations(self, category, n_recommendations=5):
        """Get recommendations based on category"""
        products = Product.query.filter_by(category=category).limit(n_recommendations).all()
        return [{'product_id': p.id, 'category_score': 1.0} for p in products]

# Initialize recommendation system
recommendation_system = RecommendationSystem()

def initialize_recommendation_system():
    """Initialize the recommendation system with current data"""
    try:
        print("Building content-based recommendations...")
        recommendation_system.build_content_based_recommendations()
        
        print("Building collaborative filtering recommendations...")
        recommendation_system.build_collaborative_filtering()
        
        print("Recommendation system initialized successfully!")
    except Exception as e:
        print(f"Error initializing recommendation system: {e}")

# ============================================================================
# END MACHINE LEARNING RECOMMENDATION SYSTEM
# ============================================================================

# Sample data for products
def create_sample_products():
    products = [
        # Tires - Most common items from invoices
        {
            'name': 'TYRES 195/65R15 TRIANGLE',
            'description': 'Triangle brand tires, size 195/65R15, suitable for compact cars',
            'price': 72000,
            'category': 'tires',
            'image_url': 'https://images.unsplash.com/photo-1604176354204-92658cb65a8f?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=2070&q=80',
            'stock': 25,
            'is_featured': True
        },
        {
            'name': 'TYRES 225/65R17 TRIANGLE',
            'description': 'Triangle brand tires, size 225/65R17, suitable for SUVs and larger vehicles',
            'price': 110000,
            'category': 'tires',
            'image_url': 'https://images.unsplash.com/photo-1604176354204-92658cb65a8f?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=2070&q=80',
            'stock': 20,
            'is_featured': True
        },
        {
            'name': 'TYRE 265/70R16 TRIANGLE',
            'description': 'Triangle brand tires, size 265/70R16, suitable for 4x4 vehicles and trucks',
            'price': 155000,
            'category': 'tires',
            'image_url': 'https://images.unsplash.com/photo-1604176354204-92658cb65a8f?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=2070&q=80',
            'stock': 15,
            'is_featured': True
        },
        {
            'name': 'TYRES 15-AP TR',
            'description': 'Triangle brand tires, size 15-AP, suitable for light trucks',
            'price': 65000,
            'category': 'tires',
            'image_url': 'https://images.unsplash.com/photo-1604176354204-92658cb65a8f?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=2070&q=80',
            'stock': 30,
            'is_featured': False
        },
        {
            'name': 'TYRE 185/70R14',
            'description': 'Triangle brand tires, size 185/70R14, suitable for small cars',
            'price': 60000,
            'category': 'tires',
            'image_url': 'https://images.unsplash.com/photo-1604176354204-92658cb65a8f?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=2070&q=80',
            'stock': 35,
            'is_featured': False
        },
        
        # Engine Oils - Very common in invoices
        {
            'name': 'ENGINE OIL 15W40 FUT 1L',
            'description': 'High-quality engine oil 15W40, 1 liter bottle, suitable for most vehicles',
            'price': 7000,
            'category': 'oils',
            'image_url': 'https://images.unsplash.com/photo-1635070041078-e363dbe005cb?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=2070&q=80',
            'stock': 100,
            'is_featured': True
        },
        {
            'name': 'ENGINE OIL SYNTHETIC 5W30',
            'description': 'Premium synthetic engine oil 5W30, 1 liter bottle, for modern engines',
            'price': 13500,
            'category': 'oils',
            'image_url': 'https://images.unsplash.com/photo-1635070041078-e363dbe005cb?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=2070&q=80',
            'stock': 80,
            'is_featured': True
        },
        {
            'name': 'ENGINE OIL 15W40 FUT 1L (Premium)',
            'description': 'Premium engine oil 15W40, 1 liter bottle, extended drain intervals',
            'price': 8500,
            'category': 'oils',
            'image_url': 'https://images.unsplash.com/photo-1635070041078-e363dbe005cb?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=2070&q=80',
            'stock': 60,
            'is_featured': False
        },
        
        # Services - Common services from invoices
        {
            'name': 'OIL FILTER SERVICE',
            'description': 'Complete oil filter replacement service including labor',
            'price': 15000,
            'category': 'services',
            'image_url': 'https://images.unsplash.com/photo-1559181567-c3190ca9959b?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=2070&q=80',
            'stock': 999,  # Service item
            'is_featured': True
        },
        {
            'name': 'REPARATION PNEU',
            'description': 'Tire repair service for punctures and minor damage',
            'price': 3000,
            'category': 'services',
            'image_url': 'https://images.unsplash.com/photo-1604176354204-92658cb65a8f?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=2070&q=80',
            'stock': 999,  # Service item
            'is_featured': False
        },
        {
            'name': 'MONTAGE PNEUS',
            'description': 'Tire mounting and balancing service',
            'price': 3000,
            'category': 'services',
            'image_url': 'https://images.unsplash.com/photo-1604176354204-92658cb65a8f?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=2070&q=80',
            'stock': 999,  # Service item
            'is_featured': False
        },
        {
            'name': 'PINCAGE',
            'description': 'Wheel alignment service for proper tire wear and handling',
            'price': 10000,
            'category': 'services',
            'image_url': 'https://images.unsplash.com/photo-1601362840469-51e4d8d58785?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=2070&q=80',
            'stock': 999,  # Service item
            'is_featured': False
        },
        
        # Batteries - From invoices
        {
            'name': 'Battery SEC',
            'description': 'High-quality car battery, suitable for most vehicles',
            'price': 60000,
            'category': 'batteries',
            'image_url': 'https://images.unsplash.com/photo-1568605117036-5fe5e7bab0b7?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=2070&q=80',
            'stock': 20,
            'is_featured': True
        },
        
        # Additional products for variety
        {
            'name': 'A/C SERVICE',
            'description': 'Complete air conditioning service and maintenance',
            'price': 25000,
            'category': 'services',
            'image_url': 'https://images.unsplash.com/photo-1565073182882-993dbb62711f?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=2070&q=80',
            'stock': 999,  # Service item
            'is_featured': False
        },
        {
            'name': 'VALVE TUBELESS REP',
            'description': 'Tubeless tire valve replacement service',
            'price': 3000,
            'category': 'services',
            'image_url': 'https://images.unsplash.com/photo-1604176354204-92658cb65a8f?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=2070&q=80',
            'stock': 999,  # Service item
            'is_featured': False
        },
        {
            'name': 'FUEL CLEANING',
            'description': 'Fuel system cleaning service for better engine performance',
            'price': 28500,
            'category': 'services',
            'image_url': 'https://images.unsplash.com/photo-1635070041078-e363dbe005cb?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=2070&q=80',
            'stock': 999,  # Service item
            'is_featured': False
        }
    ]
    
    for product_data in products:
        existing_product = Product.query.filter_by(name=product_data['name']).first()
        if not existing_product:
            product = Product(**product_data)
            db.session.add(product)
    
    db.session.commit()

def create_sample_users_and_orders():
    """Create sample users and orders for generating recommendations"""
    # Create sample users with different preferences
    users_data = [
        {
            'username': 'john_mechanic',
            'email': 'john@example.com',
            'password': 'password123',
            'first_name': 'John',
            'last_name': 'Smith',
            'is_admin': False
        },
        {
            'username': 'sarah_driver',
            'email': 'sarah@example.com',
            'password': 'password123',
            'first_name': 'Sarah',
            'last_name': 'Johnson',
            'is_admin': False
        },
        {
            'username': 'mike_enthusiast',
            'email': 'mike@example.com',
            'password': 'password123',
            'first_name': 'Mike',
            'last_name': 'Wilson',
            'is_admin': False
        },
        {
            'username': 'lisa_commuter',
            'email': 'lisa@example.com',
            'password': 'password123',
            'first_name': 'Lisa',
            'last_name': 'Brown',
            'is_admin': False
        },
        {
            'username': 'david_racer',
            'email': 'david@example.com',
            'password': 'password123',
            'first_name': 'David',
            'last_name': 'Davis',
            'is_admin': False
        }
    ]
    
    created_users = []
    for user_data in users_data:
        # Check if user already exists
        existing_user = User.query.filter_by(username=user_data['username']).first()
        if not existing_user:
            user = User(
                username=user_data['username'],
                email=user_data['email'],
                password_hash=generate_password_hash(user_data['password']),
                first_name=user_data['first_name'],
                last_name=user_data['last_name'],
                is_admin=user_data['is_admin']
            )
            db.session.add(user)
            created_users.append(user)
            print(f"Created user: {user_data['username']}")
        else:
            created_users.append(existing_user)
            print(f"User already exists: {user_data['username']}")
    
    db.session.commit()
    
    # Create sample orders with different product preferences
    users = User.query.filter_by(is_admin=False).all()
    products = Product.query.all()
    
    if not users or not products:
        print("No users or products found for creating orders.")
        return
    
    # Define user preferences (categories they prefer)
    user_preferences = {
        'john_mechanic': ['tires', 'brakes', 'suspension'],  # Mechanic - technical parts
        'sarah_driver': ['tires', 'batteries', 'lights'],    # Regular driver - essential parts
        'mike_enthusiast': ['suspension', 'brakes', 'oils'], # Car enthusiast - performance parts
        'lisa_commuter': ['tires', 'batteries', 'lights'],   # Commuter - reliability
        'david_racer': ['suspension', 'brakes', 'oils']      # Racer - performance
    }
    
    orders_created = 0
    
    for user in users:
        if user.username not in user_preferences:
            continue
            
        # Create 2-4 orders per user
        num_orders = random.randint(2, 4)
        
        for order_num in range(num_orders):
            # Get products from user's preferred categories
            preferred_categories = user_preferences[user.username]
            preferred_products = [p for p in products if p.category in preferred_categories and p.stock > 0]
            
            if not preferred_products:
                # Fallback to any available products
                preferred_products = [p for p in products if p.stock > 0]
            
            if not preferred_products:
                continue
            
            # Create order
            order_date = datetime.utcnow() - timedelta(days=random.randint(1, 90))
            order = Order(
                user_id=user.id,
                total_amount=0,
                status=random.choice(['completed', 'pending', 'shipped']),
                created_at=order_date
            )
            db.session.add(order)
            db.session.flush()  # Get the order ID
            
            # Add 1-3 items to the order
            num_items = random.randint(1, 3)
            selected_products = random.sample(preferred_products, min(num_items, len(preferred_products)))
            
            total_amount = 0
            for product in selected_products:
                quantity = random.randint(1, 2)
                price = product.price
                total_amount += price * quantity
                
                order_item = OrderItem(
                    order_id=order.id,
                    product_id=product.id,
                    quantity=quantity,
                    price=price
                )
                db.session.add(order_item)
            
            # Update order total
            order.total_amount = total_amount
            orders_created += 1
            print(f"Created order {order.id} for {user.username} with {len(selected_products)} items")
    
    db.session.commit()
    print(f"Created {orders_created} orders")
    
    # Create sample wishlists
    wishlist_items_created = 0
    
    for user in users:
        # Add 2-5 items to each user's wishlist
        num_wishlist_items = random.randint(2, 5)
        selected_products = random.sample(products, min(num_wishlist_items, len(products)))
        
        for product in selected_products:
            # Check if already in wishlist
            existing_wishlist = Wishlist.query.filter_by(
                user_id=user.id, 
                product_id=product.id
            ).first()
            
            if not existing_wishlist:
                wishlist_item = Wishlist(
                    user_id=user.id,
                    product_id=product.id
                )
                db.session.add(wishlist_item)
                wishlist_items_created += 1
                print(f"Added {product.name} to {user.username}'s wishlist")
    
    db.session.commit()
    print(f"Created {wishlist_items_created} wishlist items")
    print("Sample data creation completed!")

# Utility: Create admin user if not exists
def create_admin_user():
    admin = User.query.filter_by(username='admin').first()
    if not admin:
        admin = User()
        admin.username = 'admin'
        admin.email = 'admin@sandcy.com'
        admin.password_hash = generate_password_hash('admin')
        admin.first_name = 'Admin'
        admin.last_name = 'User'
        admin.is_admin = True
        db.session.add(admin)
        db.session.commit()

# RWF: Admin required decorator
def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated or not getattr(current_user, 'is_admin', False):
            flash('Admin access required.', 'error')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# Routes
@app.route('/')
def home():
    featured_products = Product.query.filter_by(is_featured=True).limit(8).all()
    return render_template('index.html', products=featured_products)

@app.route('/search')
def search():
    query = request.args.get('q', '').strip()
    
    if not query:
        return jsonify({'products': [], 'message': 'Please enter a search term'})
    
    # Search in product name, description, and category
    search_term = f'%{query}%'
    products = Product.query.filter(
        db.or_(
            Product.name.ilike(search_term),
            Product.description.ilike(search_term),
            Product.category.ilike(search_term)
        )
    ).limit(10).all()
    
    # Format results for JSON response
    results = []
    for product in products:
        results.append({
            'id': product.id,
            'name': product.name,
            'description': product.description,
            'price': product.price,
            'category': product.category,
            'image_url': product.image_url,
            'url': url_for('product_detail', product_id=product.id)
        })
    
    return jsonify({
        'products': results,
        'count': len(results),
        'query': query
    })

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        confirm_password = request.form['confirm_password']
        first_name = request.form['first_name']
        last_name = request.form['last_name']
        
        # Validation
        if password != confirm_password:
            flash('Passwords do not match!', 'error')
            return render_template('register.html')
        
        # Strong password validation
        is_valid, message = validate_password(password)
        if not is_valid:
            flash(message, 'error')
            return render_template('register.html')
        
        if User.query.filter_by(username=username).first():
            flash('Username already exists!', 'error')
            return render_template('register.html')
        
        if User.query.filter_by(email=email).first():
            flash('Email already registered!', 'error')
            return render_template('register.html')
        
        # Create new user
        user = User(
            username=username,
            email=email,
            password_hash=generate_password_hash(password),
            first_name=first_name,
            last_name=last_name
        )
        
        db.session.add(user)
        db.session.commit()
        
        flash('Registration successful! Please login.', 'success')
        return redirect(url_for('login'))
    
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        user = User.query.filter_by(username=username).first()
        
        if user and check_password_hash(user.password_hash, password):
            login_user(user)
            flash('Login successful!', 'success')
            
            # Redirect admin users to admin dashboard, regular users to home
            if user.is_admin:
                return redirect(url_for('admin_dashboard'))
            else:
                return redirect(url_for('home'))
        else:
            flash('Invalid username or password!', 'error')
    
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out.', 'info')
    return redirect(url_for('home'))

@app.route('/products')
def products():
    category = request.args.get('category', 'all')
    search = request.args.get('search', '').strip()
    
    # Start with base query
    query = Product.query
    
    # Apply category filter
    if category != 'all':
        query = query.filter_by(category=category)
    
    # Apply search filter
    if search:
        search_term = f'%{search}%'
        query = query.filter(
            db.or_(
                Product.name.ilike(search_term),
                Product.description.ilike(search_term),
                Product.category.ilike(search_term)
            )
        )
    
    # Get products
    products = query.all()
    
    # Get categories for filter
    categories = db.session.query(Product.category).distinct().all()
    categories = [cat[0] for cat in categories]
    
    return render_template('products.html', products=products, categories=categories, current_category=category)

@app.route('/product/<int:product_id>')
def product_detail(product_id):
    product = Product.query.get_or_404(product_id)
    return render_template('product_detail_new.html', product=product)

@app.route('/cart')
@login_required
def cart():
    return render_template('cart.html')

@app.route('/add_to_cart', methods=['POST'])
@login_required
def add_to_cart():
    data = request.get_json()
    product_id = data.get('product_id')
    quantity = data.get('quantity', 1)
    
    product = Product.query.get_or_404(product_id)
    
    # Check stock availability
    current_cart_quantity = 0
    if 'cart' in session and str(product_id) in session['cart']:
        current_cart_quantity = session['cart'][str(product_id)]['quantity']
    
    total_requested = current_cart_quantity + quantity
    
    if total_requested > product.stock:
        return jsonify({
            'success': False, 
            'message': f'Only {product.stock} items available in stock. You already have {current_cart_quantity} in your cart.'
        }), 400
    
    # Store cart in session
    if 'cart' not in session:
        session['cart'] = {}
    
    if str(product_id) in session['cart']:
        session['cart'][str(product_id)]['quantity'] += quantity
    else:
        session['cart'][str(product_id)] = {
            'name': product.name,
            'price': product.price,
            'quantity': quantity,
            'image_url': product.image_url,
            'stock': product.stock
        }
    
    session.modified = True
    
    return jsonify({'success': True, 'message': 'Product added to cart'})

@app.route('/update_cart', methods=['POST'])
@login_required
def update_cart():
    data = request.get_json()
    product_id = data.get('product_id')
    quantity = data.get('quantity')
    
    if 'cart' in session and str(product_id) in session['cart']:
        if quantity <= 0:
            del session['cart'][str(product_id)]
        else:
            # Check stock availability before updating
            product = Product.query.get(int(product_id))
            if product and quantity > product.stock:
                return jsonify({
                    'success': False, 
                    'message': f'Only {product.stock} items available in stock.'
                }), 400
            
            session['cart'][str(product_id)]['quantity'] = quantity
        session.modified = True
    
    return jsonify({'success': True})

@app.route('/checkout', methods=['GET', 'POST'])
@login_required
def checkout():
    if request.method == 'POST':
        if 'cart' not in session or not session['cart']:
            flash('Your cart is empty!', 'error')
            return redirect(url_for('cart'))
        
        # Validate stock availability before processing
        insufficient_stock = []
        for product_id, item in session['cart'].items():
            product = Product.query.get(int(product_id))
            if product and product.stock < item['quantity']:
                insufficient_stock.append({
                    'name': product.name,
                    'requested': item['quantity'],
                    'available': product.stock
                })
        
        if insufficient_stock:
            error_message = "Insufficient stock for the following items:\n"
            for item in insufficient_stock:
                error_message += f"• {item['name']}: Requested {item['requested']}, Available {item['available']}\n"
            flash(error_message, 'error')
            return redirect(url_for('cart'))
        
        # Calculate total
        subtotal = sum(item['price'] * item['quantity'] for item in session['cart'].values())
        shipping_cost = 10000  # Fixed shipping cost
        tax = subtotal * 0.08
        total_amount = subtotal + shipping_cost + tax
        
        try:
            # Create order
            order = Order(
                user_id=current_user.id, 
                total_amount=total_amount
            )
            db.session.add(order)
            db.session.flush()  # Get the order ID
            
            # Create order items and decrement stock
            for product_id, item in session['cart'].items():
                product = Product.query.get(int(product_id))
                if product:
                    # Decrement stock
                    product.stock -= item['quantity']
                    
                    # Create order item
                    order_item = OrderItem(
                        order_id=order.id,
                        product_id=int(product_id),
                        quantity=item['quantity'],
                        price=item['price']
                    )
                    db.session.add(order_item)
            
            # Commit all changes
            db.session.commit()
            
            # Clear cart
            session.pop('cart', None)
            
            flash('Order placed successfully! Stock has been updated.', 'success')
            return redirect(url_for('orders'))
            
        except Exception as e:
            # Rollback in case of error
            db.session.rollback()
            flash('An error occurred while processing your order. Please try again.', 'error')
            return redirect(url_for('cart'))
    
    return render_template('checkout.html')

@app.route('/orders')
@login_required
def orders():
    user_orders = Order.query.filter_by(user_id=current_user.id).options(
        joinedload(Order.items).joinedload(OrderItem.product)
    ).order_by(Order.created_at.desc()).all()
    return render_template('orders.html', orders=user_orders)

@app.route('/order/<int:order_id>')
@login_required
def order_detail(order_id):
    order = Order.query.filter_by(id=order_id, user_id=current_user.id).options(
        joinedload(Order.items).joinedload(OrderItem.product),
        joinedload(Order.user)
    ).first_or_404()
    return render_template('order_detail.html', order=order)

@app.route('/order/<int:order_id>/invoice')
@login_required
def download_invoice(order_id):
    order = Order.query.filter_by(id=order_id, user_id=current_user.id).options(
        joinedload(Order.items).joinedload(OrderItem.product),
        joinedload(Order.user)
    ).first_or_404()
    
    # Create PDF
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    elements = []
    styles = getSampleStyleSheet()
    
    # Title
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30,
        alignment=1  # Center alignment
    )
    elements.append(Paragraph("INVOICE", title_style))
    elements.append(Spacer(1, 20))
    
    # Company and Order Info
    company_info = [
        ["Sandcy Ltd", f"Order #: {order.id}"],
        ["123 Auto Parts Street", f"Date: {order.created_at.strftime('%B %d, %Y')}"],
        ["Kigali, Rwanda", f"Status: {order.status.title()}"],
        ["Phone: +250 123 456 789", ""]
    ]
    
    company_table = Table(company_info, colWidths=[3*inch, 3*inch])
    company_table.setStyle(TableStyle([
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
    ]))
    elements.append(company_table)
    elements.append(Spacer(1, 20))
    
    # Customer Info
    customer_info = [
        ["Bill To:", ""],
        [f"{order.user.first_name} {order.user.last_name}", ""],
        [order.user.email, ""],
        ["", ""]
    ]
    
    customer_table = Table(customer_info, colWidths=[3*inch, 3*inch])
    elements.append(customer_table)
    elements.append(Spacer(1, 20))
    
    # Items Table
    items_data = [["Item", "Description", "Qty", "Price", "Total"]]
    for item in order.items:
        if item.product:
            items_data.append([
                item.product.name,
                item.product.description[:50] + "..." if len(item.product.description) > 50 else item.product.description,
                str(item.quantity),
                f"{item.price:.0f} RWF",
                f"{(item.price * item.quantity):.0f} RWF"
            ])
        else:
            items_data.append([
                f"Product #{item.product_id}",
                "Product information unavailable",
                str(item.quantity),
                f"{item.price:.0f} RWF",
                f"{(item.price * item.quantity):.0f} RWF"
            ])
    
    items_table = Table(items_data, colWidths=[1.5*inch, 2*inch, 0.5*inch, 1*inch, 1*inch])
    items_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('ALIGN', (2, 1), (-1, -1), 'RIGHT'),  # Right align numbers
        ('ALIGN', (0, 1), (1, -1), 'LEFT'),    # Left align text
    ]))
    elements.append(items_table)
    elements.append(Spacer(1, 20))
    
    # Total
    subtotal = sum(item.price * item.quantity for item in order.items)
    shipping_cost = 10000  # Fixed shipping cost
    tax = subtotal * 0.08
    
    total_data = [
        ["", "", "", "Subtotal:", f"{subtotal:.0f} RWF"],
        ["", "", "", "Shipping:", f"{shipping_cost:.0f} RWF"],
        ["", "", "", "Tax (8%):", f"{tax:.0f} RWF"],
        ["", "", "", "Total:", f"{order.total_amount:.0f} RWF"]
    ]
    total_table = Table(total_data, colWidths=[1.5*inch, 2*inch, 0.5*inch, 1*inch, 1*inch])
    total_table.setStyle(TableStyle([
        ('ALIGN', (0, 0), (-1, -1), 'RIGHT'),
        ('FONTNAME', (3, 0), (-1, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 12),
        ('LINEABOVE', (3, -1), (-1, -1), 1, colors.black),  # Line above total only
    ]))
    elements.append(total_table)
    
    # Build PDF
    doc.build(elements)
    buffer.seek(0)
    
    return send_file(
        buffer,
        as_attachment=True,
        download_name=f'invoice_order_{order.id}.pdf',
        mimetype='application/pdf'
    )

@app.route('/order/<int:order_id>/track')
@login_required
def track_package(order_id):
    order = Order.query.filter_by(id=order_id, user_id=current_user.id).options(
        joinedload(Order.items).joinedload(OrderItem.product),
        joinedload(Order.user)
    ).first_or_404()
    
    # Generate tracking information based on order status
    tracking_info = {
        'pending': {
            'status': 'Order Confirmed',
            'description': 'Your order has been received and is being processed.',
            'estimated_delivery': '3-5 business days',
            'steps': [
                {'status': 'Order Placed', 'completed': True, 'date': order.created_at.strftime('%B %d, %Y at %I:%M %p')},
                {'status': 'Processing', 'completed': order.status != 'pending', 'date': None},
                {'status': 'Shipped', 'completed': order.status in ['shipped', 'completed'], 'date': None},
                {'status': 'Delivered', 'completed': order.status == 'completed', 'date': None}
            ]
        },
        'processing': {
            'status': 'Processing Order',
            'description': 'Your order is being prepared for shipment.',
            'estimated_delivery': '2-4 business days',
            'steps': [
                {'status': 'Order Placed', 'completed': True, 'date': order.created_at.strftime('%B %d, %Y at %I:%M %p')},
                {'status': 'Processing', 'completed': True, 'date': 'Currently in progress'},
                {'status': 'Shipped', 'completed': order.status in ['shipped', 'completed'], 'date': None},
                {'status': 'Delivered', 'completed': order.status == 'completed', 'date': None}
            ]
        },
        'shipped': {
            'status': 'Package Shipped',
            'description': 'Your package is on its way to you.',
            'estimated_delivery': '1-2 business days',
            'steps': [
                {'status': 'Order Placed', 'completed': True, 'date': order.created_at.strftime('%B %d, %Y at %I:%M %p')},
                {'status': 'Processing', 'completed': True, 'date': 'Completed'},
                {'status': 'Shipped', 'completed': True, 'date': 'Package dispatched'},
                {'status': 'Delivered', 'completed': order.status == 'completed', 'date': None}
            ]
        },
        'completed': {
            'status': 'Package Delivered',
            'description': 'Your package has been successfully delivered.',
            'estimated_delivery': 'Delivered',
            'steps': [
                {'status': 'Order Placed', 'completed': True, 'date': order.created_at.strftime('%B %d, %Y at %I:%M %p')},
                {'status': 'Processing', 'completed': True, 'date': 'Completed'},
                {'status': 'Shipped', 'completed': True, 'date': 'Package dispatched'},
                {'status': 'Delivered', 'completed': True, 'date': 'Package delivered'}
            ]
        },
        'cancelled': {
            'status': 'Order Cancelled',
            'description': 'This order has been cancelled.',
            'estimated_delivery': 'N/A',
            'steps': [
                {'status': 'Order Placed', 'completed': True, 'date': order.created_at.strftime('%B %d, %Y at %I:%M %p')},
                {'status': 'Cancelled', 'completed': True, 'date': 'Order cancelled'}
            ]
        }
    }
    
    tracking = tracking_info.get(order.status, tracking_info['pending'])
    
    return render_template('track_package.html', order=order, tracking=tracking)

@app.route('/profile')
@login_required
def profile():
    return render_template('profile.html')

@app.route('/admin')
@login_required
@admin_required
def admin_dashboard():
    users = User.query.all()
    orders = Order.query.all()
    products = Product.query.all()
    return render_template('admin_dashboard.html', users=users, orders=orders, products=products)

@app.route('/admin/products')
@login_required
@admin_required
def admin_products():
    search = request.args.get('search', '').strip()
    
    # Start with base query
    query = Product.query
    
    # Apply search filter
    if search:
        search_term = f'%{search}%'
        query = query.filter(
            db.or_(
                Product.name.ilike(search_term),
                Product.description.ilike(search_term),
                Product.category.ilike(search_term)
            )
        )
    
    # Get products
    products = query.all()
    return render_template('admin_products.html', products=products)

@app.route('/admin/products/add', methods=['GET', 'POST'])
@login_required
@admin_required
def admin_add_product():
    if request.method == 'POST':
        name = request.form['name']
        description = request.form['description']
        price = float(request.form['price'])
        category = request.form['category']
        image_url = request.form['image_url']
        stock = int(request.form['stock'])
        is_featured = 'is_featured' in request.form
        
        # Handle tire-specific fields
        tire_width = request.form.get('tire_width', '') if category == 'tires' else None
        tire_aspect_ratio = request.form.get('tire_aspect_ratio', '') if category == 'tires' else None
        tire_rim_size = request.form.get('tire_rim_size', '') if category == 'tires' else None
        tire_brand = request.form.get('tire_brand', '') if category == 'tires' else None
        
        product = Product(
            name=name,
            description=description,
            price=price,
            category=category,
            image_url=image_url,
            stock=stock,
            is_featured=is_featured,
            tire_width=tire_width,
            tire_aspect_ratio=tire_aspect_ratio,
            tire_rim_size=tire_rim_size,
            tire_brand=tire_brand
        )
        
        db.session.add(product)
        db.session.commit()
        
        flash('Product added successfully!', 'success')
        return redirect(url_for('admin_products'))
    
    return render_template('admin_add_product.html')

@app.route('/admin/products/edit/<int:product_id>', methods=['GET', 'POST'])
@login_required
@admin_required
def admin_edit_product(product_id):
    product = Product.query.get_or_404(product_id)
    
    if request.method == 'POST':
        product.name = request.form['name']
        product.description = request.form['description']
        product.price = float(request.form['price'])
        product.category = request.form['category']
        product.image_url = request.form['image_url']
        product.stock = int(request.form['stock'])
        product.is_featured = 'is_featured' in request.form
        
        # Handle tire-specific fields
        if product.category == 'tires':
            product.tire_width = request.form.get('tire_width', '')
            product.tire_aspect_ratio = request.form.get('tire_aspect_ratio', '')
            product.tire_rim_size = request.form.get('tire_rim_size', '')
            product.tire_brand = request.form.get('tire_brand', '')
        else:
            # Clear tire fields if category is not tires
            product.tire_width = None
            product.tire_aspect_ratio = None
            product.tire_rim_size = None
            product.tire_brand = None
        
        db.session.commit()
        
        flash('Product updated successfully!', 'success')
        return redirect(url_for('admin_products'))
    
    return render_template('admin_edit_product.html', product=product)

@app.route('/admin/products/delete/<int:product_id>', methods=['POST'])
@login_required
@admin_required
def admin_delete_product(product_id):
    product = Product.query.get_or_404(product_id)
    db.session.delete(product)
    db.session.commit()
    
    flash('Product deleted successfully!', 'success')
    return redirect(url_for('admin_products'))

@app.route('/admin/users')
@login_required
@admin_required
def admin_users():
    search = request.args.get('search', '').strip()
    
    # Start with base query
    query = User.query
    
    # Apply search filter
    if search:
        search_term = f'%{search}%'
        query = query.filter(
            db.or_(
                User.username.ilike(search_term),
                User.email.ilike(search_term),
                User.first_name.ilike(search_term),
                User.last_name.ilike(search_term)
            )
        )
    
    # Get users
    users = query.all()
    return render_template('admin_users.html', users=users)

@app.route('/admin/users/edit/<int:user_id>', methods=['GET', 'POST'])
@login_required
@admin_required
def admin_edit_user(user_id):
    user = User.query.get_or_404(user_id)
    
    if request.method == 'POST':
        user.username = request.form['username']
        user.email = request.form['email']
        user.first_name = request.form['first_name']
        user.last_name = request.form['last_name']
        user.is_admin = 'is_admin' in request.form
        
        if request.form['password']:
            user.password_hash = generate_password_hash(request.form['password'])
        
        db.session.commit()
        
        flash('User updated successfully!', 'success')
        return redirect(url_for('admin_users'))
    
    return render_template('admin_edit_user.html', user=user)

@app.route('/admin/users/delete/<int:user_id>', methods=['POST'])
@login_required
@admin_required
def admin_delete_user(user_id):
    if user_id == current_user.id:
        flash('You cannot delete your own account!', 'error')
        return redirect(url_for('admin_users'))
    
    user = User.query.get_or_404(user_id)
    db.session.delete(user)
    db.session.commit()
    
    flash('User deleted successfully!', 'success')
    return redirect(url_for('admin_users'))

@app.route('/admin/orders')
@login_required
@admin_required
def admin_orders():
    search = request.args.get('search', '').strip()
    
    # Start with base query
    query = Order.query.options(
        joinedload(Order.items).joinedload(OrderItem.product),
        joinedload(Order.user)
    )
    
    # Apply search filter
    if search:
        search_term = f'%{search}%'
        query = query.filter(
            db.or_(
                Order.id == search if search.isdigit() else db.literal(False),
                User.first_name.ilike(search_term),
                User.last_name.ilike(search_term),
                User.email.ilike(search_term),
                User.username.ilike(search_term)
            )
        ).join(User)
    
    # Get orders ordered by creation date
    orders = query.order_by(Order.created_at.desc()).all()
    return render_template('admin_orders.html', orders=orders)

@app.route('/admin/orders/<int:order_id>')
@login_required
@admin_required
def admin_order_detail(order_id):
    order = Order.query.options(
        joinedload(Order.items).joinedload(OrderItem.product),
        joinedload(Order.user)
    ).get_or_404(order_id)
    return render_template('admin_order_detail.html', order=order)

@app.route('/admin/orders/update_status/<int:order_id>', methods=['POST'])
@login_required
@admin_required
def admin_update_order_status(order_id):
    order = Order.query.get_or_404(order_id)
    order.status = request.form['status']
    db.session.commit()
    
    flash('Order status updated successfully!', 'success')
    return redirect(url_for('admin_order_detail', order_id=order_id))

@app.route('/account_settings', methods=['GET', 'POST'])
@login_required
def account_settings():
    if request.method == 'POST':
        # Get form data
        first_name = request.form.get('first_name', '').strip()
        last_name = request.form.get('last_name', '').strip()
        email = request.form.get('email', '').strip()
        current_password = request.form.get('current_password', '').strip()
        new_password = request.form.get('new_password', '').strip()
        confirm_password = request.form.get('confirm_password', '').strip()
        
        # Check if any changes were actually made
        changes_made = False
        password_changed = False
        
        # Check if basic info changed
        if first_name != (current_user.first_name or ''):
            changes_made = True
        if last_name != (current_user.last_name or ''):
            changes_made = True
        if email != current_user.email:
            changes_made = True
        if current_password or new_password or confirm_password:
            changes_made = True
        
        # If no changes detected, show error and redirect back
        if not changes_made:
            flash('No changes detected. Please make changes to your profile information before saving.', 'error')
            return redirect(url_for('account_settings'))
        
        # Update basic info
        if first_name:
            current_user.first_name = first_name
        if last_name:
            current_user.last_name = last_name
        if email and email != current_user.email:
            # Check if email already exists
            existing_user = User.query.filter_by(email=email).first()
            if existing_user and existing_user.id != current_user.id:
                flash('Email already exists!', 'error')
                return redirect(url_for('account_settings'))
            current_user.email = email
        
        # Update password if provided
        if current_password and new_password and confirm_password:
            if not check_password_hash(current_user.password_hash, current_password):
                flash('Current password is incorrect!', 'error')
                return redirect(url_for('account_settings'))
            
            if new_password != confirm_password:
                flash('New passwords do not match!', 'error')
                return redirect(url_for('account_settings'))
            
            # Strong password validation
            is_valid, message = validate_password(new_password)
            if not is_valid:
                flash(message, 'error')
                return redirect(url_for('account_settings'))
            
            current_user.password_hash = generate_password_hash(new_password)
            password_changed = True
        
        # Save changes
        db.session.commit()
        
        if password_changed:
            # Logout user and redirect to login after password change
            logout_user()
            flash('Password updated successfully! Please login with your new password.', 'success')
            return redirect(url_for('login'))
        else:
            # Stay on account settings if only basic info was updated
            flash('Profile updated successfully!', 'success')
            return redirect(url_for('account_settings'))
    
    return render_template('account_settings.html')

@app.route('/wishlist')
@login_required
def wishlist():
    wishlist_items = Wishlist.query.filter_by(user_id=current_user.id).options(
        joinedload(Wishlist.product)
    ).all()
    return render_template('wishlist.html', wishlist_items=wishlist_items)

@app.route('/add_to_wishlist', methods=['POST'])
@login_required
def add_to_wishlist():
    data = request.get_json()
    product_id = data.get('product_id')
    
    # Check if already in wishlist
    existing_item = Wishlist.query.filter_by(
        user_id=current_user.id, 
        product_id=product_id
    ).first()
    
    if existing_item:
        return jsonify({'success': False, 'message': 'Product already in wishlist'})
    
    # Add to wishlist
    wishlist_item = Wishlist(user_id=current_user.id, product_id=product_id)
    db.session.add(wishlist_item)
    db.session.commit()
    
    return jsonify({'success': True, 'message': 'Added to wishlist'})

@app.route('/remove_from_wishlist', methods=['POST'])
@login_required
def remove_from_wishlist():
    data = request.get_json()
    product_id = data.get('product_id')
    
    # Remove from wishlist
    wishlist_item = Wishlist.query.filter_by(
        user_id=current_user.id, 
        product_id=product_id
    ).first()
    
    if wishlist_item:
        db.session.delete(wishlist_item)
        db.session.commit()
        return jsonify({'success': True, 'message': 'Removed from wishlist'})
    
    return jsonify({'success': False, 'message': 'Item not found in wishlist'})

# ============================================================================
# RECOMMENDATION ROUTES
# ============================================================================

@app.route('/recommendations')
@login_required
def get_recommendations():
    """Get personalized recommendations for the current user"""
    try:
        # Get different types of recommendations
        hybrid_recs = recommendation_system.get_hybrid_recommendations(
            current_user.id, n_recommendations=8
        )
        
        popular_recs = recommendation_system.get_popular_products(n_recommendations=4)
        
        # Get product details for recommendations
        recommended_products = []
        
        # Add hybrid recommendations
        for rec in hybrid_recs:
            product = Product.query.get(rec['product_id'])
            if product and product.stock > 0:
                recommended_products.append({
                    'product': product,
                    'score': rec['score'],
                    'type': 'personalized'
                })
        
        # Add popular products if we don't have enough personalized ones
        if len(recommended_products) < 8:
            for rec in popular_recs:
                product = Product.query.get(rec['product_id'])
                if product and product.stock > 0:
                    # Check if product is already in recommendations
                    if not any(p['product'].id == product.id for p in recommended_products):
                        recommended_products.append({
                            'product': product,
                            'score': rec['popularity_score'],
                            'type': 'popular'
                        })
        
        return jsonify({
            'success': True,
            'recommendations': [
                {
                    'id': rec['product'].id,
                    'name': rec['product'].name,
                    'description': rec['product'].description,
                    'price': rec['product'].price,
                    'category': rec['product'].category,
                    'image_url': rec['product'].image_url,
                    'score': rec['score'],
                    'type': rec['type']
                }
                for rec in recommended_products[:8]
            ]
        })
        
    except Exception as e:
        print(f"Error getting recommendations: {e}")
        return jsonify({'success': False, 'message': 'Error getting recommendations'})

@app.route('/recommendations/product/<int:product_id>')
def get_product_recommendations(product_id):
    """Get content-based recommendations for a specific product"""
    try:
        recommendations = recommendation_system.get_content_based_recommendations(
            product_id, n_recommendations=6
        )
        
        # Get product details
        recommended_products = []
        for rec in recommendations:
            product = Product.query.get(rec['product_id'])
            if product and product.stock > 0:
                recommended_products.append({
                    'id': product.id,
                    'name': product.name,
                    'description': product.description,
                    'price': product.price,
                    'category': product.category,
                    'image_url': product.image_url,
                    'similarity_score': rec['similarity_score']
                })
        
        return jsonify({
            'success': True,
            'recommendations': recommended_products
        })
        
    except Exception as e:
        print(f"Error getting product recommendations: {e}")
        return jsonify({'success': False, 'message': 'Error getting recommendations'})

@app.route('/recommendations/category/<category>')
def get_category_recommendations(category):
    """Get recommendations based on category"""
    try:
        recommendations = recommendation_system.get_category_based_recommendations(
            category, n_recommendations=6
        )
        
        # Get product details
        recommended_products = []
        for rec in recommendations:
            product = Product.query.get(rec['product_id'])
            if product and product.stock > 0:
                recommended_products.append({
                    'id': product.id,
                    'name': product.name,
                    'description': product.description,
                    'price': product.price,
                    'category': product.category,
                    'image_url': product.image_url,
                    'category_score': rec['category_score']
                })
        
        return jsonify({
            'success': True,
            'recommendations': recommended_products
        })
        
    except Exception as e:
        print(f"Error getting category recommendations: {e}")
        return jsonify({'success': False, 'message': 'Error getting recommendations'})

@app.route('/admin/recommendations/rebuild')
@login_required
@admin_required
def rebuild_recommendations():
    """Admin route to rebuild recommendation models"""
    try:
        initialize_recommendation_system()
        flash('Recommendation models rebuilt successfully!', 'success')
    except Exception as e:
        flash(f'Error rebuilding recommendations: {e}', 'error')
    
    return redirect(url_for('admin_dashboard'))

# ============================================================================
# END RECOMMENDATION ROUTES
# ============================================================================

@app.route('/tire_filter', methods=['GET', 'POST'])
def tire_filter():
    """Handle tire filtering by size and brand"""
    if request.method == 'POST':
        data = request.get_json()
        width = data.get('width')
        aspect_ratio = data.get('aspect_ratio')
        rim_size = data.get('rim_size')
        brand = data.get('brand')
        
        # Build query for tire products
        query = Product.query.filter_by(category='tires')
        
        # Apply filters if provided
        if width and width != 'Select Width':
            query = query.filter_by(tire_width=width)
        if aspect_ratio and aspect_ratio != 'Select Ratio':
            query = query.filter_by(tire_aspect_ratio=aspect_ratio)
        if rim_size and rim_size != 'Select Rim':
            query = query.filter_by(tire_rim_size=rim_size)
        if brand and brand != 'Select Brand':
            query = query.filter_by(tire_brand=brand)
        
        # Get filtered products
        products = query.all()
        
        # Convert to JSON-serializable format
        products_data = []
        for product in products:
            products_data.append({
                'id': product.id,
                'name': product.name,
                'description': product.description,
                'price': product.price,
                'image_url': product.image_url,
                'tire_width': product.tire_width,
                'tire_aspect_ratio': product.tire_aspect_ratio,
                'tire_rim_size': product.tire_rim_size,
                'tire_brand': product.tire_brand,
                'stock': product.stock
            })
        
        return jsonify({
            'success': True,
            'products': products_data,
            'count': len(products_data)
        })
    
    # GET request - redirect to products page with tire category
    return redirect(url_for('products', category='tires'))

@app.route('/tire_filter_options')
def tire_filter_options():
    """Get available tire filter options for dropdowns"""
    # Get all tire products
    tires = Product.query.filter_by(category='tires').all()
    
    # Extract unique values for each filter
    widths = sorted(list(set(tire.tire_width for tire in tires if tire.tire_width)))
    aspect_ratios = sorted(list(set(tire.tire_aspect_ratio for tire in tires if tire.tire_aspect_ratio)))
    rim_sizes = sorted(list(set(tire.tire_rim_size for tire in tires if tire.tire_rim_size)))
    brands = sorted(list(set(tire.tire_brand for tire in tires if tire.tire_brand)))
    
    return jsonify({
        'widths': widths,
        'aspect_ratios': aspect_ratios,
        'rim_sizes': rim_sizes,
        'brands': brands
    })

# ============================================================================
# PAYMENT PROCESSING ROUTES
# ============================================================================

@app.route('/process_momopay_payment', methods=['POST'])
@login_required
def process_momopay_payment():
    """Process MoMoPay payment"""
    try:
        data = request.get_json()
        
        # Validate cart
        if 'cart' not in session or not session['cart']:
            return jsonify({'success': False, 'message': 'Cart is empty'}), 400
        
        # Calculate total
        subtotal = sum(item['price'] * item['quantity'] for item in session['cart'].values())
        shipping_cost = 10000
        tax = subtotal * 0.08
        total_amount = subtotal + shipping_cost + tax
        
        # Generate unique transaction reference
        import uuid
        tx_ref = f"MOMO_{uuid.uuid4().hex[:8].upper()}"
        
        # Create order record (pending payment)
        order = Order(
            user_id=current_user.id,
            total_amount=total_amount,
            status='pending_payment'
        )
        db.session.add(order)
        db.session.flush()
        
        # Create order items
        for product_id, item in session['cart'].items():
            order_item = OrderItem(
                order_id=order.id,
                product_id=int(product_id),
                quantity=item['quantity'],
                price=item['price']
            )
            db.session.add(order_item)
        
        db.session.commit()
        
        # Store order info in session for payment verification
        session['pending_order_id'] = order.id
        session['payment_method'] = 'momopay'
        session['tx_ref'] = tx_ref
        
        # For demo purposes, simulate MoMoPay payment
        # In production, integrate with actual MoMoPay API
        return jsonify({
            'success': True,
            'message': 'MoMoPay payment initiated',
            'order_id': order.id,
            'tx_ref': tx_ref,
            'amount': total_amount,
            'payment_url': url_for('momopay_payment_page', order_id=order.id, _external=True)
        })
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/process_card_payment', methods=['POST'])
@login_required
def process_card_payment():
    """Process card payment with Flutterwave"""
    try:
        data = request.get_json()
        
        # Validate cart
        if 'cart' not in session or not session['cart']:
            return jsonify({'success': False, 'message': 'Cart is empty'}), 400
        
        # Calculate total
        subtotal = sum(item['price'] * item['quantity'] for item in session['cart'].values())
        shipping_cost = 10000
        tax = subtotal * 0.08
        total_amount = subtotal + shipping_cost + tax
        
        # Generate unique transaction reference
        import uuid
        tx_ref = f"FLW_{uuid.uuid4().hex[:8].upper()}"
        
        # Create order record (pending payment)
        order = Order(
            user_id=current_user.id,
            total_amount=total_amount,
            status='pending_payment'
        )
        db.session.add(order)
        db.session.flush()
        
        # Create order items
        for product_id, item in session['cart'].items():
            order_item = OrderItem(
                order_id=order.id,
                product_id=int(product_id),
                quantity=item['quantity'],
                price=item['price']
            )
            db.session.add(order_item)
        
        db.session.commit()
        
        # Store order info in session
        session['pending_order_id'] = order.id
        session['payment_method'] = 'card'
        session['tx_ref'] = tx_ref
        
        # Flutterwave configuration (use test keys for development)
        flutterwave_config = {
            'public_key': 'FLWPUBK-0c4347ddda625a4f63c27b43005889bc-X', 
            'tx_ref': tx_ref,
            'amount': total_amount,
            'currency': 'RWF',
            'customer': {
                'email': data.get('email'),
                'phone_number': data.get('phone'),
                'name': f"{data.get('first_name')} {data.get('last_name')}"
            }
        }
        
        return jsonify({
            'success': True,
            'message': 'Card payment initialized',
            'order_id': order.id,
            'tx_ref': tx_ref,
            'amount': total_amount,
            'currency': 'RWF',
            'public_key': flutterwave_config['public_key'],
            'customer': flutterwave_config['customer']
        })
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/process_flutterwave_payment', methods=['POST'])
@login_required
def process_flutterwave_payment():
    """Process payment with Flutterwave (both MoMo and Card)"""
    try:
        data = request.get_json()
        payment_method = data.get('payment_method')
        
        # Validate cart
        if 'cart' not in session or not session['cart']:
            return jsonify({'success': False, 'message': 'Cart is empty'}), 400
        
        # Calculate total
        subtotal = sum(item['price'] * item['quantity'] for item in session['cart'].values())
        shipping_cost = 10000
        tax = subtotal * 0.08
        total_amount = subtotal + shipping_cost + tax
        
        # Generate unique transaction reference
        import uuid
        tx_ref = f"FLW_{uuid.uuid4().hex[:8].upper()}"
        
        # Create order record (pending payment)
        order = Order(
            user_id=current_user.id,
            total_amount=total_amount,
            status='pending_payment'
        )
        db.session.add(order)
        db.session.flush()
        
        # Create order items
        for product_id, item in session['cart'].items():
            order_item = OrderItem(
                order_id=order.id,
                product_id=int(product_id),
                quantity=item['quantity'],
                price=item['price']
            )
            db.session.add(order_item)
        
        db.session.commit()
        
        # Store order info in session
        session['pending_order_id'] = order.id
        session['payment_method'] = payment_method
        session['tx_ref'] = tx_ref
        
        # Flutterwave configuration (use test keys for development)
        # In production, use actual Flutterwave keys from config
        flutterwave_config = {
            'public_key': 'FLWPUBK-0c4347ddda625a4f63c27b43005889bc-X', 
            'tx_ref': tx_ref,
            'amount': total_amount,
            'currency': 'RWF',
            'payment_options': 'mobilemoney,card,banktransfer' if payment_method == 'momopay' else 'card,banktransfer',
            'customer': {
                'email': data.get('email'),
                'phone_number': data.get('phone'),
                'name': f"{data.get('first_name')} {data.get('last_name')}"
            }
        }
        
        return jsonify({
            'success': True,
            'message': f'{payment_method.title()} payment initialized',
            'order_id': order.id,
            'tx_ref': tx_ref,
            'amount': total_amount,
            'currency': 'RWF',
            'public_key': flutterwave_config['public_key'],
            'customer': flutterwave_config['customer']
        })
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/verify_payment', methods=['POST'])
@login_required
def verify_payment():
    """Verify payment with Flutterwave"""
    try:
        data = request.get_json()
        tx_ref = data.get('tx_ref')
        
        # In production, verify with Flutterwave API
        # For demo purposes, we'll simulate successful verification
        
        order = Order.query.filter_by(id=session.get('pending_order_id')).first()
        if not order:
            return jsonify({'success': False, 'message': 'Order not found'}), 404
        
        if order.status != 'pending_payment':
            return jsonify({'success': False, 'message': 'Invalid payment status'}), 400
        
        # Simulate payment verification
        order.status = 'completed'
        
        # Update product stock
        for item in order.items:
            product = Product.query.get(item.product_id)
            if product:
                product.stock -= item.quantity
        
        db.session.commit()
        
        # Clear cart and session data
        session.pop('cart', None)
        session.pop('pending_order_id', None)
        session.pop('payment_method', None)
        session.pop('tx_ref', None)
        
        return jsonify({
            'success': True,
            'message': 'Payment verified successfully',
            'order_id': order.id
        })
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/payment_webhook', methods=['POST'])
def payment_webhook():
    """Handle payment webhooks from Flutterwave"""
    try:
        data = request.get_json()
        
        # Verify webhook signature (implement proper verification)
        # For demo purposes, we'll accept all webhooks
        
        tx_ref = data.get('tx_ref')
        status = data.get('status')
        
        if status == 'successful':
            # Find order by tx_ref
            order = Order.query.filter_by(id=session.get('pending_order_id')).first()
            if order:
                order.status = 'completed'
                
                # Update product stock
                for item in order.items:
                    product = Product.query.get(item.product_id)
                    if product:
                        product.stock -= item.quantity
                
                db.session.commit()
                
                # Clear cart and session data
                session.pop('cart', None)
                session.pop('pending_order_id', None)
                session.pop('payment_method', None)
                session.pop('tx_ref', None)
        
        return jsonify({'success': True}), 200
        
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/momopay_payment_page/<int:order_id>')
@login_required
def momopay_payment_page(order_id):
    """MoMoPay payment page"""
    order = Order.query.filter_by(id=order_id, user_id=current_user.id).first_or_404()
    
    if order.status != 'pending_payment':
        flash('Invalid payment status', 'error')
        return redirect(url_for('orders'))
    
    return render_template('momopay_payment.html', order=order)

@app.route('/confirm_momopay_payment/<int:order_id>', methods=['POST'])
@login_required
def confirm_momopay_payment(order_id):
    """Confirm MoMoPay payment (simulated)"""
    try:
        order = Order.query.filter_by(id=order_id, user_id=current_user.id).first_or_404()
        
        if order.status != 'pending_payment':
            return jsonify({'success': False, 'message': 'Invalid payment status'}), 400
        
        # Simulate payment confirmation
        order.status = 'completed'
        
        # Update product stock
        for item in order.items:
            product = Product.query.get(item.product_id)
            if product:
                product.stock -= item.quantity
        
        db.session.commit()
        
        # Clear cart and session data
        session.pop('cart', None)
        session.pop('pending_order_id', None)
        session.pop('payment_method', None)
        session.pop('tx_ref', None)
        
        return jsonify({
            'success': True,
            'message': 'Payment confirmed successfully',
            'redirect_url': url_for('orders')
        })
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'message': str(e)}), 500

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        create_sample_products()
        create_sample_users_and_orders()
        # create_admin_user()  # Removed demo admin user creation
        
        # Initialize recommendation system
        print("Initializing recommendation system...")
        initialize_recommendation_system()
        
    app.run(debug=True) 