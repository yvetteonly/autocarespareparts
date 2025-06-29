#!/usr/bin/env python3
"""
Database migration script to add tire-specific fields to the Product model.
Run this script to update your existing database with the new tire fields.
"""

import sqlite3
import os
from app import app, db, Product
import re

def update_database():
    """Add tire-specific columns to the Product table"""
    
    # Database file path
    db_path = 'instance/autoparts.db'
    
    # Check if database exists
    if not os.path.exists(db_path):
        print(f"Database not found at {db_path}")
        print("Please run the Flask app first to create the database.")
        return
    
    try:
        # Connect to the database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Check if columns already exist
        cursor.execute("PRAGMA table_info(product)")
        columns = [column[1] for column in cursor.fetchall()]
        
        # Add new columns if they don't exist
        new_columns = [
            ('tire_width', 'VARCHAR(10)'),
            ('tire_aspect_ratio', 'VARCHAR(10)'),
            ('tire_rim_size', 'VARCHAR(10)'),
            ('tire_brand', 'VARCHAR(50)')
        ]
        
        for column_name, column_type in new_columns:
            if column_name not in columns:
                print(f"Adding column: {column_name}")
                cursor.execute(f"ALTER TABLE product ADD COLUMN {column_name} {column_type}")
            else:
                print(f"Column {column_name} already exists")
        
        # Commit changes
        conn.commit()
        print("Database updated successfully!")
        
        # Show updated table structure
        cursor.execute("PRAGMA table_info(product)")
        print("\nUpdated Product table structure:")
        for column in cursor.fetchall():
            print(f"  {column[1]} ({column[2]})")
        
    except sqlite3.Error as e:
        print(f"Database error: {e}")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if conn:
            conn.close()

def update_tire_products():
    """Update existing tire products with tire-specific fields and add more tire products"""
    
    # Update existing tire products with tire-specific fields
    tire_updates = [
        {
            'name': 'TYRES 195/65R15 TRIANGLE',
            'tire_width': '195',
            'tire_aspect_ratio': '65',
            'tire_rim_size': 'R15',
            'tire_brand': 'Triangle'
        },
        {
            'name': 'TYRES 225/65R17 TRIANGLE',
            'tire_width': '225',
            'tire_aspect_ratio': '65',
            'tire_rim_size': 'R17',
            'tire_brand': 'Triangle'
        },
        {
            'name': 'TYRE 265/70R16 TRIANGLE',
            'tire_width': '265',
            'tire_aspect_ratio': '70',
            'tire_rim_size': 'R16',
            'tire_brand': 'Triangle'
        },
        {
            'name': 'TYRES 15-AP TR',
            'tire_width': '15',
            'tire_aspect_ratio': 'AP',
            'tire_rim_size': 'TR',
            'tire_brand': 'Triangle'
        },
        {
            'name': 'TYRE 185/70R14',
            'tire_width': '185',
            'tire_aspect_ratio': '70',
            'tire_rim_size': 'R14',
            'tire_brand': 'Triangle'
        }
    ]
    
    # Update existing tire products
    for update_data in tire_updates:
        product = Product.query.filter_by(name=update_data['name']).first()
        if product:
            product.tire_width = update_data['tire_width']
            product.tire_aspect_ratio = update_data['tire_aspect_ratio']
            product.tire_rim_size = update_data['tire_rim_size']
            product.tire_brand = update_data['tire_brand']
            print(f"Updated {product.name}")
    
    # Add new tire products with different brands and sizes
    new_tires = [
        {
            'name': 'MICHELIN 205/55R16',
            'description': 'Michelin brand tires, size 205/55R16, premium quality for passenger cars',
            'price': 95000,
            'category': 'tires',
            'image_url': 'https://images.unsplash.com/photo-1604176354204-92658cb65a8f?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=2070&q=80',
            'stock': 18,
            'is_featured': True,
            'tire_width': '205',
            'tire_aspect_ratio': '55',
            'tire_rim_size': 'R16',
            'tire_brand': 'Michelin'
        },
        {
            'name': 'BRIDGESTONE 215/60R17',
            'description': 'Bridgestone brand tires, size 215/60R17, excellent grip and durability',
            'price': 105000,
            'category': 'tires',
            'image_url': 'https://images.unsplash.com/photo-1604176354204-92658cb65a8f?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=2070&q=80',
            'stock': 22,
            'is_featured': True,
            'tire_width': '215',
            'tire_aspect_ratio': '60',
            'tire_rim_size': 'R17',
            'tire_brand': 'Bridgestone'
        },
        {
            'name': 'GOODYEAR 235/65R18',
            'description': 'Goodyear brand tires, size 235/65R18, all-season performance',
            'price': 125000,
            'category': 'tires',
            'image_url': 'https://images.unsplash.com/photo-1604176354204-92658cb65a8f?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=2070&q=80',
            'stock': 16,
            'is_featured': False,
            'tire_width': '235',
            'tire_aspect_ratio': '65',
            'tire_rim_size': 'R18',
            'tire_brand': 'Goodyear'
        },
        {
            'name': 'PIRELLI 245/45R18',
            'description': 'Pirelli brand tires, size 245/45R18, high-performance sports tires',
            'price': 140000,
            'category': 'tires',
            'image_url': 'https://images.unsplash.com/photo-1604176354204-92658cb65a8f?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=2070&q=80',
            'stock': 12,
            'is_featured': False,
            'tire_width': '245',
            'tire_aspect_ratio': '45',
            'tire_rim_size': 'R18',
            'tire_brand': 'Pirelli'
        },
        {
            'name': 'CONTINENTAL 195/65R15',
            'description': 'Continental brand tires, size 195/65R15, comfort and safety',
            'price': 85000,
            'category': 'tires',
            'image_url': 'https://images.unsplash.com/photo-1604176354204-92658cb65a8f?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=2070&q=80',
            'stock': 28,
            'is_featured': False,
            'tire_width': '195',
            'tire_aspect_ratio': '65',
            'tire_rim_size': 'R15',
            'tire_brand': 'Continental'
        },
        {
            'name': 'YOKOHAMA 225/50R17',
            'description': 'Yokohama brand tires, size 225/50R17, excellent handling',
            'price': 98000,
            'category': 'tires',
            'image_url': 'https://images.unsplash.com/photo-1604176354204-92658cb65a8f?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=2070&q=80',
            'stock': 20,
            'is_featured': False,
            'tire_width': '225',
            'tire_aspect_ratio': '50',
            'tire_rim_size': 'R17',
            'tire_brand': 'Yokohama'
        },
        {
            'name': 'MICHELIN 175/70R13',
            'description': 'Michelin brand tires, size 175/70R13, perfect for small city cars',
            'price': 75000,
            'category': 'tires',
            'image_url': 'https://images.unsplash.com/photo-1604176354204-92658cb65a8f?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=2070&q=80',
            'stock': 25,
            'is_featured': False,
            'tire_width': '175',
            'tire_aspect_ratio': '70',
            'tire_rim_size': 'R13',
            'tire_brand': 'Michelin'
        },
        {
            'name': 'BRIDGESTONE 255/70R16',
            'description': 'Bridgestone brand tires, size 255/70R16, ideal for SUVs and 4x4 vehicles',
            'price': 135000,
            'category': 'tires',
            'image_url': 'https://images.unsplash.com/photo-1604176354204-92658cb65a8f?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=2070&q=80',
            'stock': 14,
            'is_featured': False,
            'tire_width': '255',
            'tire_aspect_ratio': '70',
            'tire_rim_size': 'R16',
            'tire_brand': 'Bridgestone'
        }
    ]
    
    # Add new tire products
    for tire_data in new_tires:
        existing_product = Product.query.filter_by(name=tire_data['name']).first()
        if not existing_product:
            product = Product(**tire_data)
            db.session.add(product)
            print(f"Added new tire: {tire_data['name']}")
    
    db.session.commit()
    print("Tire products updated successfully!")

if __name__ == "__main__":
    print("Updating database with tire-specific fields...")
    update_database()
    with app.app_context():
        update_tire_products()
    print("Done!") 