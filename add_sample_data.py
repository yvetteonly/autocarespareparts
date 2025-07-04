#!/usr/bin/env python3
"""
Script to add sample data for generating recommendations
"""

from app import app, db, User, Product, Order, OrderItem, Wishlist
from werkzeug.security import generate_password_hash
from datetime import datetime, timedelta
import random

def create_sample_users():
    """Create sample users with different preferences"""
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
    return created_users

def create_sample_orders():
    """Create sample orders with different product preferences"""
    users = User.query.filter_by(is_admin=False).all()
    products = Product.query.all()
    
    if not users or not products:
        print("No users or products found. Please ensure users and products exist first.")
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

def create_sample_wishlists():
    """Create sample wishlist items"""
    users = User.query.filter_by(is_admin=False).all()
    products = Product.query.all()
    
    if not users or not products:
        print("No users or products found.")
        return
    
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

def add_sample_data():
    """Main function to add all sample data"""
    with app.app_context():
        print("Adding sample data for recommendations...")
        print("=" * 50)
        
        # Create sample users
        print("\n1. Creating sample users...")
        users = create_sample_users()
        
        # Create sample orders
        print("\n2. Creating sample orders...")
        create_sample_orders()
        
        # Create sample wishlists
        print("\n3. Creating sample wishlists...")
        create_sample_wishlists()
        
        print("\n" + "=" * 50)
        print("Sample data creation completed!")
        print("\nNow you can:")
        print("1. Login as any of the created users")
        print("2. Visit the homepage to see AI recommendations")
        print("3. Check the admin dashboard to rebuild recommendations")
        
        # Print user credentials
        print("\nSample user credentials:")
        print("Username: john_mechanic, Password: password123")
        print("Username: sarah_driver, Password: password123")
        print("Username: mike_enthusiast, Password: password123")
        print("Username: lisa_commuter, Password: password123")
        print("Username: david_racer, Password: password123")

if __name__ == "__main__":
    add_sample_data()
 