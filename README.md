# Sandcy Ltd - Flask E-commerce Application

A modern e-commerce web application for automotive parts and tires built with Flask, SQLite, and Jinja2 templates.

## Features

- **User Authentication**: Register, login, and logout functionality
- **Product Catalog**: Browse products by category with filtering
- **Shopping Cart**: Add/remove items, update quantities
- **Order Management**: Complete checkout process and order history
- **Responsive Design**: Modern UI with Tailwind CSS and dark mode support
- **Database**: SQLite database with SQLAlchemy ORM
- **Admin Features**: Product management and order tracking

## Technology Stack

- **Backend**: Flask (Python web framework)
- **Database**: SQLite with SQLAlchemy ORM
- **Frontend**: HTML, Tailwind CSS, JavaScript
- **Authentication**: Flask-Login
- **Templates**: Jinja2
- **Icons**: Font Awesome

## Installation

1. **Clone the repository** (if applicable) or navigate to the project directory

2. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**:
   ```bash
   python app.py
   ```

4. **Access the application**:
   Open your web browser and go to `http://localhost:5000`

## Database Setup

The application automatically creates the SQLite database (`autoparts.db`) and populates it with sample products when you first run the application.

## Sample Data

The application comes with sample products including:
- **Tires**: Triangle brand tires in various sizes (195/65R15, 225/65R17, 265/70R16, etc.)
- **Engine Oils**: 15W40 and 5W30 synthetic oils in different grades
- **Services**: Oil filter replacement, tire repair, mounting, wheel alignment, A/C service
- **Batteries**: High-quality car batteries
- **Additional Services**: Fuel cleaning, valve replacement, and more

All products are based on real-world auto parts and services commonly found in automotive invoices.

## User Registration

To test the application:
1. Click "Register" in the navigation
2. Fill out the registration form
3. Login with your credentials
4. Start shopping!

## Admin Access

To create an admin user:
1. Register a regular user account
2. Manually update the database to set `is_admin = True` for the user
3. Or use the admin panel to promote existing users to admin status

## Features Overview

### For Customers:
- **Browse Products**: View all products or filter by category
- **Product Details**: Detailed product pages with images and specifications
- **Shopping Cart**: Add items, update quantities, remove items
- **Checkout**: Complete purchase with shipping and payment information
- **Order History**: View past orders and their status
- **User Profile**: Manage account information

### For Administrators:
- **Product Management**: Add, edit, and remove products
- **Order Management**: View and update order status
- **User Management**: View user accounts and order history

## Project Structure

```
sandcy-ltd autopartspro/
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── templates/            # Jinja2 templates
│   ├── base.html         # Base template
│   ├── index.html        # Home page
│   ├── login.html        # Login page
│   ├── register.html     # Registration page
│   ├── products.html     # Product catalog
│   ├── product_detail.html # Product detail page
│   ├── cart.html         # Shopping cart
│   ├── checkout.html     # Checkout page
│   ├── profile.html      # User profile
│   └── orders.html       # Order history
└── autoparts.db          # SQLite database (created automatically)
```

## Database Models

- **User**: User accounts with authentication
- **Product**: Product catalog with categories
- **Order**: Customer orders
- **OrderItem**: Individual items in orders

## Security Features

- Password hashing with Werkzeug
- Session management with Flask-Login
- CSRF protection
- Input validation and sanitization

## Customization

### Adding New Products:
1. Modify the `create_sample_products()` function in `app.py`
2. Add new product data to the products list
3. Restart the application

### Styling:
- The application uses Tailwind CSS for styling
- Custom CSS can be added to the `<style>` section in `base.html`
- Dark mode is supported and can be toggled

### Database Changes:
- Modify the model classes in `app.py`
- Delete the existing `autoparts.db` file
- Restart the application to recreate the database

## Deployment

For production deployment:

1. **Change the secret key** in `app.py`:
   ```python
   app.config['SECRET_KEY'] = 'your-secure-secret-key-here'
   ```

2. **Use a production database** like PostgreSQL or MySQL

3. **Set up a production WSGI server** like Gunicorn:
   ```bash
   pip install gunicorn
   gunicorn -w 4 -b 0.0.0.0:8000 app:app
   ```

4. **Configure environment variables** for sensitive data

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is open source and available under the MIT License.

## Support

For questions or issues, please create an issue in the repository or contact the development team. 