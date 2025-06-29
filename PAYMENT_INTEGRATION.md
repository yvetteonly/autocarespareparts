# Payment Integration Guide

This document explains how to set up and use the payment gateways integrated into the Sandcy Ltd auto parts e-commerce platform.

## Payment Methods Supported

### 1. MoMoPay (Mobile Money)
- **Status**: Simulated for demo purposes
- **Features**: 
  - Mobile money payment simulation
  - Payment confirmation flow
  - Order status tracking

### 2. Flutterwave (Card Payments)
- **Status**: Integrated with test environment
- **Features**:
  - Credit/Debit card payments
  - Secure payment processing
  - Webhook handling for payment confirmation

## Setup Instructions

### 1. Environment Configuration

Create a `.env` file in the project root with the following variables:

```env
# Flask Configuration
SECRET_KEY=your-super-secret-key-change-this-in-production
FLASK_DEBUG=True

# Database Configuration
DATABASE_URL=sqlite:///autoparts.db

# Flutterwave Configuration (Get from Flutterwave Dashboard)
FLUTTERWAVE_PUBLIC_KEY=FLWPUBK_TEST-your-test-public-key
FLUTTERWAVE_SECRET_KEY=FLWSECK_TEST-your-test-secret-key
FLUTTERWAVE_WEBHOOK_SECRET=your-webhook-secret

# MoMoPay Configuration (For future integration)
MOMOPAY_API_KEY=your-momopay-api-key
MOMOPAY_SECRET_KEY=your-momopay-secret-key
```

### 2. Flutterwave Setup

1. **Create Flutterwave Account**:
   - Sign up at [Flutterwave Dashboard](https://dashboard.flutterwave.com)
   - Complete account verification

2. **Get API Keys**:
   - Navigate to Settings > API Keys
   - Copy your Public Key and Secret Key
   - Update the `.env` file with these keys

3. **Configure Webhooks** (Optional):
   - Set webhook URL: `https://yourdomain.com/payment_webhook`
   - Add webhook secret to `.env` file

### 3. MoMoPay Integration (Future)

For production MoMoPay integration:
1. Contact MoMoPay for API credentials
2. Update the payment processing logic in `app.py`
3. Replace simulation with actual API calls

## Usage

### For Customers

1. **Add items to cart**
2. **Proceed to checkout**
3. **Select payment method**:
   - **MoMoPay**: Mobile money payment
   - **Card**: Credit/Debit card payment
4. **Complete payment**
5. **Receive order confirmation**

### For Developers

#### Testing Payments

**MoMoPay Testing**:
- Use the "Simulate MoMoPay Payment" button
- Payment is simulated with a 3-second delay
- Order status changes to "completed" after simulation

**Flutterwave Testing**:
- Use test card numbers from Flutterwave documentation
- Test cards work in sandbox environment
- Real payments require live API keys

#### Test Card Numbers

```
Visa: 4000 0000 0000 0002
Mastercard: 5204 8300 0000 2514
```

## API Endpoints

### Payment Processing

- `POST /process_momopay_payment` - Process MoMoPay payment
- `POST /process_card_payment` - Process card payment
- `POST /payment_webhook` - Handle Flutterwave webhooks
- `GET /momopay_payment_page/<order_id>` - MoMoPay payment page
- `POST /confirm_momopay_payment/<order_id>` - Confirm MoMoPay payment

### Request/Response Examples

**MoMoPay Payment Request**:
```json
{
  "first_name": "John",
  "last_name": "Doe",
  "email": "john@example.com",
  "phone": "+250123456789",
  "address": "123 Main St",
  "city": "Kigali",
  "state": "Kigali",
  "zip_code": "12345",
  "momopay_phone": "+250123456789"
}
```

**Card Payment Request**:
```json
{
  "first_name": "John",
  "last_name": "Doe",
  "email": "john@example.com",
  "phone": "+250123456789",
  "address": "123 Main St",
  "city": "Kigali",
  "state": "Kigali",
  "zip_code": "12345",
  "card_number": "4000000000000002",
  "expiry": "12/25",
  "cvv": "123",
  "card_name": "John Doe"
}
```

## Security Considerations

1. **Environment Variables**: Never commit API keys to version control
2. **HTTPS**: Use HTTPS in production for secure payment processing
3. **Webhook Verification**: Implement proper webhook signature verification
4. **Input Validation**: Validate all payment data before processing
5. **Error Handling**: Implement proper error handling for failed payments

## Troubleshooting

### Common Issues

1. **Payment Failed**:
   - Check API keys are correct
   - Verify webhook URL is accessible
   - Check server logs for errors

2. **Webhook Not Received**:
   - Verify webhook URL is correct
   - Check firewall settings
   - Ensure server is accessible from internet

3. **Test Cards Not Working**:
   - Ensure using test environment keys
   - Use correct test card numbers
   - Check Flutterwave dashboard for errors

### Debug Mode

Enable debug mode to see detailed error messages:
```env
FLASK_DEBUG=True
```

## Production Deployment

1. **Update API Keys**: Use live API keys from payment providers
2. **SSL Certificate**: Install SSL certificate for HTTPS
3. **Webhook URL**: Update webhook URL to production domain
4. **Error Monitoring**: Set up error monitoring and logging
5. **Backup**: Implement database backup strategy

## Support

For payment-related issues:
1. Check Flutterwave documentation
2. Review server logs
3. Contact payment provider support
4. Check this documentation for common solutions 