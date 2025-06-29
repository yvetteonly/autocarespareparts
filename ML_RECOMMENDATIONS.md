# 🤖 Machine Learning Recommendation System

## Overview

This auto parts e-commerce platform now includes a sophisticated **Machine Learning Recommendation System** that provides personalized product recommendations to users based on their behavior and preferences.

## 🧠 AI Features

### 1. **Content-Based Filtering**
- **Algorithm**: TF-IDF (Term Frequency-Inverse Document Frequency) + Cosine Similarity
- **Features**: Product names, descriptions, and categories
- **Purpose**: Recommends products similar to what users have viewed or purchased
- **Implementation**: `recommendation_system.get_content_based_recommendations()`

### 2. **Collaborative Filtering**
- **Algorithm**: Non-negative Matrix Factorization (NMF)
- **Features**: User purchase history and ratings
- **Purpose**: Finds users with similar preferences and recommends products they liked
- **Implementation**: `recommendation_system.get_collaborative_recommendations()`

### 3. **Hybrid Recommendations**
- **Algorithm**: Weighted combination of content-based and collaborative filtering
- **Weights**: 60% collaborative + 40% content-based
- **Purpose**: Combines the best of both approaches for optimal recommendations
- **Implementation**: `recommendation_system.get_hybrid_recommendations()`

### 4. **Popular Products**
- **Algorithm**: Purchase frequency analysis
- **Purpose**: Shows trending products based on sales data
- **Implementation**: `recommendation_system.get_popular_products()`

### 5. **Category-Based Recommendations**
- **Algorithm**: Category filtering
- **Purpose**: Suggests products from the same category
- **Implementation**: `recommendation_system.get_category_based_recommendations()`

## 🚀 API Endpoints

### Get Personalized Recommendations
```http
GET /recommendations
```
**Response:**
```json
{
  "success": true,
  "recommendations": [
    {
      "id": 1,
      "name": "Product Name",
      "description": "Product description",
      "price": 50000,
      "category": "tires",
      "image_url": "https://...",
      "score": 0.85,
      "type": "personalized"
    }
  ]
}
```

### Get Product-Specific Recommendations
```http
GET /recommendations/product/{product_id}
```
**Response:**
```json
{
  "success": true,
  "recommendations": [
    {
      "id": 2,
      "name": "Similar Product",
      "description": "Similar product description",
      "price": 45000,
      "category": "tires",
      "image_url": "https://...",
      "similarity_score": 0.92
    }
  ]
}
```

### Get Category Recommendations
```http
GET /recommendations/category/{category}
```
**Response:**
```json
{
  "success": true,
  "recommendations": [
    {
      "id": 3,
      "name": "Category Product",
      "description": "Product from same category",
      "price": 60000,
      "category": "tires",
      "image_url": "https://...",
      "category_score": 1.0
    }
  ]
}
```

### Rebuild Recommendation Models (Admin)
```http
GET /admin/recommendations/rebuild
```

## 🛠️ Technical Implementation

### Dependencies
```python
scikit-learn==1.3.0
pandas==2.0.3
numpy==1.24.3
```

### Core Classes

#### RecommendationSystem
```python
class RecommendationSystem:
    def __init__(self):
        self.tfidf_vectorizer = None
        self.product_similarity_matrix = None
        self.user_product_matrix = None
        self.nmf_model = None
        self.user_factors = None
        self.product_factors = None
```

### Key Methods

#### Content-Based Filtering
```python
def build_content_based_recommendations(self):
    # Creates TF-IDF vectors from product descriptions
    # Calculates cosine similarity matrix
    # Stores similarity scores for fast retrieval
```

#### Collaborative Filtering
```python
def build_collaborative_filtering(self):
    # Creates user-product matrix from purchase history
    # Applies NMF to find latent factors
    # Enables user-based recommendations
```

#### Hybrid Recommendations
```python
def get_hybrid_recommendations(self, user_id, product_id=None, n_recommendations=5):
    # Combines content-based and collaborative filtering
    # Applies weights: 60% collaborative, 40% content-based
    # Returns personalized recommendations
```

## 🎯 Frontend Integration

### Homepage Recommendations
- **Location**: Homepage for authenticated users
- **Trigger**: Automatic on page load
- **Display**: Grid of recommended products with AI badges
- **Features**: Add to cart functionality, loading states

### Product Detail Recommendations
- **Location**: Product detail pages
- **Trigger**: Based on current product
- **Display**: Similar products section
- **Features**: Content-based similarity scores

### Admin Dashboard
- **Location**: Admin panel
- **Features**: Model rebuilding, system status
- **Controls**: Manual model retraining

## 📊 Data Sources

### User Behavior Data
- Purchase history (orders and order items)
- Product views and interactions
- Wishlist additions
- Cart additions

### Product Data
- Product names and descriptions
- Categories and pricing
- Stock levels and availability
- Featured product flags

### Interaction Data
- User-product interactions
- Purchase quantities
- Order completion status

## 🔧 Configuration

### Model Parameters
```python
# TF-IDF Configuration
max_features=1000
stop_words='english'
ngram_range=(1, 2)

# NMF Configuration
n_components=10  # Latent factors
random_state=42

# Recommendation Limits
content_based_limit=5
collaborative_limit=5
hybrid_limit=8
popular_limit=4
```

### Thresholds
```python
# Similarity thresholds
content_similarity_threshold=0.1
collaborative_rating_threshold=0.5

# Weights for hybrid recommendations
collaborative_weight=0.6
content_weight=0.4
```

## 🚀 Performance Optimization

### Caching Strategy
- **Similarity Matrix**: Pre-computed and cached
- **User Factors**: Computed on-demand and cached
- **Popular Products**: Cached with periodic updates

### Scalability Features
- **Lazy Loading**: Models built only when needed
- **Batch Processing**: Efficient matrix operations
- **Memory Management**: Optimized data structures

## 📈 Monitoring & Analytics

### Model Performance
- Recommendation accuracy tracking
- User engagement metrics
- Click-through rates on recommendations

### System Health
- Model build success rates
- API response times
- Error logging and monitoring

## 🔮 Future Enhancements

### Planned Features
1. **Real-time Learning**: Update models based on new interactions
2. **A/B Testing**: Compare different recommendation algorithms
3. **Seasonal Adjustments**: Account for seasonal product preferences
4. **Price Sensitivity**: Factor in user price preferences
5. **Cross-selling**: Recommend complementary products

### Advanced Algorithms
1. **Deep Learning**: Neural network-based recommendations
2. **Graph Neural Networks**: User-product relationship modeling
3. **Multi-modal Recommendations**: Text + image similarity
4. **Contextual Bandits**: Real-time optimization

## 🛡️ Privacy & Security

### Data Protection
- **Anonymization**: User data anonymized for model training
- **Consent**: Clear user consent for recommendation tracking
- **GDPR Compliance**: Right to opt-out of recommendations

### Security Measures
- **Input Validation**: Sanitized user inputs
- **Rate Limiting**: API request throttling
- **Access Control**: Admin-only model management

## 📚 Usage Examples

### Basic Usage
```python
# Get personalized recommendations for a user
recommendations = recommendation_system.get_hybrid_recommendations(
    user_id=123, 
    n_recommendations=5
)

# Get similar products
similar_products = recommendation_system.get_content_based_recommendations(
    product_id=456, 
    n_recommendations=3
)
```

### Frontend Integration
```javascript
// Load recommendations on homepage
async function loadRecommendations() {
    const response = await fetch('/recommendations');
    const data = await response.json();
    
    if (data.success) {
        displayRecommendations(data.recommendations);
    }
}
```

## 🎉 Benefits

### For Users
- **Personalized Experience**: Tailored product suggestions
- **Discovery**: Find new products they might like
- **Convenience**: Faster product discovery
- **Relevance**: Higher quality recommendations

### For Business
- **Increased Sales**: Higher conversion rates
- **Customer Retention**: Better user engagement
- **Inventory Optimization**: Popular product insights
- **Competitive Advantage**: Advanced AI capabilities

---

**Developed by**: SIBOMANA Yvette  
**Technology Stack**: Python, Flask, scikit-learn, pandas, numpy  
**Project**: Auto Parts E-commerce Platform with ML Recommendations 