# Tesla Used Car Resale Platform

Platform for managing Tesla used car and providing price predictions for trade-ins.

## Quick Start

### Docker

```bash
# Clone the repository
git clone https://github.com/xiaobailjlj/tesla.git
cd tesla

# Start the application
cd docker/
docker-compose up --build

# Access the application
# Frontend: http://127.0.0.1:7700/
# API: http://127.0.0.1:7700/api
# API Documentation: http://127.0.0.1:7700/apidocs/
```

### Local Development (Mac)

**Prerequisites**: Python 3.8+, MySQL

1. **Database Setup**
   ```bash
   # Start MariaDB
   brew services start mariadb
   
   # Connect as root and create tesla user and database
   mysql -h127.0.0.1 -uroot -p
   
   # Init MySQL script
   ./database/init.sql
   ```

2. **Python Environment**
   ```bash
   # Create virtual environment
   python -m venv .venv
   source .venv/bin/activate
   
   # Install dependencies
   pip install -r requirements.txt
   
   # Run the application
   python app.py
   
   # Access the application at 
   # http://localhost:7700
   ```

## Tech Stack

**Backend**
- Python Flask with RESTful API
- MySQL for data storage
- scikit-learn for machine learning models

**Frontend**
- simple HTML


## Configuration

The application uses YAML configuration files:

- `conf/config.yaml` - Local development settings
- `conf/config-docker.yaml` - Docker deployment settings


## API Documentation

The application includes interactive Swagger documentation available at:

docker: `http://127.0.0.1:7700/apidocs/`

local: `http://localhost:7700/apidocs/`

## API Endpoints

### Health Check
- `GET /health`

### Car Listings
- `GET /api/car/query/{car_id}` - Get car by ID
- `POST /api/car/new` - Add a car
- `POST /api/cars/new` - Batch upload cars from CSV file
- `GET /api/cars/query` - Query cars with optional filters

### Price Predictions
- `POST /api/car/predict` - Get price prediction for vehicle characteristics based on the historical data

### Testing the API

```bash
# Health check
curl http://127.0.0.1:7700/health

# Add a single car listing
curl -X POST http://127.0.0.1:7700/api/car/new \
  -H "Content-Type: application/json" \
  -d '{
    "country_code": "CA",
    "product_name": "Model 3",
    "product_trim_name": "aaaa",
    "model_year": 2024,
    "odometer": 5000,
    "odometer_type": "km",
    "condition_status": "good",
    "revenue": 55000
  }'
  
# Query a specific car by ID
curl "http://127.0.0.1:7700/api/car/query/CAR-00005"

# Upload CSV file
curl -X POST http://127.0.0.1:7700/api/cars/new \
  -F "csv_file=@cars_data.csv"

# Query cars with filters
curl "http://127.0.0.1:7700/api/cars/query?country_code=SE&product_name=Model Y&model_year=2023&condition_status=FAIR"

# Get price prediction
curl -X POST http://127.0.0.1:7700/api/car/predict \
  -H "Content-Type: application/json" \
  -d '{
    "countrycode": "DE",
    "ProductName": "Model 3",
    "ModelYear": 2021,
    "CurrentOdometer": 15000,
    "OdometerType": "km",
    "Condition": "good"
  }'
```

## Machine Learning for Price Prediction

### Data Processing & Analysis
**Data Source:**
- Dummy dataset of past used car resales

**Data Cleaning Pipeline:**
- Unit standardization (miles to kilometers)
- Remove outlier
- Feature encoding
- Missing data handling

**Key Findings:**
- Country code shows strongest correlation with pricing
- Other features shows weak correlation with pricing (strange)

### Model Performance

**Random Forest Regression**
- R² Score: 0.9
- Issue: Only highly related to country code, not other features

**Neural Networks (MLPRegressor)**
- R² Score: nearly 0

**Possible Cause:**
- Dataset
- Categorical encoding

## TODO

- SQL injection protection
- Authentication
- Improve model performance

## Troubleshooting
**Docker Issues**
```bash
cd docker/

# Stop and clean up
docker-compose down -v

# Rebuild containers
docker-compose up --build
docker-compose up

# Check container logs
docker-compose logs web
docker-compose logs db

# Connect to MySQL container
docker-compose exec db mysql -u tesla -p
```
