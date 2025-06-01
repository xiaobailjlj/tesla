from database.db_handler import TeslaDatabase
from ml.train_model import CarPricePredictor
from flask import Flask, request, jsonify
from flask_cors import CORS
from flask import send_from_directory
from flasgger import Swagger
import os
import io
import csv
import logging
from typing import Dict
import traceback
import yaml
from datetime import datetime
import json
import uuid

# init config
def load_config(config_file=''):
    config_path = os.path.join(os.path.dirname(__file__), config_file)
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

if os.getenv('DOCKER_ENV'):
    config_file = './conf/config-docker.yaml'
else:
    config_file = './conf/config.yaml'

config = load_config(config_file)

# init logger
logging.basicConfig(level=getattr(logging, config['logging']['level']))
logger = logging.getLogger(__name__)

# init db
DB_CONFIG = {
    'host': config['database']['host'],
    'port': config['database']['port'],
    'user': config['database']['user'],
    'password': config['database']['password'],
    'database': config['database']['db']
}

db_handler = TeslaDatabase(**DB_CONFIG)

# init app
app = Flask(__name__)
CORS(app,
     resources=config['cors']['resources'],
     methods=config['cors']['methods'])

# init model predictor
model_path = os.path.join(os.getcwd(), 'ml', 'model')
predictor = CarPricePredictor(model_path=model_path)

# init swagger: /apidocs/
swagger = Swagger(app)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() == 'csv'


def get_db():
    if not db_handler.ensure_connection():
        raise Exception("Failed to establish database connection")
    return db_handler


def validate_car_data(data: Dict) -> tuple[bool, str]:
    required_fields = ['car_id', 'country_code', 'product_name', 'model_year',
                       'odometer', 'odometer_type', 'condition_status', 'revenue']

    # Check required fields
    for field in required_fields:
        if field not in data or data[field] is None or data[field] == '':
            return False, f"Missing required field: {field}"

    # Validate data types and ranges
    try:
        data['model_year'] = int(data['model_year'])
        if data['model_year'] < 2003 or data['model_year'] > datetime.now().year:
            return False, "Model year invalid"

        data['odometer'] = int(data['odometer'])
        if data['odometer'] <= 0:
            return False, "Odometer invalid"

        data['revenue'] = int(data['revenue'])
        if data['revenue'] <= 0:
            return False, "Revenue invalid"

    except (ValueError, TypeError):
        return False, "Invalid data types for numeric fields"

    # Validate string lengths
    if len(data['car_id']) > 64:
        return False, "Car ID too long (max 64 characters)"
    if len(data['country_code']) > 8:
        return False, "Country code too long (max 8 characters)"
    if len(data['product_name']) > 128:
        return False, "Product name too long (max 128 characters)"

    # Validate odometer type
    data['odometer_type'] = data['odometer_type'].lower()
    if data['odometer_type'] in ['km', 'kms', 'kilometer', 'kilometers']:
        data['odometer_type'] = 'km'
    elif data['odometer_type'] in ['miles', 'mile']:
        data['odometer_type'] = 'miles'
    if data['odometer_type'] not in ['km', 'miles']:
        return False, "Odometer type must be 'km' or 'miles'"

    # everything to lowercase: country_code, product_name, odometer_type, condition_status
    data['country_code'] = data['country_code'].lower()
    data['product_name'] = data['product_name'].lower()
    # data['product_trim_name'] = data.get('product_trim_name', '').lower()
    data['odometer_type'] = data['odometer_type'].lower()
    data['condition_status'] = data['condition_status'].lower()

    return True, ""

@app.route('/')
def serve_frontend():
    return send_from_directory('static', 'index.html')
@app.route('/health', methods=['GET'])
def health_check():
    """Check API health
        ---
        tags:
          - Health
        responses:
          200:
            description: Service is healthy
            schema:
              type: object
              properties:
                status:
                  type: string
                  example: healthy
                database:
                  type: string
                  example: connected
                timestamp:
                  type: string
                  example: "2025-01-01T12:00:00"
          503:
            description: Service is unhealthy
    """
    try:
        db = get_db()

        return jsonify({
            'status': 'healthy',
            'database': 'connected',
            'timestamp': datetime.now().isoformat()
        }), 200

    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return jsonify({
            'status': 'unhealthy',
            'database': 'disconnected',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 503


@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500


@app.route('/api/cars/query', methods=['GET'])
def query_cars():
    """Query cars with optional filters
        ---
        tags:
          - Cars
        parameters:
          - name: country_code
            in: query
            type: string
            description: Filter by country code
          - name: product_name
            in: query
            type: string
            description: Filter by Tesla model name
          - name: model_year
            in: query
            type: integer
            description: Filter by model year
          - name: condition_status
            in: query
            type: string
            description: Filter by vehicle condition
          - name: limit
            in: query
            type: integer
            description: Maximum number of results
          - name: offset
            in: query
            type: integer
            description: Number of results to skip
        responses:
          200:
            description: List of cars
            schema:
              type: object
              properties:
                success:
                  type: boolean
                  example: true
                data:
                  type: array
                  items:
                    type: object
                    properties:
                      car_id:
                        type: string
                      country_code:
                        type: string
                      product_name:
                        type: string
                      model_year:
                        type: integer
                      odometer:
                        type: integer
                      revenue:
                        type: integer
                count:
                  type: integer
          400:
            description: Invalid query parameters
    """
    try:
        # Get query parameters
        filters = {}
        valid_filters = ['country_code', 'product_name', 'model_year', 'condition_status']

        for filter_key in valid_filters:
            if filter_key in request.args:
                value = request.args.get(filter_key).lower().strip()  # Normalize to lowercase and strip whitespace
                if filter_key == 'model_year':
                    try:
                        filters[filter_key] = int(value)
                    except ValueError:
                        return jsonify({
                            'success': False,
                            'error': f'Invalid model_year: {value}'
                        }), 400
                else:
                    filters[filter_key] = value

        # Get pagination parameters
        limit = None
        offset = None

        if 'limit' in request.args:
            try:
                limit = int(request.args.get('limit'))
                if limit <= 0:
                    return jsonify({
                        'success': False,
                        'error': 'Limit must be a positive integer'
                    }), 400
            except ValueError:
                return jsonify({
                    'success': False,
                    'error': 'Invalid limit parameter'
                }), 400

        if 'offset' in request.args:
            try:
                offset = int(request.args.get('offset'))
                if offset < 0:
                    return jsonify({
                        'success': False,
                        'error': 'Offset must be non-negative'
                    }), 400
            except ValueError:
                return jsonify({
                    'success': False,
                    'error': 'Invalid offset parameter'
                }), 400

        db = get_db()
        cars = db.query_cars(filters, limit, offset)

        # Convert datetime objects to strings for JSON serialization
        for car in cars:
            if 'created_time' in car and car['created_time']:
                car['created_time'] = car['created_time'].isoformat()
            if 'updated_time' in car and car['updated_time']:
                car['updated_time'] = car['updated_time'].isoformat()

        return jsonify({
            'success': True,
            'data': cars,
            'count': len(cars),
            'filters_applied': filters
        }), 200

    except Exception as e:
        logger.error(f"Error querying cars: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Internal server error'
        }), 500


@app.route('/api/car/query/<car_id>', methods=['GET'])
def get_car(car_id: str):
    """Get a specific car by ID
        ---
        tags:
          - Cars
        parameters:
          - name: car_id
            in: path
            type: string
            required: true
            description: The car identifier
        responses:
          200:
            description: Car details
            schema:
              type: object
              properties:
                success:
                  type: boolean
                data:
                  type: object
                  properties:
                    car_id:
                      type: string
                    country_code:
                      type: string
                    product_name:
                      type: string
                    model_year:
                      type: integer
                    odometer:
                      type: integer
                    revenue:
                      type: integer
          404:
            description: Car not found
    """
    try:
        db = get_db()
        car = db.get_car_by_id(car_id)

        if car:
            # Convert datetime objects to strings for JSON serialization
            if 'created_time' in car and car['created_time']:
                car['created_time'] = car['created_time'].isoformat()
            if 'updated_time' in car and car['updated_time']:
                car['updated_time'] = car['updated_time'].isoformat()

            return jsonify({
                'success': True,
                'data': car
            }), 200
        else:
            return jsonify({
                'success': False,
                'error': f'Car with ID {car_id} not found'
            }), 404

    except Exception as e:
        logger.error(f"Error retrieving car {car_id}: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Internal server error'
        }), 500


@app.route('/api/car/new', methods=['POST'])
def add_car():
    """Add a new car
        ---
        tags:
          - Cars
        parameters:
          - name: car_data
            in: body
            required: true
            schema:
              type: object
              required:
                - country_code
                - product_name
                - model_year
                - odometer
                - odometer_type
                - condition_status
                - revenue
              properties:
                country_code:
                  type: string
                  description: Country code (e.g., US, DE)
                  example: "US"
                product_name:
                  type: string
                  description: Tesla model name
                  example: "model s"
                model_year:
                  type: integer
                  description: Year of manufacture
                  example: 2022
                odometer:
                  type: integer
                  description: Current odometer reading
                  example: 25000
                odometer_type:
                  type: string
                  enum: ["km", "miles"]
                  description: Odometer unit
                  example: "km"
                condition_status:
                  type: string
                  description: Vehicle condition
                  example: "excellent"
                revenue:
                  type: integer
                  description: Revenue in EUR
                  example: 45000
                currency:
                  type: string
                  description: Currency code
                  default: "EUR"
        responses:
          201:
            description: Car added successfully
          400:
            description: Invalid input data
          409:
            description: Car already exists
    """
    try:
        if not request.is_json:
            return jsonify({
                'success': False,
                'error': 'Content-Type must be application/json'
            }), 400

        car_data = request.get_json()
        car_data['currency'] = car_data.get('currency', 'EUR').upper()  # Default to EUR if not provided
        car_data['car_id'] = str(uuid.uuid4())

        # Validate the car data
        is_valid, error_message = validate_car_data(car_data)
        if not is_valid:
            return jsonify({
                'success': False,
                'error': error_message
            }), 400

        db = get_db()
        # Check if car ID already exists
        existing_car = db.get_car_by_id(car_data['car_id'])
        if existing_car:
            return jsonify({
                'success': False,
                'error': f'Car with ID {car_data["car_id"]} already exists'
            }), 409

        success = db.insert_car(car_data)

        if success:
            return jsonify({
                'success': True,
                'message': f'Car {car_data["car_id"]} added successfully',
                'car_id': car_data['car_id']
            }), 201
        else:
            return jsonify({
                'success': False,
                'error': 'Failed to add car to database'
            }), 500

    except Exception as e:
        logger.error(f"Error adding car: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Internal server error'
        }), 500


@app.route('/api/cars/new', methods=['POST'])
def batch_upload_csv():
    """Batch upload cars from CSV file
        ---
        tags:
          - Cars
        consumes:
          - multipart/form-data
        parameters:
          - name: csv_file
            in: formData
            type: file
            required: true
            description: CSV file containing car data
            x-example: |
              Expected CSV columns:
              CarID, countrycode, ProductName, ProductTrimName, ModelYear,
              CurrentOdometer, OdometerType, Condition, Revenue
        responses:
          200:
            description: CSV processing completed
            schema:
              type: object
              properties:
                success:
                  type: boolean
                  example: true
                message:
                  type: string
                  example: "CSV processing completed"
                results:
                  type: object
                  properties:
                    successful_inserts:
                      type: integer
                      example: 45
                    failed_inserts:
                      type: integer
                      example: 2
                    total_processed:
                      type: integer
                      example: 47
                    parse_errors_count:
                      type: integer
                      example: 3
                    total_rows_in_csv:
                      type: integer
                      example: 50
                parse_errors:
                  type: array
                  items:
                    type: string
                  example: ["Row 5: Missing required field: ModelYear"]
          400:
            description: Invalid file or no file provided
            schema:
              type: object
              properties:
                success:
                  type: boolean
                  example: false
                error:
                  type: string
                  example: "No file provided. Please upload a CSV file with field name 'csv_file'"
          500:
            description: Database or processing error
    """
    try:
        # Check if file is present in the request
        if 'csv_file' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No file provided. Please upload a CSV file with field name "csv_file"'
            }), 400

        file = request.files['csv_file']

        # Check if file is selected
        if file.filename == '':
            return jsonify({
                'success': False,
                'error': 'No file selected'
            }), 400

        # Check file extension
        if not allowed_file(file.filename):
            return jsonify({
                'success': False,
                'error': 'Only CSV files are allowed'
            }), 400

        # Parse CSV data directly from memory
        cars_data = []
        parse_errors = []

        try:
            # Read file content as string
            file_content = file.read().decode('utf-8')
            csv_reader = csv.DictReader(io.StringIO(file_content))

            for row_num, row in enumerate(csv_reader, start=2):
                try:
                    # Skip completely empty rows
                    if not any(row.values()):
                        continue

                    # Map CSV columns to database columns and validate
                    car_data = {
                        'car_id': row.get('CarID', '').strip(),
                        'country_code': row.get('countrycode', '').strip(),
                        'product_name': row.get('ProductName', '').strip(),
                        'product_trim_name': row.get('ProductTrimName', '').strip() or None,
                        'model_year': int(float(row.get('ModelYear', '0') or '0')),  # if ModelYear is empty, it defaults to '0' before attempting the conversion.
                        'odometer': int(float(row.get('CurrentOdometer', '0') or '0')),  # if CurrentOdometer is empty, it defaults to '0' before attempting the conversion.
                        'odometer_type': row.get('OdometerType', '').strip(),
                        'condition_status': row.get('Condition', '').strip(),
                        'revenue': int(float(row.get('Revenue', '0') or '0')),  # if Revenue is empty, it defaults to '0' before attempting the conversion.
                        'currency': 'EUR'  # Default currency
                    }

                    # Validate the car data using existing validation function
                    is_valid, error_message = validate_car_data(car_data)
                    if is_valid:
                        cars_data.append(car_data)
                    else:
                        parse_errors.append(f"Row {row_num}: {error_message}")

                except (ValueError, TypeError) as e:
                    parse_errors.append(f"Row {row_num}: Invalid data format - {str(e)}")
                    continue
                except Exception as e:
                    parse_errors.append(f"Row {row_num}: Unexpected error - {str(e)}")
                    continue

        except UnicodeDecodeError:
            return jsonify({
                'success': False,
                'error': 'File encoding error. Please ensure the CSV file is UTF-8 encoded.'
            }), 400
        except Exception as e:
            return jsonify({
                'success': False,
                'error': f'Error parsing CSV file: {str(e)}'
            }), 400

        # Check if we have any valid data
        if not cars_data:
            return jsonify({
                'success': False,
                'error': 'No valid car data found in CSV file',
                'parse_errors': parse_errors,
                'total_parse_errors': len(parse_errors)
            }), 400

        # Process the parsed data with database
        try:
            db = get_db()
            successful, failed = db.batch_insert_cars(cars_data)

            response_data = {
                'success': True,
                'message': 'CSV processing completed',
                'results': {
                    'successful_inserts': successful,
                    'failed_inserts': failed,
                    'total_processed': successful + failed,
                    'parse_errors_count': len(parse_errors),
                    'total_rows_in_csv': successful + failed + len(parse_errors)
                }
            }

            # Include parse errors if any (but limit to avoid large responses)
            if parse_errors:
                response_data['parse_errors'] = parse_errors

            return jsonify(response_data), 200

        except Exception as db_error:
            logger.error(f"Database error during batch insert: {str(db_error)}")
            return jsonify({
                'success': False,
                'error': 'Database error occurred during batch processing',
                'parsed_successfully': len(cars_data),
                'parse_errors_count': len(parse_errors)
            }), 500

    except Exception as e:
        logger.error(f"Error processing CSV upload: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': 'Internal server error during file processing'
        }), 500


@app.route('/api/car/predict', methods=['POST'])
def price_predict():
    """Predict car price based on characteristics
        ---
        tags:
          - Prediction
        parameters:
          - name: prediction_data
            in: body
            required: true
            schema:
              type: object
              required:
                - countrycode
                - ProductName
                - ModelYear
                - CurrentOdometer
                - OdometerType
                - Condition
              properties:
                countrycode:
                  type: string
                  example: "US"
                ProductName:
                  type: string
                  example: "model s"
                ModelYear:
                  type: integer
                  example: 2022
                CurrentOdometer:
                  type: integer
                  example: 25000
                OdometerType:
                  type: string
                  enum: ["km", "miles"]
                  example: "km"
                Condition:
                  type: string
                  example: "excellent"
                CarID:
                  type: string
                  description: Optional car ID for prediction
                  example: "PREDICT-001"
        responses:
          200:
            description: Price prediction result
            schema:
              type: object
              properties:
                success:
                  type: boolean
                prediction:
                  type: object
                  properties:
                    predicted_price:
                      type: number
                      example: 45000.50
                    currency:
                      type: string
                      example: "EUR"
                    confidence:
                      type: number
                      example: 0.85
          400:
            description: Invalid input data
          503:
            description: ML model not available
    """
    try:
        if not request.is_json:
            return jsonify({
                'success': False,
                'error': 'Content-Type must be application/json'
            }), 400

        car_data = request.get_json()

        required_fields = ['countrycode', 'ProductName', 'ModelYear', 'CurrentOdometer', 'OdometerType', 'Condition']

        for field in required_fields:
            if field not in car_data or car_data[field] is None or car_data[field] == '':
                return jsonify({
                    'success': False,
                    'error': f'Missing required field for prediction: {field}'
                }), 400

        car_data['CarID'] = car_data.get('CarID', 'PREDICT-001')
        car_data['Revenue'] = 0  # Dummy value

        try:
            predicted_price, confidence = predictor.predict_price(car_data)

            return jsonify({
                'success': True,
                'prediction': {
                    'predicted_price': round(float(predicted_price), 2),
                    'currency': 'EUR',
                    'confidence': confidence
                },
                'input_data': {
                    'country_code': car_data['countrycode'],
                    'product_name': car_data['ProductName'],
                    'model_year': car_data['ModelYear'],
                    'odometer': car_data['CurrentOdometer'],
                    'odometer_type': car_data['OdometerType'],
                    'condition': car_data['Condition']
                }
            }), 200

        except FileNotFoundError:
            return jsonify({
                'success': False,
                'error': 'Model not found. Please train the model first.'
            }), 503
        except ValueError as ve:
            return jsonify({
                'success': False,
                'error': f'Prediction failed: {str(ve)}'
            }), 400
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            return jsonify({
                'success': False,
                'error': 'Internal error during prediction'
            }), 500

    except Exception as e:
        logger.error(f"Error in price prediction endpoint: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Internal server error'
        }), 500



if __name__ == '__main__':
    server_config = config['server']
    app.run(
        debug=server_config['debug'],
        host=server_config['host'],
        port=server_config['port']
    )