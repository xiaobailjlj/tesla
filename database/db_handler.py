import mysql.connector
from mysql.connector import Error
from typing import List, Dict, Optional, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TeslaDatabase:
    def __init__(self, host: str, database: str, user: str, password: str, port: int = 3306):
        """
        Initialize database connection parameters
        """
        self.host = host
        self.port = port
        self.database = database
        self.user = user
        self.password = password
        self.connection = None

    def connect(self):
        """
        Establish database connection
        """
        try:
            self.connection = mysql.connector.connect(
                host=self.host,
                database=self.database,
                user=self.user,
                password=self.password,
                port=self.port
            )
            if self.connection.is_connected():
                logger.info("Successfully connected to MySQL database")
                return True
        except Error as e:
            logger.error(f"Error connecting to MySQL: {e}")
            return False

    def disconnect(self):
        """
        Close database connection
        """
        if self.connection and self.connection.is_connected():
            self.connection.close()
            logger.info("MySQL connection closed")

    def is_connected(self) -> bool:
        """
        Check if database connection is active
        """
        return self.connection and self.connection.is_connected()

    def ensure_connection(self):
        """
        Ensure database connection is active, reconnect if needed
        """
        if not self.is_connected():
            logger.info("Connection lost, attempting to reconnect...")
            return self.connect()
        try:
            self.connection.ping(reconnect=True)
            return True
        except Error as e:
            logger.error(f"Connection ping failed: {e}")
            return self.connect()

    def insert_car(self, car_data: Dict) -> bool:
        """
        Insert a single car record

        Args:
            car_data (Dict): Dictionary containing car information
                Required keys: car_id, country_code, product_name, model_year,
                              odometer, odometer_type, condition_status, revenue, currency
                Optional keys: product_trim_name

        Returns:
            bool: True if successful, False otherwise
        """
        if not self.ensure_connection():
            return False

        try:
            cursor = self.connection.cursor()

            query = """
            INSERT INTO cars (
                car_id, country_code, product_name, product_trim_name,
                model_year, odometer, odometer_type, condition_status,
                revenue, currency
            ) VALUES (%(car_id)s, %(country_code)s, %(product_name)s, %(product_trim_name)s,
                     %(model_year)s, %(odometer)s, %(odometer_type)s, %(condition_status)s,
                     %(revenue)s, %(currency)s)
            """

            car_data.setdefault('product_trim_name', None)

            cursor.execute(query, car_data)
            self.connection.commit()

            logger.info(f"Successfully inserted car with ID: {car_data['car_id']}")
            cursor.close()
            return True

        except Error as e:
            logger.error(f"Error inserting car: {e}")
            if self.connection:
                self.connection.rollback()
            return False

    def get_car_by_id(self, car_id: str) -> Optional[Dict]:
        """
        Retrieve a car by its ID

        Args:
            car_id (str): The car ID to search for

        Returns:
            Optional[Dict]: Car data if found, None otherwise
        """
        if not self.ensure_connection():
            return None

        try:
            cursor = self.connection.cursor(dictionary=True)

            query = "SELECT * FROM cars WHERE car_id = %s"
            cursor.execute(query, (car_id,))

            result = cursor.fetchone()
            cursor.close()

            if result:
                logger.info(f"Found car with ID: {car_id}")
                return result
            else:
                logger.info(f"No car found with ID: {car_id}")
                return None

        except Error as e:
            logger.error(f"Error retrieving car by ID: {e}")
            return None

    def batch_insert_cars(self, cars_data: List[Dict]) -> Tuple[int, int]:
        """
        Insert multiple cars in batch

        Args:
            cars_data: List of car dictionaries

        Returns:
            Tuple[int, int]: (successful_inserts, failed_inserts)
        """
        successful = 0
        failed = 0

        for car_data in cars_data:
            try:
                if self.insert_car(car_data):
                    successful += 1
                else:
                    failed += 1
            except Exception as e:
                logger.error(f"Failed to insert car {car_data.get('car_id', 'unknown')}: {e}")
                failed += 1

        return successful, failed

    def delete_car(self, car_id: str) -> bool:
        """
        Delete a car by its ID

        Args:
            car_id (str): The car ID to delete

        Returns:
            bool: True if successful, False otherwise
        """
        if not self.ensure_connection():
            return False

        try:
            cursor = self.connection.cursor()

            query = "DELETE FROM cars WHERE car_id = %s"
            cursor.execute(query, (car_id,))

            if cursor.rowcount > 0:
                self.connection.commit()
                logger.info(f"Successfully deleted car with ID: {car_id}")
                cursor.close()
                return True
            else:
                logger.warning(f"No car found with ID: {car_id}")
                cursor.close()
                return False

        except Error as e:
            logger.error(f"Error deleting car: {e}")
            if self.connection:
                self.connection.rollback()
            return False

    def query_cars(self, filters: Dict = None, limit: int = None, offset: int = None) -> List[Dict]:
        """
        Query cars with optional filters

        Args:
            filters (Dict): Optional filters (e.g., {'country_code': 'US', 'model_year': 2020})
            limit (int): Optional limit for results
            offset (int): Optional offset for pagination

        Returns:
            List[Dict]: List of car records
        """
        if not self.ensure_connection():
            return []

        try:
            cursor = self.connection.cursor(dictionary=True)

            query = "SELECT * FROM cars"
            params = []

            # Add WHERE clause if filters provided
            if filters:
                conditions = []
                for key, value in filters.items():
                    conditions.append(f"{key} = %s")
                    params.append(value)
                query += " WHERE " + " AND ".join(conditions)

            # Add ORDER BY for consistent results
            query += " ORDER BY created_time DESC"

            # Add LIMIT and OFFSET if provided
            if limit:
                query += " LIMIT %s"
                params.append(limit)
                if offset:
                    query += " OFFSET %s"
                    params.append(offset)

            cursor.execute(query, params)
            results = cursor.fetchall()
            cursor.close()

            logger.info(f"Query returned {len(results)} cars")
            return results

        except Error as e:
            logger.error(f"Error querying cars: {e}")
            return []