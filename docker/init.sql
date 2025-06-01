USE TESLA;

CREATE TABLE cars (
    car_id VARCHAR(64) PRIMARY KEY,
    country_code VARCHAR(8) NOT NULL,
    product_name VARCHAR(128) NOT NULL,
    product_trim_name VARCHAR(128),
    model_year INT NOT NULL,
    odometer INT NOT NULL,
    odometer_type ENUM('km', 'miles') NOT NULL DEFAULT 'km',
    condition_status VARCHAR(16) NOT NULL,
    revenue INT NOT NULL,
    currency VARCHAR(4) NOT NULL DEFAULT 'EUR',
    created_time DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_time DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,

    INDEX idx_country_code (country_code),
    INDEX idx_product_name (product_name),
    INDEX idx_model_year (model_year),
    INDEX idx_odometer (odometer),
    INDEX idx_condition (condition_status)
);
