import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor  # Changed from RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # Added StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import json
from datetime import datetime
import joblib
import os


class CarPricePredictor:
    def __init__(self, model_path='./model'):
        self.model = None
        self.onehot_encoders = {}
        self.scaler = None  # Added for neural network
        self.feature_columns = []
        self.training_stats = {}
        self.model_path = model_path
        self._model_loaded = False

    def clean_data(self, df):
        print("Starting data cleaning with integrated validation...")
        print(f"Initial dataset shape: {df.shape}")

        valid_rows = []
        validation_errors = []

        for idx, row in df.iterrows():
            # validate required fields
            required_fields = ['CarID', 'countrycode', 'ProductName', 'ModelYear',
                               'CurrentOdometer', 'OdometerType', 'Condition', 'Revenue']

            missing_fields = []
            for field in required_fields:
                if field not in row or pd.isna(row[field]) or row[field] == '':
                    missing_fields.append(field)

            if missing_fields:
                validation_errors.append(f"Row {idx}: Missing fields {missing_fields}")
                continue

            # validate data
            try:
                # model_year
                model_year = int(float(row['ModelYear']))
                if model_year < 2003 or model_year > datetime.now().year:
                    validation_errors.append(f"Row {idx}: Unusual model year {model_year}")
                    continue
                df.loc[idx, 'ModelYear'] = model_year

                # odometer
                odometer = float(row['CurrentOdometer'])
                if odometer < 0 or odometer > 1000000:
                    validation_errors.append(f"Row {idx}: Unusual odometer {odometer}")
                    continue
                df.loc[idx, 'CurrentOdometer'] = odometer

                # odometer_type
                odometer_type = str(row['OdometerType']).strip().lower()
                odometer_mapping = {
                    'km': 'km', 'kms': 'km', 'kilometer': 'km', 'kilometers': 'km',
                    'miles': 'miles', 'mile': 'miles'
                }
                if odometer_type not in odometer_mapping:
                    validation_errors.append(f"Row {idx}: Invalid odometer type '{odometer_type}'")
                    continue
                df.loc[idx, 'OdometerType'] = odometer_mapping[odometer_type]

                # revenue
                revenue = float(row['Revenue'])
                revenue_q99 = df['Revenue'].quantile(0.99)  # 99% of the data
                revenue_q01 = df['Revenue'].quantile(0.01)  # 1% of the data

                if revenue < revenue_q01 or revenue > revenue_q99:
                    validation_errors.append(f"Row {idx}: Unusual revenue {revenue}")
                    continue
                df.loc[idx, 'Revenue'] = revenue

            except (ValueError, TypeError):
                validation_errors.append(f"Row {idx}: Invalid data types")
                continue

            # if valid
            valid_rows.append(idx)

        # all valid rows
        df_clean = df.loc[valid_rows].copy()

        print(f"Validation results:")
        print(f"  Valid rows: {len(valid_rows)}")
        print(f"  Invalid rows: {len(validation_errors)}")
        if validation_errors:
            print("  Validation errors:")
            for error in validation_errors:
                print(f"    {error}")

        # convert miles to km
        miles_mask = df_clean['OdometerType'] == 'miles'
        if miles_mask.any():
            print(f"Converting {miles_mask.sum()} records from miles to km")
            df_clean.loc[miles_mask, 'CurrentOdometer'] = df_clean.loc[miles_mask, 'CurrentOdometer'] * 1.60934
            df_clean.loc[miles_mask, 'OdometerType'] = 'km'

        # convert condition to numeric values (some with typos)
        condition_mapping = {
            'like_new': 5,
            'excellent': 4,
            'good': 3,
            'fair': 2,
            'poor': 1,

            'god': 3,
            'goo': 3,
            'far': 2,
            'fir': 2,
            'fnair': 2,
            'mfair': 2,
            'bgood': 3,
            'ygood': 3,
        }
        df_clean['Condition'] = df_clean['Condition'].astype(str).str.lower().str.strip()
        df_clean['ConditionNumeric'] = df_clean['Condition'].map(condition_mapping)

        # use car age as a feature, instead of ModelYear
        df_clean['CarAge'] = datetime.now().year - df_clean['ModelYear']

        # lowercase
        text_fields = ['countrycode', 'ProductName']
        for field in text_fields:
            df_clean[field] = df_clean[field].astype(str).str.lower()

        print(f"Data cleaning complete. Final shape: {df_clean.shape}")
        return df_clean

    def prepare_features(self, df, is_training=True):
        import numpy as np

        df_features = df.copy()

        categorical_features = ['countrycode', 'ProductName']
        numerical_features = ['ModelYear', 'CurrentOdometer', 'CarAge', 'ConditionNumeric']

        # One-Hot Encode categorical variables
        all_encoded_features = []

        for feature in categorical_features:
            if is_training:
                # Create and fit OneHotEncoder
                ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                feature_data = df_features[feature].astype(str).values.reshape(-1, 1)
                encoded_array = ohe.fit_transform(feature_data)

                # Store the encoder
                if not hasattr(self, 'onehot_encoders'):
                    self.onehot_encoders = {}
                self.onehot_encoders[feature] = ohe

                # Create column names
                feature_names = [f'{feature}_{class_name}' for class_name in ohe.categories_[0]]

                # Add encoded columns to dataframe
                for i, col_name in enumerate(feature_names):
                    df_features[col_name] = encoded_array[:, i]

                all_encoded_features.extend(feature_names)

                print(f"\n=== {feature.upper()} ONE-HOT ENCODING ===")
                print(f"Original unique values: {len(ohe.categories_[0])}")
                print(f"Created {len(feature_names)} binary columns:")
                for i, (original, col_name) in enumerate(zip(ohe.categories_[0][:5], feature_names[:5])):
                    print(f"  '{original}' → {col_name}")
                if len(feature_names) > 5:
                    print(f"  ... and {len(feature_names) - 5} more columns")

            else:
                # Use existing encoder for prediction
                if hasattr(self, 'onehot_encoders') and feature in self.onehot_encoders:
                    ohe = self.onehot_encoders[feature]
                    feature_data = df_features[feature].astype(str).values.reshape(-1, 1)
                    encoded_array = ohe.transform(feature_data)

                    # Get the same column names as training
                    feature_names = [f'{feature}_{class_name}' for class_name in ohe.categories_[0]]

                    # Add encoded columns to dataframe
                    for i, col_name in enumerate(feature_names):
                        df_features[col_name] = encoded_array[:, i]

                    all_encoded_features.extend(feature_names)
                else:
                    # If encoder doesn't exist, create dummy columns (all zeros)
                    print(f"Warning: No encoder found for {feature}, creating dummy columns")
                    dummy_col = f'{feature}_unknown'
                    df_features[dummy_col] = 0
                    all_encoded_features.append(dummy_col)

        # Update feature columns to include all one-hot encoded features
        self.feature_columns = numerical_features + all_encoded_features

        # Get final feature matrix
        X = df_features[self.feature_columns]

        # Scale features for neural network (IMPORTANT!)
        if is_training:
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
            print(f"\n=== FEATURE SCALING ===")
            print(f"Scaled {X.shape[1]} features for neural network")
            print(f"Feature means after scaling: {X_scaled.mean(axis=0)[:5]}")  # Should be ~0
            print(f"Feature stds after scaling: {X_scaled.std(axis=0)[:5]}")  # Should be ~1
        else:
            if hasattr(self, 'scaler') and self.scaler is not None:
                X_scaled = self.scaler.transform(X)
            else:
                print("Warning: No scaler found, using unscaled features")
                X_scaled = X.values

        return pd.DataFrame(X_scaled, columns=self.feature_columns, index=X.index)

    def save_model(self):
        print(f"\nSaving model...")

        model_file = os.path.join(self.model_path, 'car_price_model.pkl')
        encoders_file = os.path.join(self.model_path, 'onehot_encoders.pkl')
        scaler_file = os.path.join(self.model_path, 'scaler.pkl')  # Added scaler
        features_file = os.path.join(self.model_path, 'feature_columns.json')
        stats_file = os.path.join(self.model_path, 'training_stats.json')

        # Create directory if it doesn't exist
        os.makedirs(self.model_path, exist_ok=True)

        joblib.dump(self.model, model_file)
        # Save onehot_encoders and scaler
        if hasattr(self, 'onehot_encoders'):
            joblib.dump(self.onehot_encoders, encoders_file)
        if hasattr(self, 'scaler') and self.scaler is not None:
            joblib.dump(self.scaler, scaler_file)

        with open(features_file, 'w') as f:
            json.dump(self.feature_columns, f, indent=2)

        with open(stats_file, 'w') as f:
            json.dump(self.training_stats, f, indent=2)

        print("Model saved successfully!")

    def load_model(self):
        """Load the trained model and associated files"""
        if self._model_loaded:
            return True

        try:
            # Check if model files exist
            model_file = os.path.join(self.model_path, 'car_price_model.pkl')
            encoders_file = os.path.join(self.model_path, 'onehot_encoders.pkl')
            scaler_file = os.path.join(self.model_path, 'scaler.pkl')  # Added scaler
            features_file = os.path.join(self.model_path, 'feature_columns.json')
            stats_file = os.path.join(self.model_path, 'training_stats.json')

            required_files = [model_file, encoders_file, scaler_file, features_file]
            if not all(os.path.exists(f) for f in required_files):
                missing = [f for f in required_files if not os.path.exists(f)]
                raise FileNotFoundError(f"Required model files not found: {missing}")

            # Load model components
            self.model = joblib.load(model_file)
            self.onehot_encoders = joblib.load(encoders_file)
            self.scaler = joblib.load(scaler_file)  # Load scaler

            with open(features_file, 'r') as f:
                self.feature_columns = json.load(f)

            # Load training stats if available
            if os.path.exists(stats_file):
                with open(stats_file, 'r') as f:
                    self.training_stats = json.load(f)

            self._model_loaded = True
            print(f"Model loaded successfully from {self.model_path}")
            return True

        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise e

    def is_model_loaded(self):
        """Check if model is loaded"""
        return self._model_loaded and self.model is not None

    def predict_price(self, car_features: dict):
        """Predict price for a single car with integrated validation"""
        # Auto-load model if not loaded
        if not self.is_model_loaded():
            self.load_model()

        if self.model is None:
            raise ValueError("Model not loaded!")

        # Create single-row DataFrame
        df_input = pd.DataFrame([car_features])

        # Apply same cleaning logic (but for single row)
        try:
            # Clean and validate the input
            df_clean = self.clean_data(df_input)

            if len(df_clean) == 0:
                raise ValueError("Input data failed validation")

            # Prepare features (includes scaling)
            X_pred = self.prepare_features(df_clean, is_training=False)

            # Make prediction
            prediction = self.model.predict(X_pred)[0]

            # Calculate confidence (neural networks don't have built-in uncertainty)
            confidence = "High" if 20000 <= prediction <= 200000 else "Medium"

            return prediction, confidence

        except Exception as e:
            raise ValueError(f"Prediction failed: {str(e)}")


def plot_data_analysis(df_clean):
    """Create simple plots: revenue frequency and revenue vs features"""
    import matplotlib.pyplot as plt

    # Create figures directory
    os.makedirs('./figure', exist_ok=True)

    # 1. Revenue Frequency Distribution (separate figure)
    plt.figure(figsize=(10, 6))
    plt.hist(df_clean['Revenue'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    plt.title('Revenue Frequency Distribution', fontsize=14, fontweight='bold')
    plt.xlabel('Revenue ($)')
    plt.ylabel('Frequency')
    plt.ticklabel_format(style='plain', axis='x')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('./figure/revenue_frequency.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("Revenue frequency plot saved as 'revenue_frequency.png'")

    # 2. Revenue vs All Features (separate figure)
    numeric_features = [
        ('ModelYear', 'Model Year'),
        ('CurrentOdometer', 'Odometer (km)'),
        ('CarAge', 'Car Age (years)'),
        ('ConditionNumeric', 'Condition (1=Poor, 5=Like New)')
    ]

    plt.figure(figsize=(15, 10))

    for i, (feature, label) in enumerate(numeric_features, 1):
        plt.subplot(2, 2, i)
        plt.scatter(df_clean[feature], df_clean['Revenue'], alpha=0.6, s=20)
        plt.title(f'Revenue vs {label}', fontsize=14, fontweight='bold')
        plt.xlabel(label)
        plt.ylabel('Revenue ($)')
        plt.ticklabel_format(style='plain', axis='y')
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('./figure/revenue_vs_features.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("Revenue vs features plot saved as 'revenue_vs_features.png'")

    # 3. Revenue vs Categorical Features (separate figure)
    plt.figure(figsize=(15, 10))

    # Revenue vs Country Code
    plt.subplot(2, 2, 1)
    country_stats = df_clean.groupby('countrycode')['Revenue'].mean().sort_values(ascending=True)
    plt.barh(range(len(country_stats)), country_stats.values, alpha=0.7)
    plt.title('Average Revenue by Country Code', fontsize=14, fontweight='bold')
    plt.xlabel('Average Revenue ($)')
    plt.ylabel('Country Code')
    plt.yticks(range(len(country_stats)), country_stats.index)
    plt.ticklabel_format(style='plain', axis='x')
    plt.grid(True, alpha=0.3, axis='x')

    # Revenue vs Product Name (top 15 most common)
    plt.subplot(2, 2, 2)
    product_counts = df_clean['ProductName'].value_counts()
    top_products = product_counts.head(15).index
    product_stats = df_clean[df_clean['ProductName'].isin(top_products)].groupby('ProductName')[
        'Revenue'].mean().sort_values(ascending=True)

    plt.barh(range(len(product_stats)), product_stats.values, alpha=0.7)
    plt.title('Average Revenue by Product Name (Top 15)', fontsize=14, fontweight='bold')
    plt.xlabel('Average Revenue ($)')
    plt.ylabel('Product Name')
    plt.yticks(range(len(product_stats)), product_stats.index, fontsize=10)
    plt.ticklabel_format(style='plain', axis='x')
    plt.grid(True, alpha=0.3, axis='x')

    # Country Code scatter plot (using encoded values as x-axis)
    plt.subplot(2, 2, 3)
    # Create temporary encoding for visualization
    country_codes = df_clean['countrycode'].unique()
    country_mapping = {code: i for i, code in enumerate(sorted(country_codes))}
    df_clean['country_encoded_temp'] = df_clean['countrycode'].map(country_mapping)

    plt.scatter(df_clean['country_encoded_temp'], df_clean['Revenue'], alpha=0.6, s=20)
    plt.title('Revenue vs Country Code (Scatter)', fontsize=14, fontweight='bold')
    plt.xlabel('Country Code (Encoded)')
    plt.ylabel('Revenue ($)')
    plt.xticks(range(len(country_codes)), sorted(country_codes), rotation=45)
    plt.ticklabel_format(style='plain', axis='y')
    plt.grid(True, alpha=0.3)

    # Product Name scatter plot (using encoded values as x-axis, top 10 only)
    plt.subplot(2, 2, 4)
    top_10_products = product_counts.head(10).index
    df_top_products = df_clean[df_clean['ProductName'].isin(top_10_products)]
    product_mapping = {prod: i for i, prod in enumerate(sorted(top_10_products))}
    df_top_products = df_top_products.copy()
    df_top_products['product_encoded_temp'] = df_top_products['ProductName'].map(product_mapping)

    plt.scatter(df_top_products['product_encoded_temp'], df_top_products['Revenue'], alpha=0.6, s=20)
    plt.title('Revenue vs Product Name (Top 10, Scatter)', fontsize=14, fontweight='bold')
    plt.xlabel('Product Name (Encoded)')
    plt.ylabel('Revenue ($)')
    plt.xticks(range(len(top_10_products)), sorted(top_10_products), rotation=45, ha='right')
    plt.ticklabel_format(style='plain', axis='y')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('./figure/revenue_vs_categorical.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("Revenue vs categorical features plot saved as 'revenue_vs_categorical.png'")


if __name__ == "__main__":
    dataset_path = './dataset/raw_dataset.csv'
    predictor = CarPricePredictor()

    try:
        # Load data
        print("\n" + "=" * 10 + "Load data" + "=" * 10)
        print(f"Loading data from {dataset_path}...")
        df = pd.read_csv(dataset_path)
        print(f"Loaded {len(df)} rows")

        # Clean data
        print("\n" + "=" * 10 + "Clean data" + "=" * 10)
        df_clean = predictor.clean_data(df)
        os.makedirs('./dataset', exist_ok=True)
        df_clean.to_csv('./dataset/cleaned_dataset.csv', index=False)
        print(f"Cleaned data saved to 'cleaned_dataset.csv'")

        # Data analysis
        print("\n" + "=" * 10 + "Data Analysis" + "=" * 10)
        plot_data_analysis(df_clean)

        # Train model: preparation
        print("\n" + "=" * 10 + "Train Model: preparation" + "=" * 10)
        X = predictor.prepare_features(df_clean, is_training=True)
        y = df_clean['Revenue']

        print(f"Feature matrix shape: {X.shape}")
        print(f"Features: {predictor.feature_columns}")

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")

        # Train MLP Neural Network model
        print("\n" + "=" * 10 + "Train Model: MLPRegressor training" + "=" * 10)
        predictor.model = MLPRegressor(
            hidden_layer_sizes=(100, 50),  # Two hidden layers: 100 and 50 neurons
            activation='relu',  # ReLU activation function
            solver='adam',  # Adam optimizer
            alpha=0.001,  # L2 regularization
            learning_rate='adaptive',  # Adaptive learning rate
            max_iter=500,  # Maximum iterations
            random_state=42,
            early_stopping=True,  # Stop when validation score stops improving
            validation_fraction=0.1,  # 10% for validation during training
            n_iter_no_change=20  # Stop after 20 iterations without improvement
        )

        print("Training neural network...")
        predictor.model.fit(X_train, y_train)
        print("Model training complete!")
        print(f"Training iterations: {predictor.model.n_iter_}")
        print(f"Final loss: {predictor.model.loss_:.4f}")

        # Evaluate model
        print("\n" + "=" * 10 + "Train Model: evaluation" + "=" * 10)
        y_train_pred = predictor.model.predict(X_train)
        y_test_pred = predictor.model.predict(X_test)

        # Calculate metrics
        train_mae = mean_absolute_error(y_train, y_train_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)

        predictor.training_stats = {
            'train_mae': train_mae,
            'test_mae': test_mae,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'features': predictor.feature_columns,
            'model_type': 'MLPRegressor',
            'hidden_layers': predictor.model.hidden_layer_sizes,
            'n_iterations': predictor.model.n_iter_,
            'final_loss': predictor.model.loss_
        }

        print(f"\nMODEL PERFORMANCE:")
        print(f"Training R2: {train_r2:.4f}")
        print(f"Test R2:     {test_r2:.4f}")
        print(f"Training MAE: ${train_mae:,.0f}")
        print(f"Test MAE:     ${test_mae:,.0f}")

        # Neural networks don't have feature_importances_, so we skip that section
        print(f"\nNEURAL NETWORK INFO:")
        print(f"Hidden layers: {predictor.model.hidden_layer_sizes}")
        print(
            f"Total parameters: ~{sum(layer.size for layer in predictor.model.coefs_) + sum(layer.size for layer in predictor.model.intercepts_)}")
        print(f"Activation function: {predictor.model.activation}")
        print(f"Solver: {predictor.model.solver}")

        # Save model and encoders
        predictor.save_model()

        # Test prediction
        print("\n" + "=" * 10 + "Test Prediction" + "=" * 10)
        sample_car = {
            'CarID': 'TEST-001',
            'countrycode': 'DE',
            'ProductName': 'Model 3',
            'ProductTrimName': 'Long Range',
            'ModelYear': 2023,
            'CurrentOdometer': 15000,
            'OdometerType': 'km',
            'Condition': 'good',
            'Revenue': 50000
        }

        try:
            price, confidence = predictor.predict_price(sample_car)
            print(f"Sample prediction: ${price:,.2f} (Confidence: {confidence})")
        except Exception as e:
            print(f"Prediction test failed: {e}")

    except Exception as e:
        print(f"Process failed: {e}")
        import traceback

        traceback.print_exc()