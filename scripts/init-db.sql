-- NeuralRetail PostgreSQL initialization script
-- Creates all required databases for the project services

-- Airflow metadata database
SELECT 'CREATE DATABASE airflow'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'airflow')\gexec

-- MLflow backend store database
SELECT 'CREATE DATABASE mlflow'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'mlflow')\gexec

-- Grant privileges to nr_user on all databases
GRANT ALL PRIVILEGES ON DATABASE airflow TO nr_user;
GRANT ALL PRIVILEGES ON DATABASE mlflow TO nr_user;
