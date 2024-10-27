# Delivergate Data Engineer Intern Practical Exam

### By: Gayandee Rajapaksha

## Project Overview
This project involves setting up a MySQL database, creating tables for customers and orders, and implementing a Streamlit application for data analysis and visualization. The main goal is to analyze customer purchasing behavior and predict repeat purchasers based on historical data.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Database Setup](#database-setup)
- [Running the Application](#running-the-application)
- [Loading the Data](#loading-the-data)

## Prerequisites
Before running the application, ensure you have the following installed on your local machine:
- Python 3.x
- MySQL Server
- MySQL Workbench
- Required Python libraries:
  ```bash
  pip install pandas sqlalchemy pymysql streamlit scikit-learn matplotlib seaborn plotly
  ```

## Database Setup

### 1. Create the Schema
1. Open MySQL Workbench.
2. Connect to your MySQL server.
3. Manually created the database schema:

### 2. Create Tables
Once the schema is created, use the following SQL commands to create the required tables:

```sql
USE delivergate;

CREATE TABLE customers (
    customer_id INT PRIMARY KEY AUTO_INCREMENT,
    customer_name VARCHAR(255) NOT NULL
);

CREATE TABLE orders (
    order_id INT PRIMARY KEY AUTO_INCREMENT,
    customer_id INT,
    total_amount DECIMAL(10, 2),
    order_date DATE,
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
);
```

### 3. Verify Tables
You can verify that the tables have been created successfully by running:

```sql
SHOW TABLES;
```

## Running the Application
1. Open a terminal or command prompt.
2. Navigate to the directory where your Streamlit application code is located.
3. Run the application using the following command:

   ```bash
   streamlit run StreamlitML.py
   ```

   Replace `StreamlitML.py` with the name of your Python file.

4. Open your web browser and go to `http://localhost:8501` to view the application.

## Loading the Data
Ensure you have the `customers.csv` and `orders.csv` files in the same directory as your application code.

The application includes the code to load the data into the MySQL database. It connects to the database (using your DB password) and imports the CSV files:

```python
import pandas as pd
from sqlalchemy import create_engine

# Load data
customers_df = pd.read_csv('customers.csv')
orders_df = pd.read_csv('order.csv')

# Create database connection
cnx = create_engine('mysql+pymysql://root:<password>@localhost:3306/delivergate')

# Import data into MySQL
customers_df.to_sql('customers', con=cnx, if_exists='replace', index=False)
orders_df.to_sql('orders', con=cnx, if_exists='replace', index=False)
```

After running the application, you can analyze the customer orders and purchase patterns directly through the navigations called Dashboard and Data Analysis in the sidebar.
