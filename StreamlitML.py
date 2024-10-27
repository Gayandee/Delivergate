import pandas as pd
import numpy as np
import streamlit as st
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt


# This code shows the title and layout configuration.
st.set_page_config(page_title="Customer Purchase", layout="centered")

# Thus creates a connection to the MySQL database as before
engine = create_engine('mysql+pymysql://root:<password>@localhost:3306/delivergate')

# These SQL queries to select all data from the customers and orders tables.
customers_query = "SELECT * FROM customers;"
orders_query = "SELECT * FROM orders;"
customers_df = pd.read_sql(customers_query, con=engine)
orders_df = pd.read_sql(orders_query, con=engine)

# Used sidebar for navigation and used a radio button allows users to switch between 
# the "Dashboard" and "Data Analysis" pages which completes 2nd and 3rd parts respectively.
st.sidebar.header("Navigation")
page = st.sidebar.radio("Select Page", ("Dashboard", "Data Analysis"))

if page == "Dashboard":
    # 01 Sidebar Filters
    st.sidebar.header("Filters")

    # Converted the order_date column in the orders DataFrame to a datetime format. 
    # This feature engineering approch will effect for the accuracy.
    orders_df['order_date'] = pd.to_datetime(orders_df['order_date'])
    
    # Get the min and max dates
    min_date = orders_df['order_date'].min()
    max_date = orders_df['order_date'].max()

    # Ensure min_date and max_date are not None
    if min_date is None or max_date is None:
        st.error("No valid dates available in the dataset.")
    else:
        # Get the date range from the user
        date_range = st.sidebar.date_input("Select Date Range", [min_date.date(), max_date.date()])

        # Validated that the date range selected is in the correct format (a list or tuple with two items)
        if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
            start_date, end_date = date_range

            start_date = pd.to_datetime(start_date)
            end_date = pd.to_datetime(end_date)
        else:
            st.error("Please select a valid date range.")

    # Total amount slider
    min_amount, max_amount = st.sidebar.slider("Select Total Amount Range",
                                                float(0),
                                                float(orders_df['total_amount'].max()),
                                                (float(0), float(orders_df['total_amount'].max())))

    # Customer order count dropdown
    order_count = st.sidebar.number_input("Minimum Number of Orders", min_value=1, value=5)

    # Filtered the orders DataFrame to include only the rows that match the selected date range and total amount criteria.
    filtered_orders = orders_df[(orders_df['order_date'] >= start_date) & 
                                 (orders_df['order_date'] <= end_date) &
                                 (orders_df['total_amount'].between(min_amount, max_amount))]

    #Counting how many orders each customer has made and filtering the orders DataFrame 
    # to include only those customers who meet the minimum order count specified by the user.
    customer_counts = filtered_orders['customer_id'].value_counts()
    filtered_customers = customer_counts[customer_counts >= int(order_count)].index
    filtered_orders = filtered_orders[filtered_orders['customer_id'].isin(filtered_customers)]

    # 02 Main dashboard
    st.title("Customer Orders Dashboard")

    # Display filtered data
    st.subheader("Filtered Orders Data")
    st.dataframe(filtered_orders)

    # Metrics
    total_revenue = filtered_orders['total_amount'].sum()
    unique_customers = filtered_orders['customer_id'].nunique()
    total_orders = filtered_orders.shape[0]

    st.subheader("Key Metrics")
    st.write(f"Total Revenue: ${total_revenue:.2f}")
    st.write(f"Number of Unique Customers: {unique_customers}")
    st.write(f"Total Orders: {total_orders}")

    # Top 10 customers by total revenue
    top_customers = filtered_orders.groupby('customer_id')['total_amount'].sum().nlargest(10)
    st.subheader("Top 10 Customers by Total Revenue")
    st.bar_chart(top_customers)

    # Total revenue over time (grouped by month)
    revenue_over_time = filtered_orders.groupby(pd.Grouper(key='order_date', freq='M'))['total_amount'].sum()
    st.subheader("Total Revenue Over Time")
    st.line_chart(revenue_over_time)

elif page == "Data Analysis":
    st.title("Data Analysis Section")
    
    # Setted a color palette for the visualizations
    sns.set_palette("Set2")

    # Summarized the orders data to get the total number of orders and revenue per customer.
    order_summary = orders_df.groupby('customer_id').agg(
        total_orders=('order_id', 'count'),
        total_revenue=('total_amount', 'sum')
    ).reset_index()

    # Create the target variable called repeat purchaser
    order_summary['repeat_purchaser'] = np.where(order_summary['total_orders'] > 1, 1, 0)

    # Features(x) and target variable(y)
    X = order_summary[['total_orders', 'total_revenue']]
    y = order_summary['repeat_purchaser']

    # Validation check for sufficient data
    if len(X) < 10:
        st.error("Not enough data to train the model.")
    else:
        # Split data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train the logistic regression model
        model = LogisticRegression()
        model.fit(X_train, y_train)

        # Predictions
        y_pred = model.predict(X_test)

        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)

        # Display results
        st.subheader("Logistic Regression Model Results")
        st.markdown(f"<h3 style='color: green;'>Model Accuracy: {accuracy:.2f}</h3>", unsafe_allow_html=True)

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Repeat', 'Repeat'], yticklabels=['Not Repeat', 'Repeat'], ax=ax)
        plt.title('Confusion Matrix', fontsize=16)
        plt.xlabel('Predicted', fontsize=14)
        plt.ylabel('Actual', fontsize=14)
        plt.tight_layout()
        st.pyplot(fig)

        st.subheader("Classification Report:")

        # Generated the classification report as a DataFrame
        report = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()

        # Displayed the report as a grid in Streamlit
        st.dataframe(report_df)

        # Total Orders Distribution
        orders_chart = alt.Chart(order_summary).mark_bar().encode(
            x=alt.X('total_orders:Q', bin=True, title='Total Orders'),
            y=alt.Y('count():Q', title='Frequency'),
            color=alt.Color('repeat_purchaser:N', scale=alt.Scale(domain=[0, 1], range=['red', 'green']), title='Repeat Purchaser')
        ).properties(
            width=600,
            height=400,
            title='Total Orders Distribution by Repeat Purchasers'
        )

        st.altair_chart(orders_chart, use_container_width=True)

        # Total Revenue Distribution
        revenue_chart = alt.Chart(order_summary).mark_bar().encode(
            x=alt.X('total_revenue:Q', bin=True, title='Total Revenue'),
            y=alt.Y('count():Q', title='Frequency'),
            color=alt.Color('repeat_purchaser:N', scale=alt.Scale(domain=[0, 1], range=['red', 'green']), title='Repeat Purchaser')
        ).properties(
            width=600,
            height=400,
            title='Total Revenue Distribution by Repeat Purchasers'
        )

        st.altair_chart(revenue_chart, use_container_width=True)


        # By this we can input new data for prediction
        st.subheader("Predict New Customer's Repeat Purchase Status")
        new_orders = st.number_input("Total Orders", min_value=1)
        new_revenue = st.number_input("Total Revenue", min_value=0.0)

        if st.button("Predict"):
            new_data = np.array([[new_orders, new_revenue]])
            prediction = model.predict(new_data)
            st.markdown(f"<h4 style='color: {'red' if prediction[0] == 0 else 'green'};'>Repeat Purchaser Status (1 = Yes, 0 = No): {prediction[0]}</h4>", unsafe_allow_html=True)


# Run the app
if __name__ == '__main__':
    st.write("Gayandee Rajapaksha!")
