import numpy as np
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import plotly as py
import plotly.graph_objects as go
from st_aggrid import AgGrid, GridOptionsBuilder
from datetime import datetime

# Configure logging
LOG_FILE = 'user_actions.log'
logging.basicConfig(filename=LOG_FILE, level=logging.INFO, format='%(asctime)s - %(message)s')

# Load data
DATA_PATH = 'cleaned_final_data.csv'
try:
    st.session_state.df = pd.read_csv(DATA_PATH)
except FileNotFoundError:
        st.error("Data file not found. Please ensure the file is in the specified directory.")
        st.stop()



def df(data):
    """
    Simulates Streamlit's st.dataframe() using Plotly for interactive table display.

    Parameters:
        data (pd.DataFrame): The DataFrame to display.
    """
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame.")

    # Create the Plotly Table
    fig = go.Figure(data=[go.Table(
        header=dict(
            values=list(data.columns),  # Column headers
            fill_color='paleturquoise',  # Header background color
            align='left',  # Align text to the left
            font=dict(size=14, color='black')  # Font size and color for headers
        ),
        cells=dict(
            values=[data[col] for col in data.columns],  # Cell values for each column
            fill_color='lavender',  # Cell background color
            align='left',  # Align text to the left
            font=dict(size=12, color='black')  # Font size and color for cells
        )
    )])

    # Show the table in the output
    fig.show()


# Example Usage
data = {
    'Name': ['Alice', 'Bob', 'Charlie'],
    'Age': [24, 27, 22],
    'City': ['New York', 'Los Angeles', 'Chicago']
}
df_example = pd.DataFrame(data)
df(df_example)

df = st.session_state.df

# Set up page
st.set_page_config(page_title="Call Center Dashboard", layout="wide")

# Title
st.markdown("<h1 style='text-align: center; color: #FF6347;'>Call Center Data Dashboard</h1>", unsafe_allow_html=True)

# Sidebar Navigation
st.sidebar.title("Navigation")
tab_choice = st.sidebar.radio("Go to", [
    "📋 View Data", "🔍 Filter Data", "➕ Add Data",
    "✏️ Edit/Delete Data", "📈 Visualizations", "📉 Correlations"])

# View Data Tab
if tab_choice == "📋 View Data":
    st.subheader("Full Dataset")
    if "df" in st.session_state:
        gb = GridOptionsBuilder.from_dataframe(st.session_state.df)
        gb.configure_pagination(paginationAutoPageSize=True)
        gb.configure_default_column(editable=True, groupable=True)
        gridOptions = gb.build()
        AgGrid(st.session_state.df, gridOptions=gridOptions, height=400, theme="material")
    else: 
         st.error("No data available. Please add or load data.")

# Filter Data Tab
elif tab_choice == "🔍 Filter Data":
    st.subheader("Filter Data")

    # Filter by Month
    if 'Month' in st.session_state.df.columns:
        st.write("### Filter by Month")
        month = st.selectbox("Select Month", options=st.session_state.df['Month'].unique())
        if st.button("Apply Month Filter"):
            filtered_month = st.session_state.df[st.session_state.df['Month'] == month]
            st.write(f"Data for {month}:")
            AgGrid(filtered_month, height=300, theme="streamlit")
            logging.info(f"Filtered data for month: {month}")
    else:
        st.error("Column 'Month' not found in dataset.")

    # Filter by Range
    st.write("### Filter by Range")
    numeric_columns = st.session_state.df.select_dtypes(include=[np.number]).columns
    if len(numeric_columns) > 0:
        column = st.selectbox("Select a column to filter:", options=numeric_columns)
        min_value = st.number_input(f"Min value for {column}:", value=float(st.session_state.df[column].min()))
        max_value = st.number_input(f"Max value for {column}:", value=float(st.session_state.df[column].max()))
        if st.button("Apply Range Filter"):
            filtered_range = st.session_state.df[(st.session_state.df[column] >= min_value) & (st.session_state.df[column] <= max_value)]
            st.write(f"Data for {column} in range {min_value} - {max_value}:")
            AgGrid(filtered_range, height=300, theme="streamlit")
            logging.info(f"Filtered data for {column} in range {min_value} - {max_value}")
    else:
        st.error("No numeric columns available for filtering.")
        
# Add Data Tab
elif tab_choice == "➕ Add Data":
    st.subheader("Add New Entry")
    with st.form("Add Entry Form"):
        new_entry = {}
        for col in st.session_state.df.columns:
            default_value = str(st.session_state.df[col].iloc[0]) if not st.session_state.df[col].isnull().all() else ""
            new_entry[col] = st.text_input(f"Enter value for {col}:", value=default_value)
        submit_add = st.form_submit_button("Add Entry")

    if submit_add:
        try:
            for col in st.session_state.df.columns:
                if st.session_state.df[col].dtype in [np.float64, np.int64]:
                    new_entry[col] = float(new_entry[col])
            new_row = pd.DataFrame([new_entry])
            st.session_state.df = pd.concat([st.session_state.df, new_row], ignore_index=True)
            st.success("New entry added successfully!")
            logging.info(f"Added new entry: {new_entry}")
        except ValueError:
            st.error("Please ensure all numeric fields are filled with valid numbers.")

# Edit/Delete Data Tab
elif tab_choice == "✏️ Edit/Delete Data":
    st.subheader("Edit or Delete Data")
    row_to_edit = st.number_input("Enter the row index to edit or delete:", min_value=0, max_value=len(st.session_state.df)-1, step=1)
    action = st.radio("Select Action", ["Edit", "Delete"])

    if action == "Edit":
        with st.form("Edit Row Form"):
            updated_values = {}
            for col in st.session_state.df.columns:
                updated_values[col] = st.text_input(f"Update value for {col}:", value=str(st.session_state.df.loc[row_to_edit, col]))
            submit_edit = st.form_submit_button("Update Row")

        if submit_edit:
            try:
                for col, value in updated_values.items():
                    if st.session_state.df[col].dtype in [np.float64, np.int64]:
                        updated_values[col] = float(value)
                    st.session_state.df.at[row_to_edit, col] = updated_values[col]
                st.success(f"Row {row_to_edit} updated successfully!")
                logging.info(f"Edited row {row_to_edit} with values: {updated_values}")
            except ValueError:
                st.error("Please ensure all numeric fields are filled with valid numbers.")

    elif action == "Delete":
        if st.button("Confirm Delete"):
            st.session_state.df = st.session_state.df.drop(index=row_to_edit).reset_index(drop=True)
            st.success(f"Row {row_to_edit} deleted successfully!")
            logging.info(f"Deleted row {row_to_edit}.")


# Visualizations Tab
elif tab_choice == "📈 Visualizations":
    st.subheader("Visualize Data")
    logging.info("User viewed visualizations.")  # Log action

   # Check if data exists
    if "df" in st.session_state:
        sample_data = st.session_state.df.sample(n=min(400, len(st.session_state.df)), random_state=55)
        if 'CallsAbandoned' in sample_data.columns and 'CallsOffered' in sample_data.columns:
            sample_data['AbandonRate'] = (
                sample_data['CallsAbandoned'] / sample_data['CallsOffered'].replace(0, np.nan)
            ) * 100

    # 1. Call Abandonment Rate by VHT Status
    try:
        if 'VHT' in st.session_state.df.columns and 'AbandonRate' in sample_data.columns:
            st.write("### Call Abandonment Rate by VHT Status")
            fig1, ax1 = plt.subplots(figsize=(10, 6))
            sns.boxplot(
                x='VHT', y='AbandonRate', data=sample_data, hue='VHT', ax=ax1
            )
            ax1.set_title("Call Abandonment Rate by VHT Status")
            ax1.set_ylabel("Abandonment Rate (%)")
            ax1.set_xlabel("Virtual Hold Technology (On/Off)")
            st.pyplot(fig1)
        else:
            st.error("Required columns for 'VHT' or 'AbandonRate' are missing.")
    except Exception as e:
        st.error(f"Error while creating boxplot for Call Abandonment Rate by VHT: {e}")

    # 2. Calls Abandoned vs ASA
    try:
        if 'ASA' in st.session_state.df.columns and 'CallsAbandoned' in st.session_state.df.columns:
            st.write("### Calls Abandoned vs Average Speed of Answer (ASA)")
            fig2, ax2 = plt.subplots(figsize=(10, 6))
            sns.scatterplot(
                x='ASA', y='CallsAbandoned', hue='VHT', data=sample_data, ax=ax2)
            sns.regplot(
                x='ASA', y='CallsAbandoned', data=sample_data,
                scatter=False, line_kws={'color': 'black', 'lw': 2}, ax=ax2)
            ax2.set_title("Calls Abandoned vs Average Speed of Answer")
            ax2.set_xlabel("Average Speed of Answer (minutes)")
            ax2.set_ylabel("Calls Abandoned")
            st.pyplot(fig2)
        else:
            st.error("Required columns for 'ASA' or 'CallsAbandoned' are missing.")
    except Exception as e:
        st.error(f"Error while creating scatterplot for Calls Abandoned vs ASA: {e}")

    # 3. Calls Offered vs Calls Abandoned
    try:
        if 'CallsOffered' in st.session_state.df.columns and 'CallsAbandoned' in st.session_state.df.columns:
            st.write("### Calls Offered vs Calls Abandoned")
            fig3, ax3 = plt.subplots(figsize=(10, 6))
            sns.scatterplot(
                x='CallsOffered', y='CallsAbandoned', hue='VHT', data=sample_data, ax=ax3
            )
            ax3.set_title("Calls Offered vs Calls Abandoned")
            ax3.set_xlabel("Calls Offered")
            ax3.set_ylabel("Calls Abandoned")
            st.pyplot(fig3)
        else:
            st.error("Required columns for 'CallsOffered' or 'CallsAbandoned' are missing.")
    except Exception as e:
        st.error(f"Error while creating scatterplot for Calls Offered vs Calls Abandoned: {e}")

    # 4. Correlation between Agents and Call Abandonment by Month
    try:
        st.write("### Correlation between Agents and Call Abandonment by Month")
        required_columns = ['Month', 'Agents', 'CallsAbandoned', 'CallsOffered']
        if all(col in st.session_state.df.columns for col in required_columns):
            monthly_data = st.session_state.df.groupby('Month').agg({
                'Agents': 'sum',
                'CallsAbandoned': 'sum',
                'CallsOffered': 'sum'
            }).reset_index()

            monthly_data['AbandonmentRate'] = (
                monthly_data['CallsAbandoned'] / monthly_data['CallsOffered'].replace(0, np.nan)
            ) * 100

            month_order = [
                'Jan-Feb', 'Feb-Mar', 'Mar-Apr', 'Apr-May', 'May-Jun',
                'Jun-Jul', 'Jul-Aug', 'Aug-Sep', 'Sep-Oct', 'Oct-Nov', 'Nov-Dec', 'Dec-Jan'
            ]
            monthly_data['Month'] = pd.Categorical(monthly_data['Month'], categories=month_order, ordered=True)
            monthly_data = monthly_data.sort_values('Month')

            fig4, ax1 = plt.subplots(figsize=(10, 6))

            ax1.bar(monthly_data['Month'], monthly_data['Agents'], color='blue', alpha=0.7)
            ax1.set_ylabel('Total Number of Agents', color='blue')
            ax1.tick_params(axis='y', labelcolor='blue')
            ax1.set_xlabel('Months')

            ax2 = ax1.twinx()
            ax2.plot(monthly_data['Month'], monthly_data['AbandonmentRate'], color='red', marker='o')
            ax2.set_ylabel('Average Call Abandonment Rate (%)', color='red')
            ax2.tick_params(axis='y', labelcolor='red')

            plt.title('Agents and Call Abandonment by Month')
            plt.tight_layout()
            st.pyplot(fig4)
        else:
            st.error("Required columns are missing from the dataset.")
    except Exception as e:
        st.error(f"Error while creating plot for Agents and Call Abandonment: {e}")

# Correlations Tab
elif tab_choice == "📉 Correlations":
    st.subheader("Correlation Heatmaps by VHT Status")
    logging.info("User viewed correlation heatmaps.")  # Log action

    numeric_columns = st.session_state.df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_columns) > 0:
        vht_on = st.session_state.df[st.session_state.df['VHT'] == 'On']
        vht_off = st.session_state.df[st.session_state.df['VHT'] == 'Off']
        correlation_on = vht_on[numeric_columns].corr()
        correlation_off = vht_off[numeric_columns].corr()
        
        fig5, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        sns.heatmap(correlation_on, annot=True, cmap='coolwarm', ax=ax1)
        ax1.set_title('Correlation when VHT is On')
        sns.heatmap(correlation_off, annot=True, cmap='coolwarm', ax=ax2)
        ax2.set_title('Correlation when VHT is Off')
        st.pyplot(fig5)

if st.sidebar.button("Save Changes"):
    st.session_state.df.to_csv(DATA_PATH, index=False)
    st.sidebar.success("Data saved successfully!")
    logging.info("Data saved to file.")

