import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load the models (make sure these match the models you saved)
with open('rf_model.pkl', 'rb') as file:
    dt2_model = pickle.load(file)

# Load your preprocessed data to get unique values for dropdowns
df = pd.read_csv("cleaned_data.csv")

def predict_price(model, bathrooms, bedrooms, built_year, garage_space, kitchens, sqft, property_type, city):    
    # Create an input array with zeros, matching the number of columns in X
    X_columns = df.drop(columns=['Price']).columns
    x = np.zeros(len(X_columns))
    
    # Set values for numerical features
    numerical_features = ['Bathrooms', 'Bedrooms', 'Built Year', 'Garage Space', 'Kitchens', 'Sqft']
    feature_mapping = {
        'Bathrooms': 0,
        'Bedrooms': 1,
        'Built Year': 2,
        'Garage Space': 3,
        'Kitchens': 4,
        'Sqft': 5
    }
    
    # Set numerical feature values
    x[feature_mapping['Bathrooms']] = bathrooms
    x[feature_mapping['Bedrooms']] = bedrooms
    x[feature_mapping['Built Year']] = built_year
    x[feature_mapping['Garage Space']] = garage_space
    x[feature_mapping['Kitchens']] = kitchens
    x[feature_mapping['Sqft']] = sqft
    
    # Set one-hot encoded values for property type
    property_type_column = f"Property Type_{property_type}"
    if property_type_column in X_columns:
        loc_index = np.where(X_columns == property_type_column)[0][0]
        x[loc_index] = 1
    
    # Set one-hot encoded values for city
    city_column = f"City_{city}"
    if city_column in X_columns:
        loc_index = np.where(X_columns == city_column)[0][0]
        x[loc_index] = 1
    
    # Predict the price using the provided model
    return model.predict([x])[0]

def main():
    st.title('Property Price Predictor')
    
    # Sidebar for input
    st.sidebar.header('Property Details')
    
    # Numerical Inputs
    bathrooms = st.sidebar.number_input('Number of Bathrooms', min_value=1, max_value=10, value=2)
    bedrooms = st.sidebar.number_input('Number of Bedrooms', min_value=1, max_value=10, value=3)
    built_year = st.sidebar.number_input('Year Built', min_value=1900, max_value=2024, value=2000)
    garage_space = st.sidebar.number_input('Garage Spaces', min_value=0, max_value=5, value=1)
    kitchens = st.sidebar.number_input('Number of Kitchens', min_value=1, max_value=5, value=1)
    sqft = st.sidebar.number_input('Square Footage', min_value=500, max_value=10000, value=2000)
    
    # Categorical Inputs (using unique values from dataset)
    property_types = [col.split('Property Type_')[1] for col in df.columns if col.startswith('Property Type_')]
    property_type = st.sidebar.selectbox(
        'Property Type', 
        property_types, 
        index=property_types.index('Single Family Residence')  # Set default to "Single Family Residence"
    )
    
    cities = [col.split('City_')[1] for col in df.columns if col.startswith('City_')]
    city = st.sidebar.selectbox('City', cities)
    
    # Prediction Button
    if st.sidebar.button('Predict Price'):
        # Predict using Decision Tree model
        predicted_price = predict_price(
            dt2_model, 
            bathrooms, 
            bedrooms, 
            built_year, 
            garage_space, 
            kitchens, 
            sqft, 
            property_type, 
            city
        )
        
        # Display Results
        st.header('Prediction Results')
        st.success(f'Estimated Property Price: ${predicted_price:,.2f}')
        
        # Additional Details Section
        st.subheader('Property Details')
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Property Type:** {property_type}")
            st.write(f"**City:** {city}")
            st.write(f"**Square Footage:** {sqft} sq ft")
        with col2:
            st.write(f"**Bedrooms:** {bedrooms}")
            st.write(f"**Bathrooms:** {bathrooms}")
            st.write(f"**Year Built:** {built_year}")

    # Optional: Add some information about the app
    st.sidebar.markdown('''
    ### About this App
    - Predicts property prices using machine learning
    - Uses Decision Tree Regression model
    - Considers multiple property features
    ''')

if __name__ == '__main__':
    main()