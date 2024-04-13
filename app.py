import streamlit as st
import pickle
import numpy as np

def load_model():
    with open('saved_steps.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

data = load_model()
regressor_loaded = data["model"]
Country = data["Country"]
Location = data["Location"]
cell = data["cell"]
gender = data["gender"]
relation = data["relation"]
marital = data["marital"]
education = data["education"]
job = data["job"]

def show_predict_page():
    st.title("Bank Account Prediction")

    st.write("""### We need some information to predict the Bank status""")
    default_values = ['Kenya', 2018, 'Rural', 'Yes', 3, 24, 'Female', 'Spouse', 'Married/Living together', 'Secondary education', 'Self employed']

    countries_list = ('Kenya', 'Tanzania', 'Uganda')
    cellphone_access = ('Yes', 'No')
    location_types = ('Rural', 'Urban')
    gender_of_respondent = ('Male', 'Female')
    relationship_with_head = ('Head of Household', 'Spouse', 'Child', 'Parent', 'Other relative', 'Other non-relatives')
    marital_status = ('Married/Living together', 'Divorced/Seperated', 'Widowed', 'Single/Never Married')
    education_level = ('No formal education', 'Primary education', 'Secondary education', 'Vocational/Specialised training', 'Tertiary education', 'Other/Dont know/RTA')
    job_type = ('Government Dependent', 'Formally employed Private', 'Formally employed Government', 'Self employed', 'Informally employed', 'Farming and Fishing', 'Remittance Dependent', 'Other Income', 'Dont Know/Refuse to answer')

    country = st.selectbox("Country", countries_list, index=countries_list.index(default_values[0]))
    year = st.number_input("Year", value=default_values[1])
    location_type = st.selectbox("Location Type", location_types, index=location_types.index(default_values[2]))
    cellphone_access_input = st.selectbox("Cellphone Access", cellphone_access, index=cellphone_access.index(default_values[3]))
    household_size = st.number_input("Household Size", value=default_values[4])
    age_of_respondent = st.number_input("Age of Respondent", value=default_values[5])
    gender_of_respondent_input = st.selectbox("Gender of Respondent", gender_of_respondent, index=gender_of_respondent.index(default_values[6]))
    relationship_with_head_input = st.selectbox("Relationship with Head", relationship_with_head, index=relationship_with_head.index(default_values[7]))
    marital_status_input = st.selectbox("Marital Status", marital_status, index=marital_status.index(default_values[8]))
    education_level_input = st.selectbox("Education Level", education_level, index=education_level.index(default_values[9]))
    job_type_input = st.selectbox("Job Type", job_type, index=job_type.index(default_values[10]))

    ok = st.button("Check")
    if ok:
        data_np = np.array([[
            country, year,location_type, cellphone_access_input,
            household_size, age_of_respondent, gender_of_respondent_input,
            relationship_with_head_input, marital_status_input, education_level_input, job_type_input
        ]])
        data_np[:, 0] = Country.transform(data_np[:,0])
        data_np[:, 2] = Location.transform(data_np[:,2])
        data_np[:, 3] = cell.transform(data_np[:,3])
        data_np[:, 6] = gender.transform(data_np[:,6])
        data_np[:, 7] = relation.transform(data_np[:,7])
        data_np[:, 8] = marital.transform(data_np[:,8])
        data_np[:, 9] = education.transform(data_np[:,9])
        data_np[:, 10] = job.transform(data_np[:,10])
        data_np=data_np.astype(float)
        y_pred = regressor_loaded.predict(data_np)
        print(y_pred)
        result = "less likely to have or use a bank account" if y_pred[0] == 1 else "most likely to have or use a bank account"
        st.write(f"Prediction Result: {result}")

show_predict_page()
