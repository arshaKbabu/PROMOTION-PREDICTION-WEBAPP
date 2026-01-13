
import streamlit as st
import numpy as np
import pickle
import base64

# Load model and scaler
model = pickle.load(open('promotion_model.pkl', 'rb'))
scaler = pickle.load(open('promotion_scaler.pkl', 'rb'))

# Mappings
dept_map = {'Finance': 0, 'HR': 1, 'IT': 2, 'Marketing': 3}
edu_map = {'Bachelors': 0, 'High School': 1, 'Masters': 2, 'PhD': 3}

# Convert image to base64
def get_base64_of_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# Inject background image
def add_bg_image(image_path):
    base64_img = get_base64_of_image(image_path)
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{base64_img}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}

        .stApp::before {{
            content: "";
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(0,0,0,0.4);
            z-index: -1;
        }}

        .css-18e3th9 {{
            background-color: rgba(255, 255, 255, 0.85);
            padding: 2rem;
            border-radius: 10px;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

def main():
    # ✅ Replace with the actual path of your image
    local_image_path = r"C:\Users\acer\OneDrive\Desktop\promotion_project\pr.jpg"
    add_bg_image(local_image_path)

    st.title("🚀 Employee Promotion Prediction")

    st.subheader("Please enter the employee details:")

    age = st.number_input("Age", min_value=18, max_value=65, value=30)
    salary = st.number_input("Salary", min_value=10000, max_value=1000000, value=50000)
    experience = st.number_input("Years of Experience", min_value=0, max_value=40, value=5)
    score = st.slider("Performance Score (0-100)", min_value=0.0, max_value=100.0, value=75.0)

    department = st.selectbox("Department", options=list(dept_map.keys()))
    education = st.selectbox("Education Level", options=list(edu_map.keys()))

    projects = st.number_input("Number of Projects", min_value=0, max_value=50, value=3)
    hours_worked = st.number_input("Hours Worked per Week", min_value=1, max_value=100, value=40)
    rating = st.slider("Manager Rating (1.0 to 5.0)", min_value=1.0, max_value=5.0, value=3.5)

    if st.button("Predict Promotion"):
        department_mapped = dept_map[department]
        education_mapped = edu_map[education]

        user_data = np.array([
            age,
            salary,
            experience,
            score,
            department_mapped,
            education_mapped,
            projects,
            hours_worked,
            rating
        ]).reshape(1, -1)

        user_data_scaled = scaler.transform(user_data)
        prediction = model.predict(user_data_scaled)

        if prediction[0] == 1:
            st.success("✅ This person is likely to get a promotion.")
        else:
            st.error("❌ This person is NOT likely to get a promotion.")

if __name__ == "__main__":
    main()

