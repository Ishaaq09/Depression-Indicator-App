import pandas as pd
import numpy as np

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    
    df = df.drop(columns=['id', 'Name', 'Academic Pressure', 'CGPA', 'Study Satisfaction'], errors='ignore')

    city_corrections = {
        "Molkata": "Kolkata",
        "Tolkata": "Kolkata",
        "Khaziabad": "Ghaziabad",
        "Nalyan": "Kalyan",
        "Less Delhi": "Delhi",
        "Less than 5 Kalyan": "Kalyan"
    }
    df['City'] = df['City'].replace(city_corrections)
    valid_cities = [
        'Ludhiana', 'Varanasi', 'Visakhapatnam', 'Mumbai', 'Kanpur',
        'Ahmedabad', 'Thane', 'Nashik', 'Bangalore', 'Patna', 'Rajkot',
        'Jaipur', 'Pune', 'Lucknow', 'Meerut', 'Agra', 'Surat',
        'Faridabad', 'Hyderabad', 'Srinagar', 'Ghaziabad', 'Kolkata',
        'Chennai', 'Kalyan', 'Nagpur', 'Vadodara', 'Vasai-Virar', 'Delhi',
        'Bhopal', 'Indore', 'Gurgaon'
    ]

    profession_corrections = {
        "Finanancial Analyst": "Financial Analyst",  # common spelling mistake
        "UX/UI Designer": "Designer",
        "Research Analyst": "Analyst"
    }
    df['Profession'] = df['Profession'].replace(profession_corrections)
    valid_professions = [
        "Software Engineer", "Doctor", "Teacher", "Lawyer", "Business Analyst",
        "Chef", "Engineer", "Nurse", "Accountant", "Data Scientist", "Journalist",
        "Consultant", "Architect", "Pharmacist", "Dentist", "Designer", "Scientist"
    ]

    sleep_corrections = {
        "8 hours": "More than 8 hours",
        "9-6 hours": "More than 8 hours",
        "10-11 hours": "More than 8 hours",
        "1-6 hours": "Less than 5 hours",
        "than 5 hours": "Less than 5 hours"
    }
    df['Sleep Duration'] = df['Sleep Duration'].replace(sleep_corrections)
    valid_sleep = [
        "Less than 5 hours", "5-6 hours", "6-7 hours", "7-8 hours", "More than 8 hours"
    ]

    dietary_corrections = {
        "More Healthy": "Healthy",
        "Less Healthy": "Unhealthy",
        "No Healthy": "Unhealthy"
    }
    df['Dietary Habits'] = df['Dietary Habits'].replace(dietary_corrections)
    valid_dietary = ["Healthy", "Unhealthy", "Moderate"]

    degree_corrections = {
        "MTech": "M.Tech",
        "B.Sc": "B.Sc",
        "MSc": "M.Sc",
        "B.Ed": "B.Ed",
        "M.Ed": "M.Ed"
    }
    df['Degree'] = df['Degree'].replace(degree_corrections)
    valid_degrees = [
        "B.Tech", "M.Tech", "MBA", "PhD", "B.Sc", "M.Sc", "B.Com", "M.Com",
        "MCA", "BBA", "MBBS", "LLB", "LLM", "M.Ed", "B.Ed", "M.Pharm", "B.Pharm"
    ]

    df['City'] = df['City'].apply(lambda x: x if str(x).strip() in valid_cities else np.nan)
    df['Profession'] = df['Profession'].apply(lambda x: x if str(x).strip() in valid_professions else np.nan)
    df['Sleep Duration'] = df['Sleep Duration'].apply(lambda x: x if str(x).strip() in valid_sleep else np.nan)
    df['Dietary Habits'] = df['Dietary Habits'].apply(lambda x: x if str(x).strip() in valid_dietary else np.nan)
    df['Degree'] = df['Degree'].apply(lambda x: x if str(x).strip() in valid_degrees else np.nan)

    return df


if __name__ == "__main__":
    df = pd.read_csv("data/train.csv")
    cleaned_df = clean_data(df)

    print("Cleaning complete!")
    print(df.head())

    cleaned_df.to_csv("cleaned_dataset.csv", index=False)
