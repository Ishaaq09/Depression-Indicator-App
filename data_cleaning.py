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
        "Finanancial Analyst": "Financial Analyst",
        "Research Analyst": "Research Analyst",   
        "UX/UI Designer": "UX/UI Designer",       
        "Analyst": "Research Analyst",            
        "Designer": "UX/UI Designer"              
    }
    df['Profession'] = df['Profession'].replace(profession_corrections)

    valid_professions = [
        'Chef', 'Teacher', 'Business Analyst', 'Financial Analyst',
        'Chemist', 'Electrician', 'Software Engineer', 'Data Scientist',
        'Plumber', 'Marketing Manager', 'Accountant', 'Entrepreneur',
        'HR Manager', 'UX/UI Designer', 'Content Writer',
        'Educational Consultant', 'Civil Engineer', 'Manager',
        'Pharmacist', 'Architect', 'Mechanical Engineer', 'Customer Support',
        'Consultant', 'Judge', 'Researcher', 'Pilot', 'Graphic Designer',
        'Travel Consultant', 'Digital Marketer', 'Lawyer',
        'Research Analyst', 'Sales Executive', 'Doctor',
        'Unemployed', 'Investment Banker', 'Family Consultant',
        'Medical Doctor', 'Working Professional', 'Student'
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
        "M_Tech": "M.Tech",
        "S.Tech": "M.Tech",
        "E.Tech": "M.Tech",
        "ME": "M.Tech",
        "BTech": "B.Tech",
        "B.Tech.": "B.Tech",
        "BSc": "B.Sc",
        "MSc": "M.Sc",
        "M.S": "M.Sc",
        "BPharm": "B.Pharm",
        "MPharm": "M.Pharm",
        "N.Pharm": "M.Pharm",
        "S.Pharm": "M.Pharm",
        "P.Pharm": "M.Pharm",
        "H_Pharm": "B.Pharm",
        "L.Ed": "B.Ed",
        "BEd": "B.Ed",
        "MEd": "M.Ed",
        "LLEd": "LLB",       
        "LLTech": "LLB",      
        "LL BA": "LLB",
        "LL B.Ed": "LLB",
        "BArch": "B.Arch",
        "B.B.Arch": "B.Arch",
        "S.Arch": "M.Arch"
    }
    df['Degree'] = df['Degree'].replace(degree_corrections)
    valid_degrees = [
        'BHM', 'LLB', 'B.Pharm', 'BBA', 'MCA', 'MD', 'B.Sc', 'M.Tech',
        'B.Arch', 'BCA', 'BE', 'MA', 'B.Ed', 'B.Com', 'MBA', 'M.Com',
        'MHM', 'BA', 'Class 12', 'PhD', 'M.Ed', 'M.Sc', 'B.Tech', 'LLM',
        'MBBS', 'M.Pharm', 'MPA', 'BH', 'M.Arch', 'BPA', 'ACA', 'LHM', 'HCA'
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
