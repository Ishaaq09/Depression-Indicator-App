# Depression Indicator App  
*A Deep Learning based Mental Health Prediction System*  

![Streamlit](https://img.shields.io/badge/Framework-Streamlit-red)  
![PyTorch](https://img.shields.io/badge/DeepLearning-PyTorch-blue)  
![Status](https://img.shields.io/badge/Status-Deployed-brightgreen)  

## Problem Theme  
Mental health is a critical issue affecting individuals worldwide. Early identification of depression can help healthcare providers, organizations, and communities take preventive measures.  
This project leverages **deep learning** to predict depression likelihood based on **demographics, lifestyle habits, and medical history** from survey data.  



## Problem Statement  
The goal of this project is to **predict whether an individual may experience depression** using survey data.  
The model is trained on multiple features such as:  
- Age, gender, city  
- Lifestyle factors (work/study hours, sleep duration, dietary habits)  
- Education background and profession  
- Mental health history (suicidal thoughts, family history, stress levels)  

The final solution includes a **PyTorch-based deep learning model** integrated with a **Streamlit web app** for real-time prediction.  



## Dataset  
The dataset used is a **Mental Health Survey dataset**.  

- [Train Dataset](https://github.com/Ishaaq09/Depression-Indicator-App/blob/main/data/train.csv)  
- [Test Dataset](https://github.com/Ishaaq09/Depression-Indicator-App/blob/main/data/test.csv)  
- [Sample Submission](https://github.com/Ishaaq09/Depression-Indicator-App/blob/main/data/sample_submission.csv)  

**Target Variable**: `Depression` (1 = Depression, 0 = No Depression)  

### Features  
- `Age`, `Gender`, `City`, `Profession`, `Degree`  
- `Work/Study Hours`, `Work Pressure`, `Job Satisfaction`, `Financial Stress`  
- `Sleep Duration`, `Dietary Habits`  
- `Have you ever had suicidal thoughts?`  
- `Family History of Mental Illness`  



## Approach  

### Data Preprocessing  
- Removed irrelevant columns (`id`, `Name`, etc.)  
- Handled missing values using **SimpleImputer**  
- Standardized numerical features with **StandardScaler**  
- One-hot encoded categorical features with **OneHotEncoder**  
- Corrected inconsistent category labels (e.g., `"Molkata"` → `"Kolkata"`, `"Finanancial Analyst"` → `"Financial Analyst"`)  

### Deep Learning Model (PyTorch MLP)  
- Custom **Multilayer Perceptron (MLP)** with layers:  
  - Input → Dense(64, ReLU) → Dropout(0.3) → Dense(32, ReLU) → Dense(1, Sigmoid)  
- Loss Function: **Binary Cross-Entropy Loss (BCELoss)**  
- Optimizer: **Adam**  
- Metrics: **Accuracy, Precision, Recall, F1-Score**  

### Training & Evaluation  
- Train-test split: 80-20  
- Batch size: 32  
- Achieved good performance on test set with balanced precision-recall  

### Streamlit Application  
- Built an interactive **web interface**  
- User can enter their details (age, city, lifestyle factors, medical history, etc.)  
- Model predicts **"High Risk"** or **"Low Risk"** of depression  
- Deployed on **Streamlit Cloud**  

**Live App**: [Depression Indicator App](https://ishaaq09-depression-indicator-app-app-qngzxe.streamlit.app/)  

## Business Use Cases  
1. **Healthcare Providers** → Identify at-risk patients early.  
2. **Mental Health Clinics** → Support data-driven treatment plans.  
3. **Corporate Wellness Programs** → Monitor employee wellbeing.  
4. **Government/NGOs** → Allocate resources for high-risk populations.  



## Tech Stack  
- **Python 3.10+**  
- **PyTorch** (Deep Learning)  
- **scikit-learn** (Preprocessing)  
- **Streamlit** (Web App)  
- **Pandas / NumPy** (Data Handling)  



## References  
- [Streamlit Documentation](https://docs.streamlit.io/)  
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)  
- [Scikit-learn Documentation](https://scikit-learn.org/stable/documentation.html)  
- [GUVI Project Guidelines](https://www.guvi.in/)  



## Author  
**Ishaaq M M**  
- GitHub: [Ishaaq09](https://github.com/Ishaaq09)  
