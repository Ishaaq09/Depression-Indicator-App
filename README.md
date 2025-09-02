# Depression Indicator App

A deep learning web application that predicts depression indicators from user responses using a trained neural network. Built with **PyTorch** for modeling and **Streamlit** for deployment, the app provides an interactive interface for real-time predictions.



## Problem Theme
Mental health issues like depression are rising globally, yet early detection remains challenging. Technology can assist in building awareness and providing tools to identify possible indicators.



## Problem Statement
Given user input (demographic and lifestyle features), build a model that can predict whether an individual is likely to exhibit signs of depression. This tool is **not for medical diagnosis**, but for **educational and research purposes**.



## Our Aim
- Develop a **deep learning classifier** to identify potential depression indicators.  
- Provide a **Streamlit-based web app** for real-time, interactive predictions.  
- Create a **user-friendly and accessible platform** that highlights the role of AI in mental health awareness.  



## Dataset
- The dataset contains **demographic, behavioral, and lifestyle attributes** correlated with mental health.  
- Preprocessing steps included:
  - Handling categorical and numerical features  
  - Normalization & encoding with **scikit-learn ColumnTransformer**  
  - Splitting into training and test sets  

*(Note: Dataset is for demonstration purposes and not from medical records.)*



## Approach
1. **Data Preprocessing**  
   - Encoded categorical variables  
   - Normalized numerical features  
   - Built a reusable preprocessing pipeline  

2. **Model Development (PyTorch)**  
   - Implemented a **Multilayer Perceptron (MLP)**  
   - Loss Function: CrossEntropyLoss  
   - Optimizer: Adam  
   - Evaluation: Accuracy, F1-Score  

3. **Deployment (Streamlit)**  
   - Built an intuitive **UI for manual input**  
   - Integrated trained model and preprocessing pipeline  
   - Deployed on **Streamlit Cloud**  



## Deep Learning Framework
- **PyTorch** was used for training the MLP model.  
- Benefits: Flexibility, strong community support, and integration with deployment pipelines.  



## Model Architecture
- Input layer: Matches preprocessed features  
- Hidden layers: Fully connected (ReLU activations, Dropout)  
- Output layer: Binary classification (Depression Indicator: Yes/No)  



## Streamlit UI
- Simple and interactive user interface  
- Input fields for all features used in training  
- Real-time prediction output  

**Live App Link:**  
[Depression Indicator App on Streamlit](https://ishaaq09-depression-indicator-app-app-qngzxe.streamlit.app/)



## Tech Stack
- **Python 3.10**  
- **PyTorch** – Deep learning model  
- **scikit-learn** – Preprocessing pipeline  
- **Streamlit** – Interactive UI + Deployment  
- **Pandas / Numpy** – Data handling  



## Disclaimer
This app is **not a medical diagnostic tool**. It is intended solely for **educational and awareness purposes**. If you are experiencing mental health issues, please consult a qualified professional.  



## Contributions
Contributions are welcome! Feel free to fork the repo, open issues, or submit pull requests.
