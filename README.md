
# Employee Salary Prediction

This project aims to predict whether an employee earns more than ₹60K based on various features such as age, education, experience, occupation, and working hours. It uses supervised machine learning models and provides both model performance comparison and a Streamlit-based interactive prediction app.

## 🔍 Problem Statement

To build a machine learning model that classifies employee salaries into two categories:  
- **> ₹60,000**  
- **≤ ₹60,000**

This helps organizations in understanding key salary-driving factors and streamlining payroll predictions.

---

## ⚙️ System Development Approach

- **Language**: Python  
- **Libraries Used**:  
  - Data Handling: `pandas`, `numpy`  
  - Visualization: `matplotlib`, `seaborn`  
  - ML Models: `scikit-learn`  
  - UI Deployment: `streamlit`  
  - Model Persistence: `joblib`  

---

## 🧠 Algorithm & Deployment Steps

1. **Data Preprocessing**
   - Handling missing values
   - Label encoding and feature scaling

2. **Exploratory Data Analysis (EDA)**
   - Univariate & bivariate analysis
   - Heatmap, distribution plots, box plots

3. **Model Building**
   - Logistic Regression
   - Random Forest
   - K-Nearest Neighbors
   - Support Vector Machine
   - Gradient Boosting

4. **Model Evaluation**
   - Accuracy, Confusion Matrix, Cross-validation
   - Best accuracy (~86%) by Gradient Boosting

5. **Deployment**
   - Streamlit app for single-user prediction

---

## 📊 Results

- **Best Model**: Gradient Boosting Classifier  
- **Accuracy**: ~86%  
- Model comparison and accuracy visualized in bar plots  
- Exported key plots and data for documentation

---

## ✅ Conclusion

A robust pipeline was developed to classify employee salaries using machine learning techniques. The model can effectively assist HR or payroll systems in salary classification.

---

## 🚀 Future Enhancements

- Integrate real-time API for live predictions  
- Expand dataset with more employee attributes  
- Deploy on cloud (e.g., Heroku, AWS EC2)

---

## 📁 Project Structure

```
├── Employee_Salary_Prediction.ipynb
├── best_model.pkl
├── app.py
├── requirements.txt
├── visuals/
│   ├── age_distribution.png
│   ├── correlation_heatmap.png
│   ├── model_accuracy_comparison.png
├── data/
│   ├── processed_employee_data.csv
│   ├── X_processed.csv
│   ├── y_labels.csv
├── README.md
```

---

## 📚 References

- [UCI Machine Learning Repository – employee salary prediction_dataset](https://archive.ics.uci.edu/ml/datasets/adult)


---

## 💡 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/Amitvyas2005/employee salaryprediction
   
   cd employee-salary-prediction
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

---


