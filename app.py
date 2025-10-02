from flask import Flask, render_template, request, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import os

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'  # Change in production
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Import models after db init
from models import Employee

# ML Model: Train a simple Linear Regression for salary prediction
def train_ml_model():
    # Synthetic dataset (in real app, load from CSV or DB)
    data = {
        'experience': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 10,
        'education': ['High School', 'Bachelor', 'Master', 'PhD'] * 25,
        'department': ['HR', 'IT', 'Finance', 'Marketing'] * 25,
        'salary': [30000, 45000, 60000, 80000, 100000, 120000, 150000, 180000, 200000, 250000] * 10
    }
    df = pd.DataFrame(data)
    
    # Preprocess: Encode categorical variables
    le_edu = LabelEncoder()
    le_dept = LabelEncoder()
    df['education_encoded'] = le_edu.fit_transform(df['education'])
    df['department_encoded'] = le_dept.fit_transform(df['department'])
    
    X = df[['experience', 'education_encoded', 'department_encoded']]
    y = df['salary']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Save encoders and model (in memory for simplicity; use joblib for persistence)
    return model, le_edu, le_dept

# Global ML components (train once)
ml_model, edu_encoder, dept_encoder = train_ml_model()

# Routes
@app.route('/')
def index():
    employees = Employee.query.all()
    return render_template('index.html', employees=employees)

@app.route('/add', methods=['GET', 'POST'])
def add_employee():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        department = request.form['department']
        experience = int(request.form['experience'])
        education = request.form['education']
        salary = float(request.form['salary'])
        
        # Validation
        if not all([name, email, department]):
            flash('Please fill all required fields.', 'error')
            return redirect(url_for('add_employee'))
        
        new_employee = Employee(name=name, email=email, department=department,
                                experience=experience, education=education, salary=salary)
        db.session.add(new_employee)
        db.session.commit()
        flash('Employee added successfully!', 'success')
        return redirect(url_for('index'))
    
    return render_template('add_employee.html')

@app.route('/edit/<int:id>', methods=['GET', 'POST'])
def edit_employee(id):
    employee = Employee.query.get_or_404(id)
    
    if request.method == 'POST':
        employee.name = request.form['name']
        employee.email = request.form['email']
        employee.department = request.form['department']
        employee.experience = int(request.form['experience'])
        employee.education = request.form['education']
        employee.salary = float(request.form['salary'])
        
        db.session.commit()
        flash('Employee updated successfully!', 'success')
        return redirect(url_for('index'))
    
    return render_template('edit_employee.html', employee=employee)

@app.route('/delete/<int:id>')
def delete_employee(id):
    employee = Employee.query.get_or_404(id)
    db.session.delete(employee)
    db.session.commit()
    flash('Employee deleted successfully!', 'success')
    return redirect(url_for('index'))

@app.route('/predict_salary', methods=['GET', 'POST'])
def predict_salary():
    prediction = None
    if request.method == 'POST':
        experience = int(request.form['experience'])
        education = request.form['education']
        department = request.form['department']
        
        # Encode inputs
        edu_encoded = edu_encoder.transform([education])[0]
        dept_encoded = dept_encoder.transform([department])[0]
        
        # Predict
        input_features = np.array([[experience, edu_encoded, dept_encoded]])
        prediction = ml_model.predict(input_features)[0]
        prediction = round(prediction, 2)
    
    return render_template('predict_salary.html', prediction=prediction)

if __name__ == '__main__':
    with app.app_context():
        db.create_all()  # Create tables if they don't exist
    app.run(debug=True)