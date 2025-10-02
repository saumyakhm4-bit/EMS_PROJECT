# Employee Management System with ML

A full-featured Employee Management System built with Python Flask that includes Machine Learning capabilities for salary prediction.

## Features
- Employee CRUD operations (Create, Read, Update, Delete)
- Salary prediction using Machine Learning (Linear Regression)
- Responsive web interface with Bootstrap
- SQLite database for data storage

## Installation
1. Install Python 3.8+
2. Install dependencies: `pip install -r requirements.txt`
3. Run the application: `python app.py`
4. Open your browser and navigate to `http://localhost:5000`

## Usage
- View all employees on the dashboard
- Add new employees using the "Add Employee" form
- Edit existing employee records
- Delete employees with confirmation
- Use the "Predict Salary" feature to estimate salaries based on experience, education, and department

## Machine Learning
The system uses a Linear Regression model trained on synthetic data to predict salaries. The model considers:
- Years of experience
- Education level (High School, Bachelor, Master, PhD)
- Department (HR, IT, Finance, Marketing)

## File Structure
ems_project/
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── database.db           # SQLite database (created automatically)
└── templates/
    ├── base.html         # Base template
    ├── index.html        # Employee list/dashboard
    ├── add_employee.html # Add employee form
    ├── edit_employee.html # Edit employee form
    └── predict_salary.html # ML prediction page
