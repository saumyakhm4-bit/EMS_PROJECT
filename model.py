from app import db

class Employee(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    department = db.Column(db.String(50), nullable=False)
    experience = db.Column(db.Integer, nullable=False)
    education = db.Column(db.String(50), nullable=False)
    salary = db.Column(db.Float, nullable=False)
    
    def __repr__(self):
        return f'<Employee {self.name}>'