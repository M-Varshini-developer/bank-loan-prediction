from flask import Flask, render_template, request
import pandas as pd
import pickle
import os

app = Flask(__name__, template_folder="templates")

# -----------------------------
# Base directory
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

print("App file:", __file__)
print("Template folder:", app.template_folder)
print("BASE_DIR:", BASE_DIR if 'BASE_DIR' in globals() else "Not yet defined")
# -----------------------------
# File paths (IMPORTANT FIX)
# -----------------------------
model_path = os.path.join(BASE_DIR, "loan_model.pkl")
encoder_path = os.path.join(BASE_DIR, "employment_encoder.pkl")
csv_path = os.path.join(BASE_DIR, "applications.csv")

# -----------------------------
# Load Model and Encoder
# -----------------------------
model = pickle.load(open(model_path, "rb"))
encoder = pickle.load(open(encoder_path, "rb"))

print("BASE_DIR:", BASE_DIR)
print("Files:", os.listdir(BASE_DIR))

# -----------------------------
# Create CSV if not exists
# -----------------------------
if not os.path.exists(csv_path):

    df = pd.DataFrame(columns=[
        "name",
        "age",
        "salary",
        "cibil",
        "experience",
        "employment_type",
        "existing_emi",
        "loan_amount_requested",
        "bank_balance",
        "dependents",
        "prediction"
    ])

    df.to_csv(csv_path, index=False)

# -----------------------------
# Home Page
# -----------------------------
@app.route("/")
def home():
    return render_template("index.html")

# -----------------------------
# Loan Prediction
# -----------------------------
@app.route("/predict", methods=["POST"])
def predict():

    name = request.form["name"]

    age = int(request.form["age"])
    salary = int(request.form["salary"])
    cibil = int(request.form["cibil"])
    experience = int(request.form["experience"])

    employment_type = request.form["employment_type"]

    
    employment_mapping = {
    "Government": 0,
    "Private": 1,
    "Self-Employed": 2,
    "Business": 3
}
    employment_type_encoded = employment_mapping[employment_type]

    existing_emi = int(request.form["existing_emi"])
    loan_amount_requested = int(request.form["loan_amount_requested"])
    bank_balance = int(request.form["bank_balance"])
    dependents = int(request.form["dependents"])

    # -----------------------------
    # Input for model
    # -----------------------------
    input_data = pd.DataFrame([{
        "age": age,
        "salary": salary,
        "cibil": cibil,
        "experience": experience,
        "employment_type": employment_type_encoded,
        "existing_emi": existing_emi,
        "loan_amount_requested": loan_amount_requested,
        "bank_balance": bank_balance,
        "dependents": dependents
    }])

    prediction = model.predict(input_data)[0]

    if prediction == 1:
        result = "Loan Approved ✅"
    else:
        result = "Loan Rejected ❌"

    # -----------------------------
    # Save to CSV (FIXED)
    # -----------------------------
    new_data = pd.DataFrame([{
        "name": name,
        "age": age,
        "salary": salary,
        "cibil": cibil,
        "experience": experience,
        "employment_type": employment_type,
        "existing_emi": existing_emi,
        "loan_amount_requested": loan_amount_requested,
        "bank_balance": bank_balance,
        "dependents": dependents,
        "prediction": result
    }])

    file_exists = os.path.isfile(csv_path)

    new_data.to_csv(
        csv_path,
        mode="a",
        header=not file_exists,
        index=False
    )

    return render_template(
        "result.html",
        prediction=result,
        name=name
    )

# -----------------------------
# Manager Dashboard
# -----------------------------
@app.route("/manager")
def manager():

    df = pd.read_csv(csv_path)

    total = len(df)
    approved = len(df[df["prediction"] == "Loan Approved ✅"])
    rejected = len(df[df["prediction"] == "Loan Rejected ❌"])

    return render_template(
        "manager.html",
        total=total,
        approved=approved,
        rejected=rejected
    )

# -----------------------------
# Customer List
# -----------------------------
@app.route("/customers")
def customers():

    df = pd.read_csv(csv_path)

    customers = df.to_dict(orient="records")

    return render_template(
        "customers.html",
        customers=customers
    )

# -----------------------------
# Run Flask
# -----------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)