from flask import Flask, render_template, request, redirect, session
import sqlite3
import joblib
import pandas as pd
from werkzeug.security import generate_password_hash, check_password_hash


# ============================================================
# FLASK APPLICATION
# ============================================================

app = Flask(__name__)

app.secret_key = "rheumacare_secret_key"


# ============================================================
# LOAD MACHINE LEARNING MODEL
# ============================================================

model = joblib.load("model/ra_model.pkl")


# ============================================================
# DATABASE CONNECTION
# ============================================================

def get_db():

    conn = sqlite3.connect("database.db")

    conn.row_factory = sqlite3.Row

    return conn


# ============================================================
# CREATE DATABASE TABLES
# ============================================================

def create_database():

    conn = get_db()


    # -------------------------
    # USERS TABLE
    # -------------------------

    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            name TEXT NOT NULL,

            email TEXT UNIQUE NOT NULL,

            password TEXT NOT NULL

        )
    """)


    # -------------------------
    # HEALTH TRACKER TABLE
    # -------------------------

    conn.execute("""
        CREATE TABLE IF NOT EXISTS health_tracker (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            user_id INTEGER NOT NULL,

            pain_level INTEGER,

            morning_stiffness INTEGER,

            tender_joints INTEGER,

            swollen_joints INTEGER,

            notes TEXT,

            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP

        )
    """)


    # -------------------------
    # MEDICINES TABLE
    # -------------------------

    conn.execute("""
        CREATE TABLE IF NOT EXISTS medicines (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            user_id INTEGER NOT NULL,

            medicine_name TEXT NOT NULL,

            prescribed_details TEXT,

            reminder_time TEXT,

            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP

        )
    """)


    conn.commit()

    conn.close()


# ============================================================
# HOME
# ============================================================

@app.route("/")
def home():

    if "user_id" in session:

        return redirect("/dashboard")

    return redirect("/login")


# ============================================================
# REGISTER
# ============================================================

@app.route("/register", methods=["GET", "POST"])
def register():

    if request.method == "POST":

        name = request.form.get("name")

        email = request.form.get("email")

        password = request.form.get("password")


        password_hash = generate_password_hash(password)


        conn = get_db()


        try:

            conn.execute(
                """
                INSERT INTO users
                (
                    name,
                    email,
                    password
                )
                VALUES (?, ?, ?)
                """,
                (
                    name,
                    email,
                    password_hash
                )
            )

            conn.commit()

            conn.close()

            return redirect("/login")


        except sqlite3.IntegrityError:

            conn.close()

            return render_template(
                "register.html",
                error="Email already registered."
            )


    return render_template("register.html")


# ============================================================
# LOGIN
# ============================================================

@app.route("/login", methods=["GET", "POST"])
def login():

    if request.method == "POST":

        email = request.form.get("email")

        password = request.form.get("password")


        conn = get_db()


        user = conn.execute(
            """
            SELECT *
            FROM users
            WHERE email = ?
            """,
            (email,)
        ).fetchone()


        conn.close()


        if user and check_password_hash(
            user["password"],
            password
        ):

            session["user_id"] = user["id"]

            session["user_name"] = user["name"]

            return redirect("/dashboard")


        else:

            return render_template(
                "login.html",
                error="Invalid email or password."
            )


    return render_template("login.html")


# ============================================================
# DASHBOARD
# ============================================================

@app.route("/dashboard")
def dashboard():

    if "user_id" not in session:

        return redirect("/login")


    return render_template(
        "dashboard.html",
        name=session["user_name"]
    )


# ============================================================
# ASSESSMENT PAGE
# ============================================================

@app.route("/assessment")
def assessment():

    if "user_id" not in session:

        return redirect("/login")


    return render_template("assessment.html")


# ============================================================
# DIET PAGE
# ============================================================

@app.route("/diet")
def diet():

    if "user_id" not in session:

        return redirect("/login")


    return render_template("diet.html")


# ============================================================
# MEDICINE
# ============================================================

@app.route("/medicine", methods=["GET", "POST"])
def medicine():

    if "user_id" not in session:

        return redirect("/login")


    # -------------------------
    # SAVE MEDICINE
    # -------------------------

    if request.method == "POST":

        medicine_name = request.form.get(
            "medicine_name"
        )

        prescribed_details = request.form.get(
            "prescribed_details"
        )

        reminder_time = request.form.get(
            "reminder_time"
        )


        conn = get_db()


        conn.execute(
            """
            INSERT INTO medicines
            (
                user_id,
                medicine_name,
                prescribed_details,
                reminder_time
            )
            VALUES (?, ?, ?, ?)
            """,
            (
                session["user_id"],
                medicine_name,
                prescribed_details,
                reminder_time
            )
        )


        conn.commit()

        conn.close()


        return redirect("/medicine")


    # -------------------------
    # GET MEDICINES
    # -------------------------

    conn = get_db()


    medicines = conn.execute(
        """
        SELECT *
        FROM medicines
        WHERE user_id = ?
        ORDER BY id DESC
        """,
        (session["user_id"],)
    ).fetchall()


    conn.close()


    return render_template(
        "medicine.html",
        medicines=medicines
    )


# ============================================================
# HEALTH TRACKER
# ============================================================

@app.route("/tracker", methods=["GET", "POST"])
def tracker():

    if "user_id" not in session:

        return redirect("/login")


    # -------------------------
    # SAVE TRACKER ENTRY
    # -------------------------

    if request.method == "POST":

        pain_level = request.form.get(
            "pain_level"
        )

        morning_stiffness = request.form.get(
            "morning_stiffness"
        )

        tender_joints = request.form.get(
            "tender_joints"
        )

        swollen_joints = request.form.get(
            "swollen_joints"
        )

        notes = request.form.get(
            "notes"
        )


        conn = get_db()


        conn.execute(
            """
            INSERT INTO health_tracker
            (
                user_id,
                pain_level,
                morning_stiffness,
                tender_joints,
                swollen_joints,
                notes
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                session["user_id"],
                pain_level,
                morning_stiffness,
                tender_joints,
                swollen_joints,
                notes
            )
        )


        conn.commit()

        conn.close()


        return redirect("/tracker")


    # -------------------------
    # GET TRACKER ENTRIES
    # -------------------------

    conn = get_db()


    entries = conn.execute(
        """
        SELECT *
        FROM health_tracker
        WHERE user_id = ?
        ORDER BY id DESC
        """,
        (session["user_id"],)
    ).fetchall()


    conn.close()


    return render_template(
        "tracker.html",
        entries=entries
    )


# ============================================================
# MACHINE LEARNING PREDICTION
# ============================================================

@app.route("/predict", methods=["POST"])
def predict():

    try:

        # -------------------------
        # GET USER INPUT
        # -------------------------

        age = float(
            request.form["age"]
        )

        rf = float(
            request.form["rf"]
        )

        anti_ccp = float(
            request.form["anti_ccp"]
        )

        esr = float(
            request.form["esr"]
        )

        crp = float(
            request.form["crp"]
        )

        tender = float(
            request.form["tender"]
        )

        swollen = float(
            request.form["swollen"]
        )

        stiffness = float(
            request.form["stiffness"]
        )

        disease_duration = float(
            request.form["disease_duration"]
        )


        # -------------------------
        # CREATE INPUT DATAFRAME
        # -------------------------

        input_data = pd.DataFrame([
            {

                "Age": age,

                "Sex": "Female",

                "SmokingStatus": "Never",

                "FamilyHistoryRA": "No",

                "RF_IU_mL": rf,

                "AntiCCP_U_mL": anti_ccp,

                "ESR_mm_hr": esr,

                "CRP_mg_L": crp,

                "TenderJointCount": tender,

                "SwollenJointCount": swollen,

                "MorningStiffness_min": stiffness,

                "DiseaseDuration_months":
                    disease_duration

            }
        ])


        # -------------------------
        # PREDICTION
        # -------------------------

        prediction = model.predict(
            input_data
        )[0]


        probabilities = model.predict_proba(
            input_data
        )[0]


        classes = model.classes_


        probability_dict = dict(
            zip(
                classes,
                probabilities
            )
        )


        ra_probability = probability_dict.get(
            "RA-Positive",
            0
        )


        ra_probability = round(
            ra_probability * 100,
            2
        )


        # -------------------------
        # RESULT
        # -------------------------

        if prediction == "RA-Positive":

            result = "RA-Positive Indicator"

            level = "HIGH"

        else:

            result = "RA-Negative Indicator"

            level = "LOW"


        return render_template(
            "assessment.html",

            result=result,

            level=level,

            score=ra_probability
        )


    except Exception as e:

        return f"Prediction Error: {e}"


# ============================================================
# RA ANALYTICS
# ============================================================

@app.route("/analytics")
def analytics():

    if "user_id" not in session:

        return redirect("/login")


    # ========================================================
    # LOAD EXCEL DATASET
    # ========================================================

    df = pd.read_excel(
        "data/ra_patients_synthetic(1).xlsx"
    )


    # ========================================================
    # BASIC STATISTICS
    # ========================================================

    total_patients = len(df)


    ra_positive = int(
        (
            df["Diagnosis"] == "RA-Positive"
        ).sum()
    )


    ra_negative = int(
        (
            df["Diagnosis"] == "RA-Negative"
        ).sum()
    )


    if total_patients > 0:

        ra_positive_percent = round(
            (
                ra_positive /
                total_patients
            ) * 100,
            1
        )

    else:

        ra_positive_percent = 0


    # ========================================================
    # AGE GROUP
    # ========================================================

    df["AgeGroup"] = pd.cut(

        df["Age"],

        bins=[
            -1,
            29,
            39,
            49,
            59,
            999
        ],

        labels=[
            "<30",
            "30-39",
            "40-49",
            "50-59",
            "60+"
        ]
    )


    age_groups = [
        "<30",
        "30-39",
        "40-49",
        "50-59",
        "60+"
    ]


    age_positive = []

    age_negative = []


    for group in age_groups:

        age_positive.append(

            int(
                (
                    (df["AgeGroup"] == group)
                    &
                    (
                        df["Diagnosis"]
                        == "RA-Positive"
                    )
                ).sum()
            )
        )


        age_negative.append(

            int(
                (
                    (df["AgeGroup"] == group)
                    &
                    (
                        df["Diagnosis"]
                        == "RA-Negative"
                    )
                ).sum()
            )
        )


    # ========================================================
    # GENDER
    # ========================================================

    genders = [
        "Male",
        "Female"
    ]


    gender_positive = []

    gender_negative = []


    for gender in genders:

        gender_positive.append(

            int(
                (
                    (df["Sex"] == gender)
                    &
                    (
                        df["Diagnosis"]
                        == "RA-Positive"
                    )
                ).sum()
            )
        )


        gender_negative.append(

            int(
                (
                    (df["Sex"] == gender)
                    &
                    (
                        df["Diagnosis"]
                        == "RA-Negative"
                    )
                ).sum()
            )
        )


    # ========================================================
    # SEPARATE RA POSITIVE / NEGATIVE
    # ========================================================

    positive = df[
        df["Diagnosis"] == "RA-Positive"
    ]


    negative = df[
        df["Diagnosis"] == "RA-Negative"
    ]


    # ========================================================
    # INFLAMMATORY INDICATORS
    # ========================================================

    inflammation_positive = [

        round(
            positive["ESR_mm_hr"].mean(),
            2
        ),

        round(
            positive["CRP_mg_L"].mean(),
            2
        ),

        round(
            positive["RF_IU_mL"].mean(),
            2
        ),

        round(
            positive["AntiCCP_U_mL"].mean(),
            2
        )
    ]


    inflammation_negative = [

        round(
            negative["ESR_mm_hr"].mean(),
            2
        ),

        round(
            negative["CRP_mg_L"].mean(),
            2
        ),

        round(
            negative["RF_IU_mL"].mean(),
            2
        ),

        round(
            negative["AntiCCP_U_mL"].mean(),
            2
        )
    ]


    # ========================================================
    # JOINT SYMPTOMS
    # ========================================================

    symptoms_positive = [

        round(
            positive["TenderJointCount"].mean(),
            2
        ),

        round(
            positive["SwollenJointCount"].mean(),
            2
        ),

        round(
            positive["MorningStiffness_min"].mean(),
            2
        )
    ]


    symptoms_negative = [

        round(
            negative["TenderJointCount"].mean(),
            2
        ),

        round(
            negative["SwollenJointCount"].mean(),
            2
        ),

        round(
            negative["MorningStiffness_min"].mean(),
            2
        )
    ]


    # ========================================================
    # DAS28 SEVERITY
    # ========================================================

    def classify_das28(score):

        if score < 3.2:

            return "Low"

        elif score < 5.1:

            return "Moderate"

        else:

            return "High"


    df["Severity"] = df[
        "DAS28_Score"
    ].apply(
        classify_das28
    )


    severity_labels = [
        "Low",
        "Moderate",
        "High"
    ]


    severity_values = [

        int(
            (
                df["Severity"] == "Low"
            ).sum()
        ),

        int(
            (
                df["Severity"] == "Moderate"
            ).sum()
        ),

        int(
            (
                df["Severity"] == "High"
            ).sum()
        )
    ]


    # ========================================================
    # SEND DATA TO ANALYTICS.HTML
    # ========================================================

    return render_template(

        "analytics.html",

        total_patients=total_patients,

        ra_positive=ra_positive,

        ra_negative=ra_negative,

        ra_positive_percent=
            ra_positive_percent,

        age_groups=age_groups,

        age_positive=age_positive,

        age_negative=age_negative,

        genders=genders,

        gender_positive=gender_positive,

        gender_negative=gender_negative,

        inflammation_positive=
            inflammation_positive,

        inflammation_negative=
            inflammation_negative,

        symptoms_positive=
            symptoms_positive,

        symptoms_negative=
            symptoms_negative,

        severity_labels=
            severity_labels,

        severity_values=
            severity_values
    )


# ============================================================
# LOGOUT
# ============================================================

@app.route("/logout")
def logout():

    session.clear()

    return redirect("/login")


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":

    create_database()

    app.run(
        debug=True
    )