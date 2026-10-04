import os
import sqlite3
import json
from datetime import date, timedelta
from werkzeug.utils import secure_filename
from functools import wraps

from flask import Flask, jsonify, redirect, render_template, request, session, url_for, flash, send_from_directory
from werkzeug.security import generate_password_hash, check_password_hash

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, 'instance', 'sfms.db')
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'sfms-demo-secret-change-in-production')
UPLOAD_DIR = os.path.join(BASE_DIR, 'static', 'uploads')
os.makedirs(UPLOAD_DIR, exist_ok=True)
ALLOWED_IMAGE_EXTENSIONS = {'png','jpg','jpeg','webp'}


def db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute('PRAGMA foreign_keys = ON')
    return conn


def init_db():
    conn = db()
    conn.executescript('''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        username TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL,
        role TEXT NOT NULL CHECK(role IN ('staff','student','authority'))
    );
    CREATE TABLE IF NOT EXISTS meal_records (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        meal_date TEXT NOT NULL,
        meal_type TEXT NOT NULL,
        food_item TEXT NOT NULL,
        category TEXT,
        expected_students INTEGER NOT NULL CHECK(expected_students >= 0),
        attendance_count INTEGER,
        food_attendance_count INTEGER,
        predicted_quantity REAL,
        prediction_confidence REAL,
        prediction_model TEXT,
        prediction_confirmed INTEGER NOT NULL DEFAULT 0,
        prepared_quantity REAL,
        preparation_confirmed INTEGER NOT NULL DEFAULT 0,
        served_quantity REAL,
        serving_confirmed INTEGER NOT NULL DEFAULT 0,
        remaining_food REAL,
        plate_waste REAL,
        plate_waste_unit TEXT DEFAULT 'kg',
        total_food_waste REAL,
        waste_percentage REAL,
        waste_snapshot_filename TEXT,
        status TEXT NOT NULL DEFAULT 'menu_saved',
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(meal_date, meal_type, food_item)
    );
    CREATE TABLE IF NOT EXISTS feedback (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        meal_record_id INTEGER NOT NULL,
        student_name TEXT,
        food_quality_rating INTEGER NOT NULL CHECK(food_quality_rating BETWEEN 1 AND 5),
        taste_rating INTEGER NOT NULL CHECK(taste_rating BETWEEN 1 AND 5),
        quantity_satisfaction INTEGER NOT NULL CHECK(quantity_satisfaction BETWEEN 1 AND 5),
        comments TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(meal_record_id) REFERENCES meal_records(id) ON DELETE CASCADE
    );
    CREATE INDEX IF NOT EXISTS idx_meal_date_type ON meal_records(meal_date, meal_type);
    CREATE INDEX IF NOT EXISTS idx_meal_ml_history ON meal_records(attendance_count, prepared_quantity);
    CREATE INDEX IF NOT EXISTS idx_feedback_record ON feedback(meal_record_id);
    CREATE UNIQUE INDEX IF NOT EXISTS uq_feedback_student_meal ON feedback(meal_record_id, student_name);
    CREATE TABLE IF NOT EXISTS dish_suggestions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        dish_name TEXT NOT NULL,
        category TEXT DEFAULT 'Main Course',
        student_name TEXT NOT NULL,
        note TEXT,
        votes INTEGER NOT NULL DEFAULT 1,
        status TEXT NOT NULL DEFAULT 'Open',
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(dish_name, student_name)
    );
    CREATE TABLE IF NOT EXISTS dish_votes (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        suggestion_id INTEGER NOT NULL,
        student_name TEXT NOT NULL,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(suggestion_id, student_name),
        FOREIGN KEY(suggestion_id) REFERENCES dish_suggestions(id) ON DELETE CASCADE
    );
    CREATE TABLE IF NOT EXISTS food_issues (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        student_name TEXT NOT NULL,
        issue_type TEXT NOT NULL,
        description TEXT,
        image_filename TEXT,
        status TEXT NOT NULL DEFAULT 'Open',
        created_at TEXT DEFAULT CURRENT_TIMESTAMP
    );
    CREATE TABLE IF NOT EXISTS meal_intentions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        student_name TEXT NOT NULL,
        meal_date TEXT NOT NULL,
        meal_type TEXT NOT NULL,
        going_to_eat INTEGER NOT NULL CHECK(going_to_eat IN (0,1)),
        source TEXT NOT NULL DEFAULT 'student',
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(student_name, meal_date, meal_type)
    );
    CREATE TABLE IF NOT EXISTS holidays (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT NOT NULL,
        start_date TEXT NOT NULL,
        end_date TEXT NOT NULL,
        notes TEXT,
        expected_home_students INTEGER DEFAULT 0,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP
    );
    CREATE INDEX IF NOT EXISTS idx_suggestions_votes ON dish_suggestions(votes DESC);
    CREATE INDEX IF NOT EXISTS idx_issues_created ON food_issues(created_at DESC);
    ''')
    # Backward-compatible migration for databases created by earlier versions.
    cols = {row['name'] for row in conn.execute("PRAGMA table_info(meal_records)").fetchall()}
    if 'waste_snapshot_filename' not in cols:
        conn.execute("ALTER TABLE meal_records ADD COLUMN waste_snapshot_filename TEXT")
    users = [
        ('Admin', 'admin', generate_password_hash('admin123'), 'staff'),
        ('Student Demo', 'student', generate_password_hash('student123'), 'student'),
    ]
    for u in users:
        conn.execute('INSERT OR IGNORE INTO users(name,username,password,role) VALUES (?,?,?,?)', u)
    # Backward-compatible migration for databases created by earlier SFMS versions.
    cols = {row['name'] for row in conn.execute("PRAGMA table_info(meal_records)").fetchall()}
    if 'food_attendance_count' not in cols:
        conn.execute('ALTER TABLE meal_records ADD COLUMN food_attendance_count INTEGER')
    conn.commit()
    conn.close()


def login_required(role=None):
    def deco(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            if 'user_id' not in session:
                return redirect(url_for('login', next=request.path))
            if role and session.get('role') != role:
                return redirect(url_for('staff_dashboard' if session.get('role') == 'staff' else ('student_home' if session.get('role') == 'student' else 'authority_dashboard')))
            return fn(*args, **kwargs)
        return wrapper
    return deco


def get_record(record_id):
    conn = db()
    row = conn.execute('SELECT * FROM meal_records WHERE id=?', (record_id,)).fetchone()
    conn.close()
    return row


def can_progress(record, required):
    order = ['menu_saved','attendance_saved','prediction_confirmed','preparation_confirmed','served_quantity_confirmed','plate_waste_saved','feedback_submitted']
    try:
        return order.index(record['status']) >= order.index(required)
    except ValueError:
        return False


def build_prediction_model(rows):
    """Train one prediction model for a request instead of retraining per food item."""
    if len(rows) < 4:
        return None, 'Insufficient historical preparation data'
    try:
        import numpy as np
        from sklearn.compose import ColumnTransformer
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.preprocessing import OneHotEncoder

        X = np.array([[r['attendance_count'], r['meal_type'], r['food_item'], r['category'] or 'Other'] for r in rows], dtype=object)
        y = np.array([float(r['prepared_quantity']) for r in rows], dtype=float)
        preprocessor = ColumnTransformer([
            ('categorical', OneHotEncoder(handle_unknown='ignore'), [1, 2, 3])
        ], remainder='passthrough')
        X_encoded = preprocessor.fit_transform(X)
        model = RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            min_samples_leaf=2,
            n_jobs=-1
        )
        model.fit(X_encoded, y)
        return (preprocessor, model, np), None
    except Exception:
        return None, 'Prediction unavailable until historical data is sufficient'


def prediction_for(record, predictor=None):
    """Predict prepared quantity using saved attendance and historical data.

    The expensive Random Forest is trained once per prediction-page request and
    reused for all foods on that page. This keeps the ML result data-driven while
    avoiding one full model training for every menu item.
    """
    attendance = record['food_attendance_count'] if record['food_attendance_count'] is not None else record['attendance_count']
    if attendance is None:
        return None, None, 'Attendance required before prediction'
    attendance = max(0, int(attendance))

    if predictor is None:
        conn = db()
        rows = conn.execute("""
            SELECT attendance_count, meal_type, food_item, category, prepared_quantity
            FROM meal_records
            WHERE attendance_count IS NOT NULL
              AND prepared_quantity IS NOT NULL
              AND prepared_quantity > 0
              AND id <> ?
        """, (record['id'],)).fetchall()
        conn.close()
        predictor, error = build_prediction_model(rows)
        if predictor is None:
            return None, None, error

    preprocessor, model, np = predictor
    try:
        sample = np.array([[attendance, record['meal_type'], record['food_item'], record['category'] or 'Other']], dtype=object)
        transformed = preprocessor.transform(sample)
        predicted = float(max(0, model.predict(transformed)[0]))
        tree_values = np.array([float(tree.predict(transformed)[0]) for tree in model.estimators_])
        mean = max(abs(predicted), 0.01)
        spread = float(np.std(tree_values) / mean)
        confidence = max(50.0, min(96.0, 94.0 - spread * 100.0))
        return round(predicted, 2), round(confidence, 1), 'Random Forest • historical preparation data'
    except Exception:
        return None, None, 'Prediction unavailable until historical data is sufficient'


DEFAULT_MENU_BY_WEEKDAY = {
    0: [('Breakfast','Poha','Main Course'),('Breakfast','Idli','Main Course'),('Breakfast','Tea','Beverage'),('Lunch','Roti','Main Course'),('Lunch','Rice','Main Course'),('Lunch','Dal','Side Dish'),('Lunch','Seasonal Veg','Side Dish'),('Lunch','Salad','Salad'),('High Tea','Samosa','Snack'),('High Tea','Tea','Beverage'),('Dinner','Roti','Main Course'),('Dinner','Jeera Rice','Main Course'),('Dinner','Dal','Side Dish'),('Dinner','Mix Veg','Side Dish')],
    1: [('Breakfast','Aloo Paratha','Main Course'),('Breakfast','Curd','Side Dish'),('Breakfast','Tea','Beverage'),('Lunch','Rice','Main Course'),('Lunch','Rajma','Curry'),('Lunch','Roti','Main Course'),('Lunch','Aloo Gobi','Side Dish'),('Lunch','Salad','Salad'),('High Tea','Bread Pakora','Snack'),('High Tea','Tea','Beverage'),('Dinner','Roti','Main Course'),('Dinner','Rice','Main Course'),('Dinner','Dal','Side Dish'),('Dinner','Paneer Curry','Curry')],
    2: [('Breakfast','Upma','Main Course'),('Breakfast','Boiled Egg','Other'),('Breakfast','Banana','Other'),('Breakfast','Tea','Beverage'),('Lunch','Roti','Main Course'),('Lunch','Rice','Main Course'),('Lunch','Chole','Curry'),('Lunch','Cabbage Peas','Side Dish'),('Lunch','Curd','Side Dish'),('High Tea','Kachori','Snack'),('High Tea','Tea','Beverage'),('Dinner','Roti','Main Course'),('Dinner','Rice','Main Course'),('Dinner','Dal','Side Dish'),('Dinner','Mixed Veg','Side Dish')],
    3: [('Breakfast','Poha','Main Course'),('Breakfast','Jalebi','Sweet'),('Breakfast','Tea','Beverage'),('Lunch','Roti','Main Course'),('Lunch','Rice','Main Course'),('Lunch','Dal','Side Dish'),('Lunch','Paneer Curry','Curry'),('Lunch','Salad','Salad'),('High Tea','Pakora','Snack'),('High Tea','Tea','Beverage'),('Dinner','Roti','Main Course'),('Dinner','Pulao','Main Course'),('Dinner','Dal','Side Dish'),('Dinner','Aloo Matar','Side Dish')],
    4: [('Breakfast','Dosa','Main Course'),('Breakfast','Sambhar','Side Dish'),('Breakfast','Chutney','Side Dish'),('Breakfast','Tea','Beverage'),('Lunch','Roti','Main Course'),('Lunch','Rice','Main Course'),('Lunch','Dal','Side Dish'),('Lunch','Kadhi','Curry'),('Lunch','Salad','Salad'),('High Tea','Samosa','Snack'),('High Tea','Tea','Beverage'),('Dinner','Roti','Main Course'),('Dinner','Veg Pulao','Main Course'),('Dinner','Dal','Side Dish'),('Dinner','Curd','Side Dish')],
    5: [('Breakfast','Poori','Main Course'),('Breakfast','Aloo Sabzi','Side Dish'),('Breakfast','Tea','Beverage'),('Lunch','Rice','Main Course'),('Lunch','Dal Makhani','Curry'),('Lunch','Roti','Main Course'),('Lunch','Mix Veg','Side Dish'),('Lunch','Salad','Salad'),('High Tea','Cutlet','Snack'),('High Tea','Tea','Beverage'),('Dinner','Roti','Main Course'),('Dinner','Rice','Main Course'),('Dinner','Chana Masala','Curry'),('Dinner','Seasonal Veg','Side Dish')],
    6: [('Breakfast','Paratha','Main Course'),('Breakfast','Curd','Side Dish'),('Breakfast','Tea','Beverage'),('Lunch','Veg Biryani','Main Course'),('Lunch','Raita','Side Dish'),('Lunch','Dal Tadka','Curry'),('Lunch','Roti','Main Course'),('Lunch','Salad','Salad'),('High Tea','Pav Bhaji','Snack'),('High Tea','Tea','Beverage'),('Dinner','Roti','Main Course'),('Dinner','Rice','Main Course'),('Dinner','Dal','Side Dish'),('Dinner','Paneer Butter Masala','Curry')]
}

# Backward compatibility for code that referenced DEFAULT_MENU.
DEFAULT_MENU = DEFAULT_MENU_BY_WEEKDAY[date.today().weekday()]
LEGACY_DEFAULT_MENU = [
    ('Breakfast','Poha','Main Course'),('Breakfast','Idli','Main Course'),('Breakfast','Sambhar','Side Dish'),('Breakfast','Banana','Other'),('Breakfast','Tea','Beverage'),
    ('Lunch','Roti','Main Course'),('Lunch','Rice','Main Course'),('Lunch','Dal','Side Dish'),('Lunch','Paneer Curry','Curry'),('Lunch','Salad','Salad'),
    ('High Tea','Samosa','Snack'),('High Tea','Tea','Beverage'),
    ('Dinner','Roti','Main Course'),('Dinner','Jeera Rice','Main Course'),('Dinner','Dal','Side Dish'),('Dinner','Mix Veg','Side Dish'),('Dinner','Curd','Side Dish')
]


def ensure_default_menu(conn, target_date=None):
    """Add a clearly labelled demo/default menu for a date when that date has no menu yet."""
    target_date = target_date or date.today().isoformat()
    created = 0
    weekday = date.fromisoformat(target_date).weekday()
    for meal_type, food_item, category in DEFAULT_MENU_BY_WEEKDAY.get(weekday, DEFAULT_MENU):
        exists = conn.execute(
            'SELECT id FROM meal_records WHERE meal_date=? AND meal_type=? AND food_item=?',
            (target_date, meal_type, food_item)
        ).fetchone()
        if not exists:
            conn.execute("""INSERT INTO meal_records
                (meal_date,meal_type,food_item,category,expected_students,status)
                VALUES (?,?,?,?,?,?)""", (target_date, meal_type, food_item, category, 500, 'menu_saved'))
            created += 1
    return created


def seed_future_demo_data():
    """Seed a demo default menu and an upcoming 3-day holiday for a fresh installation."""
    conn = db()
    today = date.today()
    # Keep a different demo menu available for each day of the current 7-day window.
    # If an untouched older installation still has the old identical demo menu, replace
    # that demo-only data with the new weekday menu. Custom staff menus are never overwritten.
    for offset in range(7):
        target=(today + timedelta(days=offset)).isoformat()
        existing=conn.execute('SELECT meal_type,food_item,category FROM meal_records WHERE meal_date=? ORDER BY id',(target,)).fetchall()
        existing_set={(r['meal_type'],r['food_item'],r['category']) for r in existing}
        legacy_set=set(LEGACY_DEFAULT_MENU)
        if existing and existing_set == legacy_set:
            conn.execute('DELETE FROM meal_records WHERE meal_date=?',(target,))
        ensure_default_menu(conn, target)
    holiday_start = today + timedelta(days=1)
    holiday_end = holiday_start + timedelta(days=2)
    exists = conn.execute('SELECT id FROM holidays WHERE start_date=? AND end_date=?',
                          (holiday_start.isoformat(), holiday_end.isoformat())).fetchone()
    if not exists:
        conn.execute('INSERT INTO holidays(title,start_date,end_date,notes) VALUES (?,?,?,?)',
                     ('Upcoming 3-Day Campus Holiday (Demo)', holiday_start.isoformat(), holiday_end.isoformat(),
                      'Demo holiday for showing student travel plans and reduced meal-demand forecasting.'))
    # Demo training history for the ML-assisted intention forecast. It is clearly synthetic demo data.
    hist = conn.execute('SELECT COUNT(*) AS c FROM meal_intentions').fetchone()['c']
    if hist == 0:
        demo_names = ['Demo Student A','Demo Student B','Demo Student C','Demo Student D']
        for i in range(28):
            d = today - timedelta(days=28-i)
            for mi, meal in enumerate(['Breakfast','Lunch','Dinner']):
                # More no-eat responses on weekends, giving the model a usable pattern.
                eat = 0 if d.weekday() >= 5 and mi != 1 else 1
                if i % 7 == 0 and meal == 'Dinner': eat = 0
                conn.execute('INSERT OR IGNORE INTO meal_intentions(student_name,meal_date,meal_type,going_to_eat,source) VALUES (?,?,?,?,?)',
                             (demo_names[(i+mi)%len(demo_names)], d.isoformat(), meal, eat, 'demo_training'))
    # A few visible responses for the upcoming holiday.
    for idx, name in enumerate(['Demo Student A','Demo Student B','Demo Student C','Demo Student D']):
        for off in range(3):
            d=(holiday_start+timedelta(days=off)).isoformat()
            conn.execute('INSERT OR IGNORE INTO meal_intentions(student_name,meal_date,meal_type,going_to_eat,source) VALUES (?,?,?,?,?)',
                         (name,d,'Lunch',1 if idx==3 else 0,'demo'))
    conn.commit(); conn.close()


def holiday_forecast(conn, holiday):
    """Return ML-assisted eaters forecast and method. Uses LogisticRegression when history supports it."""
    start = date.fromisoformat(holiday['start_date'])
    end = date.fromisoformat(holiday['end_date'])
    rows = conn.execute('SELECT meal_date,meal_type,going_to_eat FROM meal_intentions WHERE source != "demo"').fetchall()
    # Include demo_training rows so a fresh exhibition install can demonstrate the ML pipeline.
    rows = conn.execute('SELECT meal_date,meal_type,going_to_eat FROM meal_intentions').fetchall()
    import numpy as np
    features=[]; labels=[]
    meal_map={'Breakfast':0,'Lunch':1,'High Tea':2,'Dinner':3}
    for r in rows:
        try: d=date.fromisoformat(r['meal_date'])
        except Exception: continue
        days_to=(d-start).days
        features.append([int(d.weekday()), meal_map.get(r['meal_type'],1), int(abs(days_to)), 1 if start <= d <= end else 0])
        labels.append(int(r['going_to_eat']))
    method='Holiday rule-based fallback • waiting for enough student history'
    probability=None
    if len(set(labels)) >= 2 and len(labels) >= 12:
        try:
            from sklearn.linear_model import LogisticRegression
            model=LogisticRegression(max_iter=500).fit(np.array(features),np.array(labels))
            probs=[]
            for meal in ['Breakfast','Lunch','High Tea','Dinner']:
                for off in range((end-start).days+1):
                    d=start+timedelta(days=off)
                    x=np.array([[d.weekday(),meal_map[meal],off,1]])
                    probs.append(float(model.predict_proba(x)[0][1]))
            probability=float(np.mean(probs))
            method='Logistic Regression • student meal-intention history'
        except Exception:
            pass
    # Population estimate uses registered students plus demo training participants.
    registered=conn.execute("SELECT COUNT(*) AS c FROM users WHERE role='student'").fetchone()['c']
    population=max(registered, 4)
    if probability is None:
        responses=conn.execute('SELECT AVG(going_to_eat) AS p FROM meal_intentions WHERE meal_date BETWEEN ? AND ?', (holiday['start_date'],holiday['end_date'])).fetchone()['p']
        probability=float(responses) if responses is not None else 0.35
    return {'probability':round(probability*100,1),'predicted_eaters':max(0,round(population*probability)), 'population':population,'reduction_pct':round(max(0,1-probability)*100,1),'method':method}


def seed_demo_data():
    """Populate realistic demo records on first run so the reviewer can explore immediately."""
    conn = db()
    count = conn.execute("SELECT COUNT(*) AS c FROM meal_records").fetchone()["c"]
    if count:
        conn.close(); return
    demo = [
        ("2026-09-29","Lunch","Rice","Main Course",480,425,46.8,48.0,45.0,3.0,1.8),
        ("2026-09-29","Lunch","Dal","Side Dish",480,425,27.2,28.0,26.0,2.0,1.1),
        ("2026-09-30","Lunch","Rice","Main Course",500,438,48.2,50.0,46.0,4.0,2.4),
        ("2026-09-30","Lunch","Dal","Side Dish",500,438,28.1,29.0,27.0,2.0,1.3),
        ("2026-10-01","Lunch","Rice","Main Course",510,451,49.5,51.0,47.0,4.0,2.1),
        ("2026-10-01","Lunch","Dal","Side Dish",510,451,29.0,30.0,28.0,2.0,1.2),
        ("2026-10-02","Breakfast","Poha","Main Course",500,462,45.0,47.0,44.0,3.0,1.4),
        ("2026-10-02","Lunch","Rice","Main Course",520,468,51.5,53.0,49.0,4.0,2.6),
        ("2026-10-02","Lunch","Dal","Side Dish",520,468,30.0,31.0,29.0,2.0,1.5),
        ("2026-10-02","Lunch","Paneer Curry","Curry",520,468,34.5,36.0,34.0,2.0,1.1),
        ("2026-10-02","Dinner","Chapati","Main Course",520,0,0,0,0,0,0),
        ("2026-10-02","Dinner","Mixed Vegetable","Side Dish",520,0,0,0,0,0,0),
    ]
    for meal_date, meal_type, food_item, category, expected, attendance, predicted, prepared, served, remaining, plate in demo:
        if attendance:
            total=round(remaining+plate,2); pct=round(total/prepared*100,2)
            status="plate_waste_saved"; conf=92.0; model="Demo historical model • seeded data"
        else:
            total=pct=0; conf=None; model=None; status="menu_saved"
            attendance=predicted=prepared=served=remaining=plate=None
        conn.execute("""INSERT INTO meal_records(meal_date,meal_type,food_item,category,expected_students,attendance_count,
            predicted_quantity,prediction_confidence,prediction_model,prediction_confirmed,prepared_quantity,preparation_confirmed,
            served_quantity,serving_confirmed,remaining_food,plate_waste,plate_waste_unit,total_food_waste,waste_percentage,status)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (meal_date,meal_type,food_item,category,expected,attendance,predicted,conf,model,1 if attendance else 0,
             prepared,1 if prepared else 0,served,1 if served else 0,remaining,plate,"kg",total,pct,status))
    rice=conn.execute("SELECT id FROM meal_records WHERE meal_date='2026-10-01' AND meal_type='Lunch' AND food_item='Rice'").fetchone()
    dal=conn.execute("SELECT id FROM meal_records WHERE meal_date='2026-09-30' AND meal_type='Lunch' AND food_item='Dal'").fetchone()
    seed_suggestions=[
        ('Rajma Chawal','Main Course','Demo Student A','Popular comfort-food request'),
        ('Masala Dosa','Breakfast','Demo Student B','Please consider for Sunday breakfast'),
        ('Paneer Tikka','Main Course','Demo Student A','Good dinner option'),
        ('Veg Biryani','Main Course','Demo Student B','Requested for special dinner')
    ]
    for dish,cat,student,note in seed_suggestions:
        conn.execute('INSERT OR IGNORE INTO dish_suggestions(dish_name,category,student_name,note,votes,status) VALUES (?,?,?,?,?,?)',(dish,cat,student,note,1,'Open'))
    if rice:
        conn.execute("INSERT INTO feedback(meal_record_id,student_name,food_quality_rating,taste_rating,quantity_satisfaction,comments) VALUES (?,?,?,?,?,?)",
                     (rice["id"],"Demo Student A",5,4,4,"Good taste and sufficient quantity."))
    if dal:
        conn.execute("INSERT INTO feedback(meal_record_id,student_name,food_quality_rating,taste_rating,quantity_satisfaction,comments) VALUES (?,?,?,?,?,?)",
                     (dal["id"],"Demo Student B",4,5,4,"Dal was tasty today."))
    # Seed visible Student Voice demo data as well, so the reviewer can open voting and issue-report pages with content.
    if conn.execute("SELECT COUNT(*) AS c FROM dish_suggestions").fetchone()["c"] < 4:
        for dish,cat,student,note,votes in [
            ('Rajma Chawal','Main Course','Demo Student A','Popular comfort-food request',12),
            ('Masala Dosa','Breakfast','Demo Student B','Please consider for Sunday breakfast',9),
            ('Paneer Tikka','Main Course','Demo Student A','Good dinner option',7),
            ('Veg Biryani','Main Course','Demo Student B','Requested for special dinner',5)]:
            conn.execute('INSERT OR IGNORE INTO dish_suggestions(dish_name,category,student_name,note,votes,status) VALUES (?,?,?,?,?,?)',(dish,cat,student,note,votes,'Open'))
    if conn.execute("SELECT COUNT(*) AS c FROM food_issues").fetchone()["c"] == 0:
        conn.execute("INSERT INTO food_issues(student_name,issue_type,description,image_filename,status) VALUES (?,?,?,?,?)",
                     ('Demo Student A','Foreign material / insect','Small foreign material noticed while eating lunch. Demo evidence is shown for reviewer navigation.','foreign-material.svg','Open'))
        conn.execute("INSERT INTO food_issues(student_name,issue_type,description,image_filename,status) VALUES (?,?,?,?,?)",
                     ('Demo Student B','Too oily','Curry appeared unusually oily during dinner.','oily-food.svg','Under Review'))
    if conn.execute("SELECT COUNT(*) AS c FROM meal_intentions WHERE source='student'").fetchone()["c"] == 0:
        for student, meal_date, meal_type, eat in [
            ('Demo Student A','2026-10-04','Breakfast',1),
            ('Demo Student A','2026-10-04','Lunch',1),
            ('Demo Student A','2026-10-04','Dinner',0),
            ('Demo Student B','2026-10-05','Breakfast',0),
            ('Demo Student B','2026-10-05','Lunch',0),
        ]:
            conn.execute('INSERT OR IGNORE INTO meal_intentions(student_name,meal_date,meal_type,going_to_eat,source) VALUES (?,?,?,?,?)',(student,meal_date,meal_type,eat,'student'))
    conn.commit(); conn.close()


@app.context_processor
def inject_globals():
    return {'today': date.today().isoformat(), 'current_user': session.get('name'), 'role': session.get('role')}


@app.route('/')
def index():
    if session.get('role') == 'staff':
        return redirect(url_for('staff_dashboard'))
    if session.get('role') == 'student':
        return redirect(url_for('student_home'))
    if session.get('role') == 'authority':
        return redirect(url_for('authority_dashboard'))
    return redirect(url_for('login'))


@app.route('/login')
def login():
    """Role selection page. Users choose a dedicated Staff or Student portal."""
    if session.get('role') == 'staff':
        return redirect(url_for('staff_dashboard'))
    if session.get('role') == 'student':
        return redirect(url_for('student_home'))
    if session.get('role') == 'authority':
        return redirect(url_for('authority_dashboard'))
    return render_template('login.html')


@app.route('/login/<role>', methods=['GET', 'POST'])
def role_login(role):
    if role not in ('staff', 'student', 'authority'):
        return redirect(url_for('login'))
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        if role == 'authority' and username == 'hostel' and password == 'hostel123':
            session.clear()
            session['user_id'] = 'authority-demo'
            session['name'] = 'Hostel Authority'
            session['role'] = 'authority'
            return redirect(url_for('authority_dashboard'))
        conn = db()
        user = conn.execute('SELECT * FROM users WHERE username=? AND role=?', (username, role)).fetchone()
        # Support an older local database containing plain-text demo passwords, then upgrade it.
        valid = False
        if user:
            stored = user['password'] or ''
            try:
                valid = check_password_hash(stored, password)
            except Exception:
                valid = False
            if not valid and stored == password:
                valid = True
                conn.execute('UPDATE users SET password=? WHERE id=?', (generate_password_hash(password), user['id']))
                conn.commit()
        conn.close()
        if valid:
            session.clear()
            session['user_id'] = user['id']
            session['name'] = user['name']
            session['role'] = user['role']
            return redirect(url_for('staff_dashboard' if role == 'staff' else ('student_home' if role == 'student' else 'authority_dashboard')))
        flash('Invalid username or password for this portal.', 'error')
    return render_template('auth.html', role=role, mode='login')


@app.route('/staff/login', methods=['GET', 'POST'])
def staff_login():
    return role_login('staff')


@app.route('/student/login', methods=['GET', 'POST'])
def student_login():
    return role_login('student')


@app.route('/staff/register', methods=['GET', 'POST'])
def staff_register():
    return register('staff')


@app.route('/student/register', methods=['GET', 'POST'])
def student_register():
    return register('student')


@app.route('/register/<role>', methods=['GET', 'POST'])
def register(role):
    if role not in ('staff', 'student'):
        return redirect(url_for('login'))
    if request.method == 'POST':
        name = request.form.get('name', '').strip()
        username = request.form.get('username', '').strip().lower()
        password = request.form.get('password', '')
        confirm = request.form.get('confirm_password', '')
        if len(name) < 2 or len(username) < 3 or len(password) < 6:
            flash('Name, username and password must be valid. Password must be at least 6 characters.', 'error')
        elif password != confirm:
            flash('Passwords do not match.', 'error')
        else:
            conn = db()
            try:
                cur = conn.execute(
                    'INSERT INTO users(name,username,password,role) VALUES (?,?,?,?)',
                    (name, username, generate_password_hash(password), role)
                )
                conn.commit()
                user_id = cur.lastrowid
                session.clear(); session['user_id'] = user_id; session['name'] = name; session['role'] = role
                flash(f'{role.capitalize()} account created successfully.', 'success')
                return redirect(url_for('staff_dashboard' if role == 'staff' else ('student_home' if role == 'student' else 'authority_dashboard')))
            except sqlite3.IntegrityError:
                flash('That username is already registered. Please choose another.', 'error')
            finally:
                conn.close()
    return render_template('auth.html', role=role, mode='register')


@app.route('/logout')
def logout():
    session.clear(); return redirect(url_for('login'))


@app.route('/staff')
@login_required('staff')
def staff_dashboard():
    selected_date = request.args.get('date') or date.today().isoformat()
    meal_type = request.args.get('meal_type') or 'Lunch'
    conn = db()
    records = conn.execute('SELECT * FROM meal_records WHERE meal_date=? AND meal_type=? ORDER BY id DESC', (selected_date, meal_type)).fetchall()
    all_recent = conn.execute('SELECT * FROM meal_records ORDER BY meal_date DESC, id DESC LIMIT 8').fetchall()
    feedback_rows = conn.execute('''SELECT f.*, m.food_item, m.meal_type, m.meal_date FROM feedback f
                                    JOIN meal_records m ON m.id=f.meal_record_id ORDER BY f.id DESC LIMIT 20''').fetchall()
    holidays = conn.execute('SELECT * FROM holidays WHERE end_date>=? ORDER BY start_date LIMIT 5', (date.today().isoformat(),)).fetchall()
    holiday_forecasts=[(h, holiday_forecast(conn,h)) for h in holidays]
    # Dashboard comparison range: one day, two selected dates, or a 7-day week.
    # Defensive handling prevents malformed dates from ever causing an IndexError.
    graph_mode = request.args.get('graph_mode', 'day')
    if graph_mode not in ('day', 'two', 'week'):
        graph_mode = 'day'

    today_iso = date.today().isoformat()
    graph_date = request.args.get('graph_date') or today_iso
    graph_date2 = request.args.get('graph_date2') or graph_date
    graph_week_start = request.args.get('graph_week_start') or graph_date

    def safe_date(value, fallback):
        try:
            return date.fromisoformat(value).isoformat()
        except (TypeError, ValueError):
            return fallback

    graph_date = safe_date(graph_date, today_iso)
    graph_date2 = safe_date(graph_date2, graph_date)
    graph_week_start = safe_date(graph_week_start, graph_date)

    if graph_mode == 'week':
        start = date.fromisoformat(graph_week_start)
        selected_dates = [(start + timedelta(days=i)).isoformat() for i in range(7)]
    elif graph_mode == 'two':
        if graph_date2 == graph_date:
            graph_date2 = (date.fromisoformat(graph_date) + timedelta(days=1)).isoformat()
        selected_dates = [graph_date, graph_date2]
    else:
        selected_dates = [graph_date]

    placeholders = ','.join('?' for _ in selected_dates)
    raw_chart_rows = conn.execute(f'''
        SELECT meal_date, meal_type,
               COALESCE(SUM(predicted_quantity),0) AS predicted_food,
               COALESCE(SUM(prepared_quantity),0) AS actual_food,
               COALESCE(MAX(attendance_count),0) AS attendance,
               COALESCE(MAX(total_food_waste),0) AS total_waste
        FROM meal_records
        WHERE meal_date IN ({placeholders})
        GROUP BY meal_date, meal_type
        ORDER BY meal_date,
                 CASE meal_type WHEN 'Breakfast' THEN 1 WHEN 'Lunch' THEN 2 WHEN 'High Tea' THEN 3 WHEN 'Dinner' THEN 4 ELSE 5 END
    ''', selected_dates).fetchall()

    if graph_mode == 'day':
        # Always keep Breakfast/Lunch/High Tea/Dinner positions stable.
        row_by_meal = {r['meal_type']: r for r in raw_chart_rows}
        meal_order = ['Breakfast', 'Lunch', 'High Tea', 'Dinner']
        chart_labels = meal_order
        chart_predicted = [round(float(row_by_meal[m]['predicted_food'] or 0), 2) if m in row_by_meal else 0 for m in meal_order]
        chart_actual = [round(float(row_by_meal[m]['actual_food'] or 0), 2) if m in row_by_meal else 0 for m in meal_order]
        chart_attendance = [int(float(row_by_meal[m]['attendance'] or 0)) if m in row_by_meal else 0 for m in meal_order]
        chart_waste = [round(float(row_by_meal[m]['total_waste'] or 0), 2) if m in row_by_meal else 0 for m in meal_order]
    else:
        # One bucket per selected date; dates without records remain visible as zero.
        by_date = {d: {'predicted': 0.0, 'actual': 0.0, 'attendance': 0.0, 'waste': 0.0} for d in selected_dates}
        for r in raw_chart_rows:
            bucket = by_date.get(r['meal_date'])
            if bucket is None:
                continue
            bucket['predicted'] += float(r['predicted_food'] or 0)
            bucket['actual'] += float(r['actual_food'] or 0)
            # Attendance is meal-level. Use the maximum meal attendance for the date,
            # rather than adding the same student count across meals.
            bucket['attendance'] = max(bucket['attendance'], float(r['attendance'] or 0))
            bucket['waste'] += float(r['total_waste'] or 0)
        chart_labels = list(selected_dates)
        chart_predicted = [round(by_date[d]['predicted'], 2) for d in selected_dates]
        chart_actual = [round(by_date[d]['actual'], 2) for d in selected_dates]
        chart_attendance = [int(by_date[d]['attendance']) for d in selected_dates]
        chart_waste = [round(by_date[d]['waste'], 2) for d in selected_dates]
    conn.close()
    # Expected students and attendance are meal-level values. Never sum them once per food item.
    summary = {'expected':0,'attendance':0,'prepared':0,'served':0,'remaining':0,'plate_waste':0,'total_waste':0}
    meal_seen = set()
    for r in records:
        meal_key = r['meal_type']
        if meal_key not in meal_seen:
            summary['expected'] = max(summary['expected'], float(r['expected_students'] or 0))
            summary['attendance'] = max(summary['attendance'], float(r['attendance_count'] or 0))
            meal_seen.add(meal_key)
        for k, col in [('prepared','prepared_quantity'),('served','served_quantity'),('remaining','remaining_food'),('plate_waste','plate_waste'),('total_waste','total_food_waste')]:
            summary[k] += float(r[col] or 0)
    summary['waste_pct'] = round((summary['total_waste']/summary['prepared']*100),1) if summary['prepared'] else 0
    return render_template('staff_dashboard.html', records=records, recent=all_recent, feedback=feedback_rows, summary=summary, selected_date=selected_date, meal_type=meal_type, holiday_forecasts=holiday_forecasts, chart_labels=chart_labels, chart_predicted=chart_predicted, chart_actual=chart_actual, chart_attendance=chart_attendance, chart_waste=chart_waste, graph_mode=graph_mode, graph_date=graph_date, graph_date2=graph_date2, graph_week_start=graph_week_start)


@app.route('/staff/menu', methods=['GET', 'POST'])
@login_required('staff')
def menu_management():
    selected_date = request.args.get('date') or date.today().isoformat()
    if request.method == 'POST':
        meal_date=request.form.get('meal_date') or selected_date
        meal_type=request.form.get('meal_type')
        food_item=request.form.get('food_item','').strip()
        category=request.form.get('category','').strip()
        try: expected=int(request.form.get('expected_students','0'))
        except ValueError: expected=-1
        if not meal_date or not meal_type or not food_item or expected < 0:
            flash('Please complete all menu fields with valid values.', 'error')
        else:
            conn=db()
            try:
                existing = conn.execute('SELECT id FROM meal_records WHERE meal_date=? AND meal_type=? LIMIT 1',(meal_date,meal_type)).fetchone()
                conn.execute('INSERT INTO meal_records(meal_date,meal_type,food_item,category,expected_students,status) VALUES (?,?,?,?,?,?)', (meal_date,meal_type,food_item,category,expected,'menu_saved'))
                # Expected students belongs to the meal, so keep every food row aligned.
                conn.execute('UPDATE meal_records SET expected_students=? WHERE meal_date=? AND meal_type=?',(expected,meal_date,meal_type))
                conn.commit(); flash('Menu saved successfully.','success')
                selected_date = meal_date
            except sqlite3.IntegrityError:
                flash('A menu item already exists for this date and meal. Edit the existing record instead.','error')
            finally: conn.close()
    conn=db()
    records=conn.execute('SELECT * FROM meal_records WHERE meal_date=? ORDER BY CASE meal_type WHEN "Breakfast" THEN 1 WHEN "Lunch" THEN 2 ELSE 3 END, id DESC', (selected_date,)).fetchall()
    conn.close()
    return render_template('menu.html', records=records, selected_date=selected_date)


@app.route('/staff/menu/default', methods=['POST'])
@login_required('staff')
def apply_default_menu():
    target=request.form.get('meal_date') or date.today().isoformat()
    conn=db()
    created=ensure_default_menu(conn,target)
    conn.commit(); conn.close()
    if created:
        flash(f'Default menu loaded for {target}. You can edit or delete any item.', 'success')
    else:
        flash('This date already has menu records. Use Edit Menu to change them.', 'error')
    return redirect(url_for('menu_management', date=target))

@app.route('/staff/menu/import-image', methods=['POST'])
@login_required('staff')
def import_menu_image():
    """Create menu records from OCR rows generated in the browser from an uploaded menu image."""
    raw=request.form.get('items','').strip()
    default_date=request.form.get('meal_date') or date.today().isoformat()
    default_meal=request.form.get('meal_type') or 'Lunch'
    try:
        items=json.loads(raw)
    except Exception:
        flash('Could not read the extracted menu data. Please try the image again.', 'error')
        return redirect(url_for('menu_management', date=default_date))
    conn=db(); created=0; skipped=0
    try:
        for item in items:
            food_item=str(item.get('food_item','')).strip()
            if not food_item: continue
            meal_date=str(item.get('meal_date') or default_date)
            meal_type=str(item.get('meal_type') or default_meal)
            category=str(item.get('category') or 'Other').strip()
            try: expected=int(item.get('expected_students') or 500)
            except Exception: expected=500
            expected=max(0,expected)
            try:
                conn.execute('INSERT INTO meal_records(meal_date,meal_type,food_item,category,expected_students,status) VALUES (?,?,?,?,?,?)',
                             (meal_date,meal_type,food_item,category,expected,'menu_saved'))
                created += 1
            except sqlite3.IntegrityError:
                skipped += 1
        conn.commit()
    finally:
        conn.close()
    if created:
        msg=f'{created} menu item(s) created from the uploaded menu image.'
        if skipped: msg += f' {skipped} duplicate item(s) were skipped.'
        flash(msg,'success')
    else:
        flash('No new menu items were created. Check the extracted rows or existing menus.','error')
    return redirect(url_for('menu_management', date=default_date))


@app.route('/staff/menu/<int:record_id>/edit', methods=['POST'])
@login_required('staff')
def edit_menu(record_id):
    meal_date=request.form.get('meal_date'); meal_type=request.form.get('meal_type'); food_item=request.form.get('food_item','').strip(); category=request.form.get('category','').strip(); expected=request.form.get('expected_students','0')
    try: expected=int(expected)
    except: expected=-1
    if expected < 0: flash('Expected students must be zero or more.','error'); return redirect(url_for('menu_management'))
    conn=db()
    try:
        conn.execute('UPDATE meal_records SET meal_date=?,meal_type=?,food_item=?,category=?,expected_students=?,updated_at=CURRENT_TIMESTAMP WHERE id=?', (meal_date,meal_type,food_item,category,expected,record_id)); conn.execute('UPDATE meal_records SET expected_students=? WHERE meal_date=? AND meal_type=?',(expected,meal_date,meal_type)); conn.commit(); flash('Menu updated successfully.','success')
    except sqlite3.IntegrityError: flash('Another menu already uses the same date, meal and item.','error')
    finally: conn.close()
    return redirect(url_for('menu_management', date=meal_date or date.today().isoformat()))


@app.route('/staff/menu/<int:record_id>/delete', methods=['POST'])
@login_required('staff')
def delete_menu(record_id):
    conn=db(); r=conn.execute('SELECT meal_date FROM meal_records WHERE id=?',(record_id,)).fetchone(); conn.execute('DELETE FROM meal_records WHERE id=?',(record_id,)); conn.commit(); conn.close(); flash('Menu item deleted successfully.','success'); return redirect(url_for('menu_management', date=(r['meal_date'] if r else date.today().isoformat())))


@app.route('/staff/attendance', methods=['GET','POST'])
@login_required('staff')
def attendance():
    if request.method == 'POST':
        action = request.form.get('action', 'meal')
        redirect_date = request.form.get('meal_date') or date.today().isoformat()
        conn = db()
        try:
            if action == 'meal':
                meal_date = request.form.get('meal_date','')
                meal_type = request.form.get('meal_type','')
                count = int(request.form.get('attendance_count','-1'))
                rows = conn.execute('SELECT * FROM meal_records WHERE meal_date=? AND meal_type=?', (meal_date, meal_type)).fetchall()
                if not rows:
                    flash('Menu for that meal was not found.','error')
                elif count < 0:
                    flash('Enter a valid attendance count.','error')
                else:
                    # One attendance figure applies to the whole meal. Every food item keeps its own menu record.
                    for r in rows:
                        # Saving/re-saving attendance starts a fresh prediction cycle.
                        # Never keep an old/default prediction when attendance changes.
                        conn.execute("""UPDATE meal_records
                            SET attendance_count=?, predicted_quantity=NULL, prediction_confidence=NULL,
                                prediction_model=NULL, prediction_confirmed=0, status='attendance_saved',
                                updated_at=CURRENT_TIMESTAMP
                            WHERE id=? AND status IN ('menu_saved','attendance_saved','prediction_confirmed')""", (count, r['id']))
                    conn.commit()
                    flash(f'Attendance saved: {count} students for {meal_type}.','success')
            elif action == 'food':
                rid = int(request.form.get('record_id'))
                count = int(request.form.get('food_attendance_count','-1'))
                r = conn.execute('SELECT * FROM meal_records WHERE id=?',(rid,)).fetchone()
                if not r or count < 0:
                    flash('Enter a valid food-specific attendance count.','error')
                elif r['attendance_count'] is None:
                    flash('Save the total meal attendance first.','error')
                else:
                    conn.execute('UPDATE meal_records SET food_attendance_count=?,updated_at=CURRENT_TIMESTAMP WHERE id=?',(count,rid)); conn.commit()
                    flash(f'Food-specific count saved for {r["food_item"]}.','success')
        except ValueError:
            flash('Please enter a valid number.','error')
        finally:
            conn.close()
        return redirect(url_for('attendance', date=redirect_date))

    selected_date=request.args.get('date') or date.today().isoformat()
    conn = db()
    rows = conn.execute('SELECT * FROM meal_records WHERE meal_date=? ORDER BY CASE meal_type WHEN "Breakfast" THEN 1 WHEN "Lunch" THEN 2 WHEN "High Tea" THEN 3 WHEN "Dinner" THEN 4 ELSE 5 END, id', (selected_date,)).fetchall()
    conn.close()
    groups = []
    seen = {}
    for r in rows:
        key = (r['meal_date'], r['meal_type'])
        if key not in seen:
            seen[key] = {'meal_date': r['meal_date'], 'meal_type': r['meal_type'], 'expected': 0, 'attendance': r['attendance_count'], 'items': []}
            groups.append(seen[key])
        g = seen[key]
        g['expected'] = max(g['expected'], r['expected_students'] or 0)
        if g['attendance'] is None and r['attendance_count'] is not None:
            g['attendance'] = r['attendance_count']
        g['items'].append(r)
    return render_template('attendance.html', groups=groups, selected_date=selected_date)


@app.route('/staff/attendance/<int:record_id>/edit', methods=['POST'])
@login_required('staff')
def edit_attendance(record_id):
    count=int(request.form.get('attendance_count','-1'))
    r=get_record(record_id)
    if not r or count < 0:
        flash('Invalid attendance.','error')
    else:
        conn=db()
        conn.execute("""UPDATE meal_records
            SET attendance_count=?, predicted_quantity=NULL, prediction_confidence=NULL,
                prediction_model=NULL, prediction_confirmed=0, status='attendance_saved',
                updated_at=CURRENT_TIMESTAMP
            WHERE meal_date=? AND meal_type=?""",(count,r['meal_date'],r['meal_type']))
        conn.commit(); conn.close(); flash(f'Attendance updated: {count} students for {r["meal_type"]}.','success')
    return redirect(url_for('attendance', date=r['meal_date']))


@app.route('/staff/prediction', methods=['GET','POST'])
@login_required('staff')
def prediction():
    selected_date = request.args.get('date') or date.today().isoformat()
    # Only show the selected date. Historical records remain available by changing the date.
    conn=db(); records=conn.execute("""SELECT * FROM meal_records
        WHERE meal_date=? AND attendance_count IS NOT NULL
          AND status IN ('attendance_saved','prediction_confirmed')
        ORDER BY CASE meal_type WHEN 'Breakfast' THEN 1 WHEN 'Lunch' THEN 2 WHEN 'High Tea' THEN 3 ELSE 4 END, id""", (selected_date,)).fetchall()
    conn.close()

    # Do not train ML just to open the page. Training is needed only when at
    # least one item needs a new prediction or when the user submits a confirm.
    needs_prediction = request.method == 'POST' or any(not r['prediction_confirmed'] for r in records)
    predictor = None
    predictor_error = None
    if needs_prediction:
        conn=db()
        history_rows=conn.execute("""SELECT attendance_count, meal_type, food_item, category, prepared_quantity
            FROM meal_records
            WHERE attendance_count IS NOT NULL AND prepared_quantity IS NOT NULL
              AND prepared_quantity > 0
        """).fetchall()
        conn.close()
        predictor, predictor_error = build_prediction_model(history_rows)

    if request.method=='POST':
        
        try:
            rid=int(request.form.get('record_id','0'))
        except (TypeError, ValueError):
            rid=0
        r=get_record(rid)
        if not r or r['attendance_count'] is None: flash('Complete meal attendance first.','error')
        elif r['status'] not in ('attendance_saved','prediction_confirmed') and not r['prediction_confirmed']: flash('Prediction cannot be changed at this stage.','error')
        else:
            pred, conf, model = prediction_for(r, predictor)
            if pred is None:
                flash(model or 'Prediction is unavailable until sufficient historical data exists.','error')
            else:
                conn=db(); conn.execute("UPDATE meal_records SET predicted_quantity=?,prediction_confidence=?,prediction_model=?,prediction_confirmed=1,status='prediction_confirmed',updated_at=CURRENT_TIMESTAMP WHERE id=?",(pred,conf,model,rid)); conn.commit(); conn.close(); flash(f'Prediction confirmed for {r["food_item"]}.','success')
        return redirect(url_for('prediction', date=r['meal_date'] if r else selected_date))
    groups=[]; seen={}
    for r in records:
        key=(r['meal_date'],r['meal_type'])
        if key not in seen:
            seen[key]={'meal_date':r['meal_date'],'meal_type':r['meal_type'],'attendance':r['attendance_count'],'items':[]}
            groups.append(seen[key])
        item=dict(r)
        # At this point attendance has already been saved, so prediction may be
        # calculated. There is deliberately no prediction for menu-only records.
        if not r['prediction_confirmed']:
            if predictor is not None:
                p,c,m=prediction_for(r, predictor)
            else:
                p,c,m=None,None,predictor_error
            item['predicted_preview']=p; item['confidence_preview']=c; item['model_preview']=m
        seen[key]['items'].append(item)
    return render_template('prediction.html', groups=groups, selected_date=selected_date)


@app.route('/staff/preparation', methods=['GET','POST'])
@login_required('staff')
def preparation():
    selected_date=request.args.get('date') or date.today().isoformat()
    conn=db(); records=conn.execute("SELECT * FROM meal_records WHERE meal_date=? AND prediction_confirmed=1 ORDER BY CASE meal_type WHEN 'Breakfast' THEN 1 WHEN 'Lunch' THEN 2 WHEN 'High Tea' THEN 3 ELSE 4 END,id DESC", (selected_date,)).fetchall(); conn.close()
    if request.method=='POST':
        rid=int(request.form.get('record_id')); qty=float(request.form.get('prepared_quantity','-1')); r=get_record(rid)
        if not r or not r['prediction_confirmed']: flash('Confirm prediction before preparation.','error')
        elif qty < 0: flash('Prepared quantity cannot be negative.','error')
        elif r['status'] not in ('prediction_confirmed','preparation_confirmed'): flash('Preparation has already been confirmed.','error')
        else:
            conn=db(); conn.execute("UPDATE meal_records SET prepared_quantity=?,preparation_confirmed=1,status='preparation_confirmed',updated_at=CURRENT_TIMESTAMP WHERE id=?",(qty,rid)); conn.commit(); conn.close(); flash('Preparation details saved successfully.','success')
        return redirect(url_for('preparation', date=r['meal_date'] if r else selected_date))
    return render_template('preparation.html', records=records, selected_date=selected_date)


@app.route('/staff/serving', methods=['GET','POST'])
@login_required('staff')
def serving():
    selected_date=request.args.get('date') or date.today().isoformat()
    conn=db(); records=conn.execute("SELECT * FROM meal_records WHERE meal_date=? AND preparation_confirmed=1 ORDER BY CASE meal_type WHEN 'Breakfast' THEN 1 WHEN 'Lunch' THEN 2 WHEN 'High Tea' THEN 3 ELSE 4 END,id DESC", (selected_date,)).fetchall(); conn.close()
    if request.method=='POST':
        rid=int(request.form.get('record_id')); qty=float(request.form.get('served_quantity','-1')); r=get_record(rid)
        if not r or not r['preparation_confirmed']: flash('Confirm preparation before serving.','error')
        elif qty < 0 or qty > r['prepared_quantity']: flash('Served quantity must be between 0 and prepared quantity.','error')
        elif r['status'] not in ('preparation_confirmed','served_quantity_confirmed'): flash('Serving has already been confirmed.','error')
        else:
            remaining=round(r['prepared_quantity']-qty,3)
            conn=db(); conn.execute("UPDATE meal_records SET served_quantity=?,remaining_food=?,serving_confirmed=1,status='served_quantity_confirmed',updated_at=CURRENT_TIMESTAMP WHERE id=?",(qty,remaining,rid)); conn.commit(); conn.close(); flash('Served quantity saved successfully.','success')
        return redirect(url_for('serving', date=r['meal_date'] if r else selected_date))
    return render_template('serving.html', records=records, selected_date=selected_date)


@app.route('/staff/waste', methods=['GET','POST'])
@login_required('staff')
def waste():
    if request.method=='POST':
        action=request.form.get('action','meal')
        try: value=float(request.form.get('plate_waste','-1'))
        except ValueError: value=-1
        unit=request.form.get('unit','kg')
        snapshot=request.files.get('waste_snapshot')
        if value < 0:
            flash('Waste weight cannot be negative.','error'); return redirect(url_for('waste', date=request.form.get('meal_date') or date.today().isoformat()))
        kg=round(value/1000 if unit=='g' else value,3)
        conn=db()
        try:
            if action=='meal':
                meal_date=request.form.get('meal_date'); meal_type=request.form.get('meal_type')
                rows=conn.execute('SELECT * FROM meal_records WHERE meal_date=? AND meal_type=? AND serving_confirmed=1 ORDER BY id',(meal_date,meal_type)).fetchall()
                if not rows:
                    flash('Complete serving before plate waste.','error')
                else:
                    prepared=round(sum(float(r['prepared_quantity'] or 0) for r in rows),3)
                    remaining=round(sum(float(r['remaining_food'] or 0) for r in rows),3)
                    total=round(remaining+kg,3); pct=round(total/prepared*100,2) if prepared else 0
                    first_id=rows[0]['id']
                    snapshot_filename=None
                    if snapshot and snapshot.filename:
                        ext=snapshot.filename.rsplit('.',1)[-1].lower() if '.' in snapshot.filename else ''
                        if ext not in ALLOWED_IMAGE_EXTENSIONS:
                            flash('Waste snapshot must be PNG, JPG, JPEG or WEBP.','error'); return redirect(url_for('waste', date=meal_date or date.today().isoformat()))
                        safe=secure_filename(snapshot.filename)
                        snapshot_filename=f"waste_{session.get('user_id')}_{int(__import__('time').time())}_{safe}"
                        snapshot.save(os.path.join(UPLOAD_DIR,snapshot_filename))
                    conn.execute("UPDATE meal_records SET plate_waste=NULL,total_food_waste=0,waste_percentage=NULL,waste_snapshot_filename=NULL WHERE meal_date=? AND meal_type=?",(meal_date,meal_type))
                    conn.execute("UPDATE meal_records SET plate_waste=?,plate_waste_unit=?,total_food_waste=?,waste_percentage=?,waste_snapshot_filename=?,status='plate_waste_saved',updated_at=CURRENT_TIMESTAMP WHERE id=?",(kg,unit,total,pct,snapshot_filename,first_id))
                    conn.execute("UPDATE meal_records SET status='plate_waste_saved',updated_at=CURRENT_TIMESTAMP WHERE meal_date=? AND meal_type=?",(meal_date,meal_type))
                    conn.commit(); flash(f'Total plate waste saved: {kg} kg for {meal_type}.','success')
            elif action=='food':
                rid=int(request.form.get('record_id')); r=conn.execute('SELECT * FROM meal_records WHERE id=?',(rid,)).fetchone()
                if not r or not r['serving_confirmed']:
                    flash('Complete serving before individual plate waste.','error')
                else:
                    conn.execute("UPDATE meal_records SET plate_waste=?,plate_waste_unit=?,updated_at=CURRENT_TIMESTAMP WHERE id=?",(kg,unit,rid))
                    rows=conn.execute('SELECT * FROM meal_records WHERE meal_date=? AND meal_type=? AND serving_confirmed=1 ORDER BY id',(r['meal_date'],r['meal_type'])).fetchall()
                    prepared=round(sum(float(x['prepared_quantity'] or 0) for x in rows),3); remaining=round(sum(float(x['remaining_food'] or 0) for x in rows),3); plate=round(sum(float(x['plate_waste'] or 0) for x in rows),3); total=round(remaining+plate,3); pct=round(total/prepared*100,2) if prepared else 0
                    first_id=rows[0]['id']
                    conn.execute("UPDATE meal_records SET total_food_waste=0,waste_percentage=NULL WHERE meal_date=? AND meal_type=?",(r['meal_date'],r['meal_type']))
                    conn.execute("UPDATE meal_records SET total_food_waste=?,waste_percentage=?,status='plate_waste_saved',updated_at=CURRENT_TIMESTAMP WHERE id=?",(total,pct,first_id))
                    conn.execute("UPDATE meal_records SET status='plate_waste_saved',updated_at=CURRENT_TIMESTAMP WHERE meal_date=? AND meal_type=?",(r['meal_date'],r['meal_type']))
                    conn.commit(); flash(f'Individual plate waste saved for {r["food_item"]}.','success')
        finally:
            conn.close()
        return redirect(url_for('waste', date=request.form.get('meal_date') or date.today().isoformat()))
    selected_date=request.args.get('date') or date.today().isoformat()
    conn=db(); records=conn.execute("SELECT * FROM meal_records WHERE meal_date=? AND serving_confirmed=1 ORDER BY meal_date DESC, CASE meal_type WHEN 'Breakfast' THEN 1 WHEN 'Lunch' THEN 2 WHEN 'High Tea' THEN 3 ELSE 4 END, id", (selected_date,)).fetchall(); conn.close()
    groups=[]; seen={}
    for r in records:
        key=(r['meal_date'],r['meal_type'])
        if key not in seen:
            seen[key]={'meal_date':r['meal_date'],'meal_type':r['meal_type'],'prepared':0,'served':0,'remaining':0,'plate_waste':0,'total_food_waste':0,'waste_percentage':None,'saved':False,'items':[],'snapshot':None}
            groups.append(seen[key])
        g=seen[key]; g['prepared']+=float(r['prepared_quantity'] or 0); g['served']+=float(r['served_quantity'] or 0); g['remaining']+=float(r['remaining_food'] or 0); g['plate_waste']+=float(r['plate_waste'] or 0); g['items'].append(dict(r))
        if r['waste_snapshot_filename']: g['snapshot']=r['waste_snapshot_filename']
        if r['total_food_waste'] is not None and r['total_food_waste']>0: g['total_food_waste']=r['total_food_waste']; g['waste_percentage']=r['waste_percentage']; g['saved']=True
    return render_template('waste.html', groups=groups, selected_date=selected_date)


@app.route('/staff/feedback')
@login_required('staff')
def feedback_staff():
    conn=db()
    rows=conn.execute('''SELECT f.*,m.food_item,m.meal_type,m.meal_date FROM feedback f JOIN meal_records m ON m.id=f.meal_record_id ORDER BY f.id DESC''').fetchall()
    suggestions=conn.execute('SELECT * FROM dish_suggestions ORDER BY votes DESC, created_at DESC').fetchall()
    issues=conn.execute('SELECT * FROM food_issues ORDER BY id DESC').fetchall()
    conn.close()
    return render_template('feedback_staff.html', feedback=rows, suggestions=suggestions, issues=issues)


@app.route('/staff/holidays', methods=['GET','POST'])
@login_required('staff')
def staff_holidays():
    return redirect(url_for('authority_dashboard'))


@app.route('/authority', methods=['GET','POST'])
@login_required('authority')
def authority_dashboard():
    if request.method == 'POST':
        title=request.form.get('title','').strip(); start=request.form.get('start_date',''); end=request.form.get('end_date',''); notes=request.form.get('notes','').strip()
        try: sd=date.fromisoformat(start); ed=date.fromisoformat(end); expected=max(0,int(request.form.get('expected_home_students','0') or 0))
        except Exception: sd=ed=None; expected=0
        if not title or not sd or not ed or ed < sd:
            flash('Enter a valid holiday title and date range.','error')
        else:
            conn=db(); conn.execute('INSERT INTO holidays(title,start_date,end_date,notes,expected_home_students) VALUES (?,?,?,?,?)',(title,start,end,notes,expected)); conn.commit(); conn.close(); flash('Academic-calendar holiday added successfully.','success')
    conn=db()
    today=date.today().isoformat()
    holidays=conn.execute('SELECT * FROM holidays WHERE end_date>=? ORDER BY start_date', (today,)).fetchall()
    holiday_rows=[]
    for h in holidays:
        f=holiday_forecast(conn,h)
        actual=conn.execute("SELECT COUNT(DISTINCT student_name) AS c FROM meal_intentions WHERE meal_date BETWEEN ? AND ? AND going_to_eat=0 AND source='student'",(h['start_date'],h['end_date'])).fetchone()['c']
        holiday_rows.append((h,f,actual))
    # Student review summary
    student_counts=conn.execute("SELECT COUNT(*) AS c FROM users WHERE role='student'").fetchone()['c']
    today_home=conn.execute("SELECT COUNT(DISTINCT student_name) AS c FROM meal_intentions WHERE meal_date=? AND going_to_eat=0 AND source='student'",(today,)).fetchone()['c']
    # Staff operational review for today
    staff_total=conn.execute("SELECT COUNT(*) AS c FROM meal_records WHERE meal_date=?",(today,)).fetchone()['c']
    attendance_done=conn.execute("SELECT COUNT(*) AS c FROM meal_records WHERE meal_date=? AND attendance_count IS NOT NULL",(today,)).fetchone()['c']
    prediction_done=conn.execute("SELECT COUNT(*) AS c FROM meal_records WHERE meal_date=? AND prediction_confirmed=1",(today,)).fetchone()['c']
    preparation_done=conn.execute("SELECT COUNT(*) AS c FROM meal_records WHERE meal_date=? AND preparation_confirmed=1",(today,)).fetchone()['c']
    waste_done=conn.execute("SELECT COUNT(*) AS c FROM meal_records WHERE meal_date=? AND status='plate_waste_saved'",(today,)).fetchone()['c']
    conn.close()
    return render_template('authority_dashboard.html', holiday_rows=holiday_rows, student_counts=student_counts, today_home=today_home, today=today, staff_total=staff_total, attendance_done=attendance_done, prediction_done=prediction_done, preparation_done=preparation_done, waste_done=waste_done)


@app.route('/authority/students')
@login_required('authority')
def authority_students():
    """Read-only student review for Hostel Authority. No student records can be edited here."""
    conn=db()
    students=conn.execute("SELECT id,name,username FROM users WHERE role='student' ORDER BY name").fetchall()
    today=date.today().isoformat()
    intentions=conn.execute("""SELECT mi.student_name,mi.meal_date,mi.meal_type,mi.going_to_eat,mi.created_at
                             FROM meal_intentions mi
                             WHERE mi.source='student'
                             ORDER BY mi.meal_date DESC, mi.student_name, mi.meal_type""").fetchall()
    feedback=conn.execute("""SELECT f.student_name,m.meal_date,m.meal_type,m.food_item,
                             f.food_quality_rating,f.taste_rating,f.quantity_satisfaction,f.comments,f.created_at
                             FROM feedback f JOIN meal_records m ON m.id=f.meal_record_id
                             ORDER BY f.created_at DESC LIMIT 80""").fetchall()
    issues=conn.execute("""SELECT student_name,issue_type,description,status,image_filename,created_at
                           FROM food_issues ORDER BY created_at DESC LIMIT 80""").fetchall()
    suggestions=conn.execute("""SELECT dish_name,category,student_name,votes,status,created_at
                                FROM dish_suggestions ORDER BY votes DESC,created_at DESC""").fetchall()
    conn.close()
    return render_template('authority_students.html', students=students, intentions=intentions, feedback=feedback, issues=issues, suggestions=suggestions, today=today)


@app.route('/authority/staff')
@login_required('authority')
def authority_staff():
    """Read-only mess staff operational review for Hostel Authority."""
    conn=db()
    today=date.today().isoformat()
    active_date=request.args.get('date') or today
    records=conn.execute("""SELECT * FROM meal_records WHERE meal_date=?
        ORDER BY CASE meal_type WHEN 'Breakfast' THEN 1 WHEN 'Lunch' THEN 2 WHEN 'High Tea' THEN 3 ELSE 4 END, id""", (active_date,)).fetchall()
    summary={'attendance':0,'expected':0,'prepared':0,'served':0,'remaining':0,'plate_waste':0,'total_waste':0}
    meals=set()
    for r in records:
        if r['meal_type'] not in meals:
            summary['attendance']=max(summary['attendance'],float(r['attendance_count'] or 0))
            summary['expected']=max(summary['expected'],float(r['expected_students'] or 0))
            meals.add(r['meal_type'])
        for k,col in [('prepared','prepared_quantity'),('served','served_quantity'),('remaining','remaining_food'),('plate_waste','plate_waste'),('total_waste','total_food_waste')]: summary[k]+=float(r[col] or 0)
    conn.close()
    return render_template('authority_staff.html', records=records, summary=summary, today=today, active_date=active_date, using_latest=False)


@app.route('/student')
@login_required('student')
def student_home():
    selected_date=date.today().isoformat()
    conn=db(); meals=conn.execute('SELECT * FROM meal_records WHERE meal_date=? ORDER BY CASE meal_type WHEN "Breakfast" THEN 1 WHEN "Lunch" THEN 2 ELSE 3 END, id', (selected_date,)).fetchall(); conn.close()
    return render_template('student.html', meals=meals)


@app.route('/student/meal-plan', methods=['GET','POST'])
@login_required('student')
def student_meal_plan():
    today=date.today()
    if request.method=='POST':
        meal_date=request.form.get('meal_date','')
        meal_type=request.form.get('meal_type','')
        eat=request.form.get('going_to_eat')
        try: d=date.fromisoformat(meal_date)
        except Exception: d=None
        if not d or d not in (today,today+timedelta(days=1)) or meal_type not in ('Breakfast','Lunch','High Tea','Dinner') or eat not in ('0','1'):
            flash('Please choose today or tomorrow and a valid meal option.','error')
        else:
            conn=db()
            conn.execute("""INSERT INTO meal_intentions(student_name,meal_date,meal_type,going_to_eat,source) VALUES (?,?,?,?,?)
                            ON CONFLICT(student_name,meal_date,meal_type) DO UPDATE SET going_to_eat=excluded.going_to_eat, source='student' """,
                         (session.get('name'),meal_date,meal_type,int(eat),'student'))
            conn.commit(); conn.close(); flash('Your meal plan was saved and will help the mess team estimate preparation.','success')
    conn=db()
    dates=[today,today+timedelta(days=1)]
    menus={d.isoformat():conn.execute('SELECT * FROM meal_records WHERE meal_date=? ORDER BY CASE meal_type WHEN "Breakfast" THEN 1 WHEN "Lunch" THEN 2 WHEN "High Tea" THEN 3 ELSE 4 END,id',(d.isoformat(),)).fetchall() for d in dates}
    intentions=conn.execute('SELECT * FROM meal_intentions WHERE student_name=? AND meal_date IN (?,?) ORDER BY meal_date, meal_type',(session.get('name'),dates[0].isoformat(),dates[1].isoformat())).fetchall()
    holidays=conn.execute('SELECT * FROM holidays WHERE end_date>=? ORDER BY start_date LIMIT 5',(today.isoformat(),)).fetchall()
    forecasts=[(h,holiday_forecast(conn,h)) for h in holidays]
    conn.close()
    return render_template('student_meal_plan.html', menus=menus, dates=dates, intentions=intentions, holidays=holidays, forecasts=forecasts)


@app.route('/student/holiday-plan/<int:holiday_id>', methods=['POST'])
@login_required('student')
def student_holiday_plan(holiday_id):
    choice=request.form.get('choice')
    if choice not in ('home','stay'):
        flash('Please choose whether you are going home or staying on campus.','error')
        return redirect(url_for('student_meal_plan'))
    conn=db(); holiday=conn.execute('SELECT * FROM holidays WHERE id=?',(holiday_id,)).fetchone()
    if not holiday:
        conn.close(); flash('Holiday information was not found.','error'); return redirect(url_for('student_meal_plan'))
    start=date.fromisoformat(holiday['start_date']); end=date.fromisoformat(holiday['end_date'])
    if end < date.today():
        conn.close(); flash('This holiday has already ended.','error'); return redirect(url_for('student_meal_plan'))
    eat=0 if choice=='home' else 1
    d=start
    while d<=end:
        for meal in ('Breakfast','Lunch','High Tea','Dinner'):
            conn.execute("""INSERT INTO meal_intentions(student_name,meal_date,meal_type,going_to_eat,source) VALUES (?,?,?,?,?)
                            ON CONFLICT(student_name,meal_date,meal_type) DO UPDATE SET going_to_eat=excluded.going_to_eat, source='student'""",
                         (session.get('name'),d.isoformat(),meal,eat,'student'))
        d += timedelta(days=1)
    conn.commit(); conn.close()
    flash(('Home-going plan saved for the whole holiday. The mess team can reduce preparation for those meals.' if choice=='home' else 'Stay-on-campus plan saved for the whole holiday.'),'success')
    return redirect(url_for('student_meal_plan'))


@app.route('/student/feedback')
@login_required('student')
def student_feedback_page():
    # Show the complete available menu, grouped by date and meal. Students choose
    # only the food(s) they actually want to rate instead of filling a form for every item.
    conn=db()
    meals=conn.execute("""SELECT m.*, EXISTS(SELECT 1 FROM feedback f WHERE f.meal_record_id=m.id AND f.student_name=?) AS already_feedback
                        FROM meal_records m
                        WHERE m.status IN ('plate_waste_saved','feedback_submitted')
                        ORDER BY m.meal_date DESC, CASE m.meal_type WHEN 'Breakfast' THEN 1 WHEN 'Lunch' THEN 2 WHEN 'High Tea' THEN 3 ELSE 4 END, m.id""", (session.get('name'),)).fetchall()
    conn.close()
    return render_template('student_feedback.html', meals=meals)

@app.route('/student/feedback/bulk', methods=['POST'])
@login_required('student')
def student_feedback_bulk():
    ids=request.form.getlist('record_ids')
    if not ids:
        flash('Select at least one food or meal item to rate.','error')
        return redirect(url_for('student_feedback_page'))
    try:
        quality=int(request.form['food_quality_rating']); taste=int(request.form['taste_rating']); quantity=int(request.form['quantity_satisfaction'])
    except Exception:
        flash('Please select all ratings.','error')
        return redirect(url_for('student_feedback_page'))
    if not all(1<=x<=5 for x in (quality,taste,quantity)):
        flash('Ratings must be between 1 and 5.','error')
        return redirect(url_for('student_feedback_page'))
    conn=db(); added=0; skipped=0
    for raw_id in ids:
        try: rid=int(raw_id)
        except Exception: continue
        r=conn.execute("SELECT id,status FROM meal_records WHERE id=?",(rid,)).fetchone()
        if not r or r['status'] not in ('plate_waste_saved','feedback_submitted'):
            continue
        existing=conn.execute('SELECT id FROM feedback WHERE meal_record_id=? AND student_name=?',(rid,session.get('name'))).fetchone()
        if existing:
            skipped += 1
            continue
        conn.execute('INSERT INTO feedback(meal_record_id,student_name,food_quality_rating,taste_rating,quantity_satisfaction,comments) VALUES (?,?,?,?,?,?)',
                     (rid,session.get('name'),quality,taste,quantity,request.form.get('comments','').strip()))
        conn.execute("UPDATE meal_records SET status='feedback_submitted',updated_at=CURRENT_TIMESTAMP WHERE id=?",(rid,))
        added += 1
    conn.commit(); conn.close()
    if added:
        msg=f'Feedback submitted for {added} selected food item(s).'+(f' {skipped} already had feedback.' if skipped else '')
        flash(msg,'success')
    else:
        flash('No new feedback was submitted. Selected items may already have feedback.','error')
    return redirect(url_for('student_feedback_page'))

@app.route('/student/feedback/<int:record_id>', methods=['POST'])
@login_required('student')
def student_feedback(record_id):
    r=get_record(record_id)
    if not r: flash('Meal not found.','error'); return redirect(url_for('student_feedback_page'))
    if r['status'] not in ('plate_waste_saved','feedback_submitted'):
        flash('Feedback opens after plate waste is recorded for this meal.', 'error'); return redirect(url_for('student_feedback_page'))
    try:
        quality=int(request.form['food_quality_rating']); taste=int(request.form['taste_rating']); quantity=int(request.form['quantity_satisfaction'])
    except: flash('Please select all ratings.','error'); return redirect(url_for('student_feedback_page'))
    if not all(1<=x<=5 for x in (quality,taste,quantity)): flash('Ratings must be between 1 and 5.','error'); return redirect(url_for('student_feedback_page'))
    conn=db()
    existing=conn.execute('SELECT id FROM feedback WHERE meal_record_id=? AND student_name=?',(record_id,session.get('name'))).fetchone()
    if existing:
        conn.close(); flash('You have already submitted feedback for this meal.','error'); return redirect(url_for('student_feedback_page'))
    conn.execute('INSERT INTO feedback(meal_record_id,student_name,food_quality_rating,taste_rating,quantity_satisfaction,comments) VALUES (?,?,?,?,?,?)',(record_id,session.get('name'),quality,taste,quantity,request.form.get('comments','').strip()))
    conn.execute("UPDATE meal_records SET status='feedback_submitted',updated_at=CURRENT_TIMESTAMP WHERE id=?",(record_id,))
    conn.commit(); conn.close(); flash('Feedback submitted successfully.','success'); return redirect(url_for('student_feedback_page'))

@app.route('/student/voting')
@login_required('student')
def student_voting():
    conn=db()
    suggestions=conn.execute("SELECT s.*, EXISTS(SELECT 1 FROM dish_votes v WHERE v.suggestion_id=s.id AND v.student_name=?) AS voted FROM dish_suggestions s ORDER BY s.votes DESC, s.created_at DESC",(session.get('name'),)).fetchall()
    conn.close()
    return render_template('student_voting.html', suggestions=suggestions)

@app.route('/student/preferences')
@login_required('student')
def student_preferences():
    return redirect(url_for('student_voting'))

@app.route('/student/preferences/suggest', methods=['POST'])
@login_required('student')
def suggest_dish():
    dish=request.form.get('dish_name','').strip()
    category=request.form.get('category','Main Course').strip()
    note=request.form.get('note','').strip()
    if len(dish)<2:
        flash('Please enter a dish name.','error')
        return redirect(url_for('student_voting'))
    conn=db()
    try:
        conn.execute("INSERT INTO dish_suggestions(dish_name,category,student_name,note,votes,status) VALUES (?,?,?,?,1,'Open')",(dish,category,session.get('name'),note))
        conn.commit(); flash('Dish suggestion added. Students can now vote for it.','success')
    except sqlite3.IntegrityError:
        flash('You have already suggested this dish. Vote for the existing suggestion instead.','error')
    finally: conn.close()
    return redirect(url_for('student_voting'))

@app.route('/student/preferences/vote/<int:suggestion_id>', methods=['POST'])
@login_required('student')
def vote_dish(suggestion_id):
    conn=db()
    try:
        exists=conn.execute('SELECT id FROM dish_suggestions WHERE id=?',(suggestion_id,)).fetchone()
        if not exists:
            flash('Dish suggestion not found.','error')
            return redirect(url_for('student_voting'))
        conn.execute('INSERT INTO dish_votes(suggestion_id,student_name) VALUES (?,?)',(suggestion_id,session.get('name')))
        conn.execute('UPDATE dish_suggestions SET votes=votes+1 WHERE id=?',(suggestion_id,))
        conn.commit(); flash('Your vote was added. The mess team can use the vote count to plan future dishes.','success')
    except sqlite3.IntegrityError:
        flash('You have already voted for this dish.','error')
    finally: conn.close()
    return redirect(url_for('student_voting'))

@app.route('/student/report-issue')
@login_required('student')
def student_report_issue():
    conn=db()
    issues=conn.execute('SELECT * FROM food_issues WHERE student_name=? ORDER BY id DESC',(session.get('name'),)).fetchall()
    conn.close()
    return render_template('student_issue.html', issues=issues)

@app.route('/student/food-issue', methods=['POST'])
@login_required('student')
def report_food_issue():
    issue_type=request.form.get('issue_type','').strip()
    description=request.form.get('description','').strip()
    image=request.files.get('image')
    if not issue_type:
        flash('Please select the food issue type.','error')
        return redirect(url_for('student_report_issue'))
    filename=None
    if image and image.filename:
        ext=image.filename.rsplit('.',1)[-1].lower() if '.' in image.filename else ''
        if ext not in ALLOWED_IMAGE_EXTENSIONS:
            flash('Evidence image must be PNG, JPG, JPEG or WEBP.','error')
            return redirect(url_for('student_report_issue'))
        safe=secure_filename(image.filename)
        filename=f"{session.get('user_id')}_{int(__import__('time').time())}_{safe}"
        image.save(os.path.join(UPLOAD_DIR,filename))
    conn=db()
    conn.execute("INSERT INTO food_issues(student_name,issue_type,description,image_filename,status) VALUES (?,?,?,?, 'Open')",(session.get('name'),issue_type,description,filename))
    conn.commit(); conn.close()
    flash('Food quality issue reported to the mess team.','success')
    return redirect(url_for('student_report_issue'))

@app.route('/staff/waste/uploads/<path:filename>')
@login_required('staff')
def staff_waste_upload(filename):
    return send_from_directory(UPLOAD_DIR, filename)

@app.route('/student/uploads/<path:filename>')
@login_required()
def student_upload(filename):
    return send_from_directory(UPLOAD_DIR, filename)

@app.route('/api/health')
def health():
    return jsonify({'status':'ok','service':'Smart Food Waste Management System'})


@app.route('/api/records/<int:record_id>')
@login_required()
def api_record(record_id):
    r=get_record(record_id)
    return jsonify(dict(r)) if r else (jsonify({'error':'not found'}),404)


init_db()
seed_demo_data()
seed_future_demo_data()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT',5000)), debug=True)
