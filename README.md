# Smart Food Waste Management System — Review Ready v5

## What is included
- Separate Staff and Student portals with login/register
- Staff workflow: Menu → Attendance → ML Prediction → Preparation → Serving → Plate Waste → Feedback → Dashboard
- Calendar-based menu planning
- Menu image OCR import with editable preview
- Seeded demo menu/waste/feedback data on first run
- Student meal ratings and comments
- **Student Voice:** dedicated My Feedback, Dish Voting, and Report Food Issue pages
- **Dish voting:** students suggest dishes and vote once per dish; vote counts are visible to staff
- **Food Quality Issue reporting:** categories for foreign material/insect, too oily, burnt/overcooked, undercooked, salty/spicy and hygiene concerns
- Dedicated issue page with illustrated issue examples and evidence-photo upload
- Optional evidence photo upload for food-quality issues
- Staff view of dish voting and food-quality reports
- SQLite database, duplicate prevention, status workflow, responsive UI
- Gunicorn/Render deployment files

## Demo accounts
Staff: `admin` / `admin123`
Student: `student` / `student123`

## Run locally
```bash
python -m pip install -r requirements.txt
python app.py
```
Open http://127.0.0.1:5000

## Notes
- A fresh database is seeded with sample data so the reviewer can explore immediately.
- Menu image OCR uses Tesseract.js from a CDN in the browser; an internet connection is needed when first loading the OCR library.
- Uploaded food-quality evidence photos are stored under `static/uploads` for the local demo. For production, use object storage or a persistent disk and add authentication/retention rules appropriate for your deployment.

## v6 planning additions
- Staff Menu Management now has a **Default Menu** section that can load a ready-made breakfast/lunch/high-tea/dinner menu for a selected date.
- Staff has a dedicated **Edit Menu** section with direct edit and manual delete controls.
- Students have a **Meal Plan** page to mark today's and tomorrow's Breakfast/Lunch/High Tea/Dinner as “I will eat” or “I will not eat”.
- Staff has a **Holiday & Demand** page where upcoming holidays can be entered and monitored.
- Students can mark an upcoming holiday as **I am going home** or **I will stay** for the complete holiday period.
- The system provides an **ML-assisted meal-demand forecast** using historical student meal-intention data and shows predicted eaters plus suggested preparation reduction.
- The holiday feature is designed to support the project logic: more students going home → lower predicted meal demand → lower recommended preparation → less avoidable food waste.

## ML prediction behavior (v19)
- Prediction is generated only after meal attendance is saved.
- There is no fixed kg-per-student fallback and no hard-coded default prediction.
- The model uses saved attendance plus meal, food item and category context.
- The target is historical actual prepared quantity stored in the database.
- A Random Forest regression model is trained from prior records with actual preparation data.
- If sufficient historical target data is not available, the system does not invent a number; it asks for more actual preparation records.
