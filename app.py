# app.py (full file — replace or merge with your current app.py)
from flask import Flask, request, jsonify, send_from_directory, send_file
import joblib
import numpy as np
import os
import io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import random

app = Flask(__name__, static_folder="static", static_url_path="/static")

MODEL_PATH = "model/calorie_model.joblib"
FOOD_DB_PATH = "model/food_db.csv"

# --- Load ML model ---
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("Model not found. Run train.py first.")
data = joblib.load(MODEL_PATH)
model = data["model"]
features = data["features"]

# --- Load food database (CSV) ---
if not os.path.exists(FOOD_DB_PATH):
    raise FileNotFoundError("Food DB not found. Create model/food_db.csv")
food_df = pd.read_csv(FOOD_DB_PATH)

# helper: macro calculation (same as before)
def macro_split(calories, goal="maintenance"):
    if goal == "weight_loss":
        protein_pct = 0.30
        fat_pct = 0.30
        carb_pct = 0.40
    elif goal == "muscle_gain":
        protein_pct = 0.25
        fat_pct = 0.25
        carb_pct = 0.50
    else:  # maintenance
        protein_pct = 0.20
        fat_pct = 0.30
        carb_pct = 0.50

    carbs_kcal = calories * carb_pct
    protein_kcal = calories * protein_pct
    fat_kcal = calories * fat_pct

    carbs_g = round(carbs_kcal / 4)
    protein_g = round(protein_kcal / 4)
    fat_g = round(fat_kcal / 9)

    return {
        "carbs_g": carbs_g,
        "protein_g": protein_g,
        "fat_g": fat_g,
        "carbs_pct": int(carb_pct*100),
        "protein_pct": int(protein_pct*100),
        "fat_pct": int(fat_pct*100)
    }

# --- Basic greedy meal planner ---
def build_meal_plan(total_calories, goal="maintenance"):
    """
    total_calories: daily kcal target (int/float)
    goal: maintenance | weight_loss | muscle_gain
    Returns: dict with breakfast/lunch/dinner/snacks lists of {name, grams, kcal}
    """

    # distribution of calories across meals (simple heuristic)
    distribution = {
        "breakfast": 0.25,
        "lunch": 0.35,
        "dinner": 0.30,
        "snacks": 0.10
    }

    # For weight_loss reduce by small amount or keep calories same (we assume caller used goal)
    plan = {}
    rng = random.Random(42)  # deterministic for demo; remove seed if you want randomness

    # convert food db to list of dicts for easy sampling
    foods = food_df.to_dict(orient="records")

    # Define a lightweight strategy:
    # - For each meal, try to pick 2 items: one 'main' (protein/carb), one 'side' (veg/fruit/dairy/fat) to meet the calories.
    # - Compute needed kcal for the meal and pick best-fit items by caloric density (cal per g).
    for meal, frac in distribution.items():
        target_kcal = total_calories * frac
        # choose potential mains and sides based on category
        mains = [f for f in foods if f['category'] in ('protein', 'carb', 'protein/legume', 'dairy')]
        sides = [f for f in foods if f['category'] not in ('protein', 'carb', 'protein/legume', 'dairy')]
        # if snacks, allow small items (fruit, nuts)
        if meal == "snacks":
            mains = [f for f in foods if f['category'] in ('fruit','fat/seed','dairy','carb')]
            sides = [f for f in foods if f['category'] in ('fruit','veg','fat/seed','dairy')]

        # choose 1 main and 1-2 sides randomly but weighted toward caloric density for mains
        mains_sorted = sorted(mains, key=lambda x: x['cal_per_100g'], reverse=True)
        sides_sorted = sorted(sides, key=lambda x: x['cal_per_100g'], reverse=False)  # sides lower cal density

        # pick a top main and a random side among top 6 sides
        main = mains_sorted[0] if mains_sorted else foods[0]
        side_candidates = sides_sorted[:8] if len(sides_sorted) >= 8 else sides_sorted
        side = rng.choice(side_candidates) if side_candidates else main

        # now compute grams to reach approx target: allocate 70% kcal to main, 30% to side for balance
        main_kcal = target_kcal * 0.7
        side_kcal = target_kcal * 0.3

        # Prevent divide-by-zero if cal_per_100g==0
        def grams_for_kcal(food_item, kcal):
            c100 = max(food_item.get('cal_per_100g', 1), 1)
            grams = (kcal / c100) * 100.0
            # Round grams to a sensible size
            return max(10, int(round(grams / 5) * 5))  # round to nearest 5g, min 10g

        main_grams = grams_for_kcal(main, main_kcal)
        side_grams = grams_for_kcal(side, side_kcal)

        # compute actual kcal from rounded grams
        main_actual_kcal = round(main_grams * main['cal_per_100g'] / 100.0)
        side_actual_kcal = round(side_grams * side['cal_per_100g'] / 100.0)
        plan[meal] = [
            {
                "name": main['name'],
                "grams": main_grams,
                "kcal": int(main_actual_kcal),
                "category": main.get('category')
            },
            {
                "name": side['name'],
                "grams": side_grams,
                "kcal": int(side_actual_kcal),
                "category": side.get('category')
            }
        ]

    # compute totals
    total_planned = sum(item['kcal'] for meal_items in plan.values() for item in meal_items)
    # if difference, add small snack adjustment (e.g., extra 20-50 kcal) to snacks
    diff = int(total_calories - total_planned)
    if diff != 0:
        # adjust snacks: increase or decrease first snack item grams proportionally
        snack_items = plan.get('snacks', [])
        if snack_items:
            adjust_item = snack_items[0]
            # How many grams correspond to diff?
            cal_per_g = max(1.0, food_df.loc[food_df['name'] == adjust_item['name'], 'cal_per_100g'].values[0] / 100.0)
            adjust_grams = int(round(diff / cal_per_g)) if cal_per_g > 0 else 0
            new_grams = max(5, adjust_item['grams'] + int(round(adjust_grams)))
            adjust_item['grams'] = new_grams
            adjust_item['kcal'] = int(round(new_grams * cal_per_g))
            # recompute totals
            total_planned = sum(item['kcal'] for meal_items in plan.values() for item in meal_items)
            diff = int(total_calories - total_planned)

    result = {
        "total_target_kcal": int(total_calories),
        "total_planned_kcal": int(total_planned),
        "difference_kcal": int(total_calories - total_planned),
        "meals": plan
    }
    return result

# --- Existing routes: index, predict, chart.png (same as before) ---
@app.route("/")
def index():
    return send_from_directory("static", "index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data_json = request.get_json()
    try:
        age = float(data_json.get("age"))
        gender_raw = data_json.get("gender", "male")
        gender = 1 if str(gender_raw).lower().startswith("m") else 0
        height = float(data_json.get("height_cm"))
        weight = float(data_json.get("weight_kg"))
        activity_level = int(data_json.get("activity_level"))
        goal = data_json.get("goal", "maintenance")
    except Exception as e:
        return jsonify({"error": "Invalid input: " + str(e)}), 400

    x = np.array([[age, gender, height, weight, activity_level]])
    pred = float(round(model.predict(x)[0], 0))
    macros = macro_split(pred, goal)

    result = {
        "predicted_calories_kcal": pred,
        "macros": macros,
        "input": {
            "age": age,
            "gender": "male" if gender == 1 else "female",
            "height_cm": height,
            "weight_kg": weight,
            "activity_level": activity_level,
            "goal": goal
        }
    }
    return jsonify(result)

@app.route("/chart.png")
def chart_png():
    calories = request.args.get("calories", type=float)
    goal = request.args.get("goal", default="maintenance")
    if calories is None:
        return "Missing calories query param. Example: /chart.png?calories=2500", 400

    macros = macro_split(calories, goal)
    labels = ['Carbs (g)', 'Protein (g)', 'Fat (g)']
    values = [macros['carbs_g'], macros['protein_g'], macros['fat_g']]

    fig, ax = plt.subplots(figsize=(6,4))
    bars = ax.bar(labels, values)
    ax.set_title(f"Macro distribution for {int(calories)} kcal ({goal})")
    ax.set_ylabel("Grams")
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{int(height)}', xy=(bar.get_x()+bar.get_width()/2, height),
                    xytext=(0,3), textcoords="offset points", ha='center', va='bottom')

    buf = io.BytesIO()
    plt.tight_layout()
    fig.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    return send_file(buf, mimetype='image/png')

# --- NEW endpoint: meal plan ---
@app.route("/mealplan", methods=["GET"])
def mealplan():
    """
    GET /mealplan?calories=2500&goal=maintenance
    Returns JSON meal plan based on local food_db.csv
    """
    calories = request.args.get("calories", type=float)
    goal = request.args.get("goal", default="maintenance")
    if calories is None:
        return jsonify({"error": "Please provide calories as query param, e.g. /mealplan?calories=2500"}), 400

    plan = build_meal_plan(calories, goal)
    return jsonify(plan)

if __name__ == "__main__":
    app.run(debug=True, port=5000)