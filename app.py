from flask import Flask, render_template, request
from utils.data_loader import load_careers, search_careers
import matplotlib
matplotlib.use('Agg')  # Prevents GUI issues
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import io
import base64
import os
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)

model, tfidf, le = joblib.load('model/career_predictor.pkl')

# Loading dataset once on startup
careers_data = load_careers("data/ai_career_compass_dataset.csv")

@app.route("/", methods=["GET", "POST"])
def index():
    query = request.form.get("query", "")
    filtered_careers = careers_data
    if query:
        filtered_careers = search_careers(careers_data, query)
    return render_template("index.html", careers=filtered_careers, query=query)

@app.route("/career/<role>")
def career_detail(role):
    career = next((c for c in careers_data if c['role'].lower() == role.lower()), None)
    if not career:
        return "Career Not Found", 404
    return render_template("career_detail.html", career=career)

@app.route('/saved')
def saved():
    print("✅ /saved route was hit!")
    return render_template('saved.html')

@app.route('/search')
def search():
    query = request.args.get('query', '')
    domain = request.args.get('domain', '').lower()
    skill = request.args.get('skill', '').lower()

    results = search_careers(careers, query)

    # Applying domain filter
    if domain:
        results = [c for c in results if domain in c['domain_industries'].lower()]
    
    # Applying skill filter
    if skill:
        results = [c for c in results if skill in c['required_skills'].lower()]

    return render_template("results.html", results=results, query=query)

@app.route('/all')
def all_careers():
    return render_template("results.html", results=careers, query="All Careers")

@app.route("/visualizations")
def visualizations():
    df = load_careers("data/ai_career_compass_dataset.csv")
    df = pd.DataFrame(df)
    
    # Setup for saving images
    viz_dir = os.path.join("static", "visuals")
    os.makedirs(viz_dir, exist_ok=True)

    plots = []

    # Convert career_growth_score to numeric
    if 'career_growth_score' in df.columns:
        df['career_growth_score'] = pd.to_numeric(df['career_growth_score'], errors='coerce')

    # Convert salary range to numeric midpoint safely
    if 'average_salary_range' in df.columns:
        # Remove $ and commas
        df['average_salary_range_clean'] = df['average_salary_range'].str.replace(r'[\$,]', '', regex=True)

        # Extract using regex and fill missing with NaN
        salary_extract = df['average_salary_range_clean'].str.extract(r'(?P<low>\d+)\s*-\s*(?P<high>\d+)')
        df['salary_low'] = pd.to_numeric(salary_extract['low'], errors='coerce')
        df['salary_high'] = pd.to_numeric(salary_extract['high'], errors='coerce')

        # Calculate midpoint only for valid numeric rows
        df['salary_midpoint'] = df[['salary_low', 'salary_high']].mean(axis=1)

    # 1. Line Chart – Career Growth Score by Role
    plt.figure(figsize=(12, 6))
    df_sorted = df.sort_values(by="career_growth_score")
    plt.plot(df_sorted["role"], df_sorted["career_growth_score"], marker='o')
    plt.xticks(rotation=90)
    plt.title("Career Growth Score by Role")
    plt.xlabel("AI Career Roles")
    plt.ylabel("Career Growth Score")
    path = os.path.join(viz_dir, "line_chart.png")
    plt.tight_layout()
    plt.savefig(path)
    plots.append(("Line Chart – Career Growth Score by Role", "visuals/line_chart.png"))
    plt.close()

    # 2. Bar Chart – Job Demand Level
    plt.figure(figsize=(8, 5))
    df["job_demand_level"].value_counts().plot(kind="bar", color="skyblue")
    plt.title("Job Demand Levels Distribution")
    plt.xlabel("Job Demand Level")
    plt.ylabel("Number of Roles")
    path = os.path.join(viz_dir, "bar_chart.png")
    plt.tight_layout()
    plt.savefig(path)
    plots.append(("Bar Chart – Job Demand Levels", "visuals/bar_chart.png"))
    plt.close()

    # 3. Histogram – Salary Midpoint
    plt.figure(figsize=(8, 5))
    df["salary_midpoint"].dropna().plot(kind="hist", bins=10, color="purple", edgecolor='black')
    plt.title("Distribution of Average Salary Midpoints")
    plt.xlabel("Salary (approximate midpoint)")
    plt.ylabel("Number of Roles")
    path = os.path.join(viz_dir, "histogram.png")
    plt.tight_layout()
    plt.savefig(path)
    plots.append(("Histogram – Salary Midpoints", "visuals/histogram.png"))
    plt.close()

    # 4. Box Plot – Career Growth Score
    plt.figure(figsize=(6, 5))
    sns.boxplot(y=df["career_growth_score"])
    plt.title("Boxplot of Career Growth Scores")
    plt.ylabel("Career Growth Score")
    path = os.path.join(viz_dir, "box_plot.png")
    plt.tight_layout()
    plt.savefig(path)
    plots.append(("Box Plot – Career Growth Score", "visuals/box_plot.png"))
    plt.close()

    # 5. Correlation Matrix + Heatmap
    plt.figure(figsize=(8, 6))
    numeric_df = df[["career_growth_score", "salary_midpoint"]].dropna()
    corr_matrix = numeric_df.corr()
    sns.heatmap(corr_matrix, annot=True, cmap="YlGnBu", fmt=".2f")
    plt.title("Correlation Matrix (Career Growth vs Salary)")
    path = os.path.join(viz_dir, "correlation_heatmap.png")
    plt.tight_layout()
    plt.savefig(path)
    plots.append(("Correlation Heatmap – Career Growth vs Salary", "visuals/correlation_heatmap.png"))
    plt.close()

    return render_template("visualizations.html", plots=plots)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    predicted_role = None
    recommended_career = None

    if request.method == 'POST':
        skills_input = request.form.get('skills_input')
        if skills_input:
            combined_input = skills_input
            X_input = tfidf.transform([combined_input])
            y_pred = model.predict(X_input)
            predicted_role = le.inverse_transform(y_pred)[0]

            # Match predicted role to data for details
            for c in careers_data:
                if c['role'].lower() == predicted_role.lower():
                    recommended_career = c
                    break

    return render_template('predict.html',
                           predicted_role=predicted_role,
                           recommended_career=recommended_career)



if __name__ == "__main__":
    app.run(debug=True)