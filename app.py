from flask import Flask, render_template, request
from utils.data_loader import load_careers, search_careers

app = Flask(__name__)

# Load dataset once on startup
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

    # Apply domain filter
    if domain:
        results = [c for c in results if domain in c['domain_industries'].lower()]
    
    # Apply skill filter
    if skill:
        results = [c for c in results if skill in c['required_skills'].lower()]

    return render_template("results.html", results=results, query=query)

@app.route('/all')
def all_careers():
    return render_template("results.html", results=careers, query="All Careers")

if __name__ == "__main__":
    app.run(debug=True)