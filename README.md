# AI Career Compass

An intelligent web application that predicts the most suitable career path based on your skills using machine learning and NLP.

---

## Overview

**AI Career Compass** is designed to assist users—especially students and professionals—in identifying potential career paths aligned with their skill sets. It uses a trained machine learning model (Random Forest Classifier) on a curated dataset to make intelligent recommendations, showing job demand, salary potential, and growth outlook.

---

## Features

- Skill-based career prediction
- Career insights (average salary, demand, and growth score)
- Machine Learning model trained with TF-IDF & Random Forest
- Visual representation of popular careers
- Simple and intuitive Flask web interface

---

## Tech Stack

| Layer          | Tools & Libraries Used                         |
|----------------|------------------------------------------------|
| **Frontend**   | HTML, CSS, Bootstrap, Jinja                    |
| **Backend**    | Python, Flask                                  |
| **ML/NLP**     | scikit-learn, pandas, numpy, nltk, joblib      |
| **Visualization** | matplotlib, seaborn                        |
| **Deployment** | GitHub + Git LFS                               |

---

## Algorithm & Deployment

We used the **Random Forest Classifier**, a robust ensemble learning algorithm known for its accuracy and ability to handle feature importance across large datasets. It works well with our TF-IDF transformed feature matrix of user skills.

Input is taken as raw text (skills) and processed using **NLP techniques**. The cleaned data is vectorized using **TF-IDF**, trained on labeled career data, and the model is saved using **Joblib**. The model then predicts a matching career path and displays additional insights.

---

## Project Structure 
```bash
ai_career_compass/ 
    career_clean/
        data/
            ai_career_compass_dataset.csv/
        model/
            career_predictor.pkl/
        css/
            styles.css/
        visualizations/
            bar_chart.png/
            box_plot.png/
            correlation_heatmap.png/
            histogram.png/
            line_chart.png/
        templates/
            base.html/
            career_detail.html/
            index.html/
            predict.html/
            result.html/
            saved.html/
            visualizations.html/
        utils/
            data_loader.py/
        app.py/
        career_predictor_training.py/
        requirements.txt/

```
---
## 🛠️ How to Run Locally

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/your-username/ai-career-compass.git
   cd ai-career-compass

2. **Run the Flask App:**
    ```bash
    python app.py

3. **Visit in Browser:**
http://127.0.0.1:5000

---

## Full Documentation

You can read the full project documentation [here](https://drive.google.com/file/d/17tZtAIrdL9WVoi8D_km4BYJMXdKDJxqm/view?usp=sharing).

---

## Author

**Vanshika M.**  
B.Tech CSE (Data Science), 3rd Year  
Malla Reddy Engineering College for Women

---

## Acknowledgment

This project was developed as part of AICTE's **Microsoft - AI Azure Virtual Internship**, supported by **Edunet Foundation** and **Microsoft**. All implementation, design, and development work was done independently by the author.

---

## License

This project is currently not licensed for public or open-source use. This project is intended for review and educational purposes only. Running the application locally for evaluation is permitted. Unauthorized commercial use, redistribution, or modification is not allowed without written permission. *All rights reserved by author*

---

## Contributions

Contributions are welcome but currently limited. If you'd like to suggest a feature or fix, please raise an issue or reach out to the maintainer for guidance before submitting a pull request.
