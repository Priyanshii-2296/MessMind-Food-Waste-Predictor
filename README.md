# MessMind – AI Food-Waste Predictor for College Mess
Machine Learning & Applied AI project (IBM SkillsBuild – AICTE–BharatCares internship 2026). Supports UN SDG 12.3 (halve food waste).

**Problem:** college messes cook a fixed quantity every meal, so large amounts of food are wasted when attendance drops (exams, events, hot days, unpopular menus).
**Solution:** a regression model predicts food waste (kg) for the next meal from day, meal, menu popularity, exam week, event/holiday, temperature and registered students. A Streamlit app turns the prediction into "cook X kg less" and a rupee saving.

**Data:** 1,095 meal records (365 days x 3 meals), *simulated* with realistic patterns. Replace `data/mess_data.csv` with real mess logs (same columns) and re-run `python train.py`.
**Models compared:** Linear Regression, Random Forest, Gradient Boosting (best by MAE).

## Run
```
pip install -r requirements.txt
python generate_data.py   # optional: regenerate simulated data
python train.py           # trains models, saves models/ and charts/
streamlit run app.py
```
