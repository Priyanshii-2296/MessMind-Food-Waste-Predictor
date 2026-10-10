"""Generates a simulated college-mess dataset (replace data/mess_data.csv with real mess logs if available)."""
import numpy as np, pandas as pd
rng = np.random.default_rng(42)
dates = pd.date_range("2025-01-01", periods=365)
rows = []
for d in dates:
    exam = int(d.month in (4, 5, 11, 12) and d.day >= 10 and d.day <= 28)
    event = int(rng.random() < 0.07 or d.dayofweek == 6 and rng.random() < 0.25)
    for meal in (0, 1, 2):                       # 0 breakfast, 1 lunch, 2 dinner
        reg = int(rng.integers(430, 520))
        pop = int(rng.integers(1, 6))            # menu popularity 1-5
        temp = float(np.clip(rng.normal(31 - 6 * (d.month in (12, 1, 2)) + 4 * (d.month in (4, 5, 6)), 3), 14, 44))
        att = 0.62 + 0.07 * pop - 0.06 * exam - 0.12 * event - 0.004 * max(temp - 34, 0) * (meal == 1)
        att += 0.05 * (d.dayofweek >= 5 and meal == 0) * -1 + 0.04 * (meal == 2)
        att = float(np.clip(att + rng.normal(0, 0.03), 0.35, 0.97))
        cooked = reg * 0.42 * (1.0 + rng.normal(0.08, 0.02))   # kg cooked (fixed-recipe habit)
        eaten = reg * att * 0.42
        waste = max(cooked - eaten + rng.normal(0, 1.2), 0)
        rows.append([d.date(), d.dayofweek, meal, pop, exam, event, round(temp, 1), reg, round(waste, 2)])
df = pd.DataFrame(rows, columns=["date", "day_of_week", "meal", "menu_popularity", "is_exam_week",
                                 "is_event_or_holiday", "temperature_c", "students_registered", "food_waste_kg"])
df.to_csv("data/mess_data.csv", index=False)
print(df.shape, df.food_waste_kg.describe().round(2).to_dict())
