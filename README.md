# ATMOS AI — Professional Weather Prediction PBL App

Run:
1. `python -m venv .venv`
2. `.venv\Scripts\activate`
3. `pip install -r requirements.txt`
4. `python app.py`
5. Open `http://127.0.0.1:5000`

The application accepts Temperature, Humidity, Wind Speed and Rainfall, trains a Random Forest classifier, and displays the prediction and evaluation metrics.

If `weatheraus.csv` contains compatible data, it is used. Otherwise the built-in demo dataset keeps the PBL demonstration functional.
