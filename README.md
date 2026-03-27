# NYC Taxi Analytics with PySpark

A portfolio-ready end-to-end data project built with **PySpark** and **Spark MLlib** using the **NYC TLC Yellow Taxi** dataset.

This project shows how to:
- ingest large monthly taxi trip files,
- clean and validate raw trip data,
- engineer useful time and trip features,
- train a machine learning model in PySpark,
- generate evaluation metrics and business-friendly outputs.

It is designed for a **Data Analyst / Analytics Engineer / Junior Data Engineer** portfolio and can be published directly on GitHub.

---

## Project Goal

The project predicts **trip duration in minutes** from raw taxi trip records using PySpark.

This is useful because trip duration is closely tied to:
- operational planning,
- driver utilization,
- customer waiting expectations,
- pricing and demand analysis.

In addition to the model, the pipeline also creates a small **zone-level demand summary** that can be used for reporting or dashboarding.

---

## Tech Stack

- **Python 3.10+**
- **PySpark**
- **Spark MLlib**
- **Pandas** (lightweight output handling)
- **Parquet** input from NYC TLC

---

## Dataset

The code is built for the public NYC TLC Yellow Taxi parquet files.

Example source pattern:

`https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_YYYY-MM.parquet`

Example months for testing:
- `2024-01`
- `2024-02`
- `2024-03`

---

## Project Structure

```text
nyc_taxi_pyspark_portfolio_project/
├── README.md
├── requirements.txt
├── .gitignore
├── config/
│   └── settings.yaml
├── src/
│   ├── pipeline.py
│   ├── feature_engineering.py
│   └── utils.py
└── output/
    ├── metrics.json
    ├── demand_by_pickup_zone.csv
    └── sample_predictions.csv
```

---

## Business Questions Answered

This project is built around practical questions such as:

1. Can we estimate taxi trip duration based on trip characteristics?
2. Which pickup zones generate the most demand?
3. How can raw trip data be transformed into repeatable reporting outputs?
4. How can a scalable Spark workflow support both analytics and machine learning?

---

## Features Used for the ML Model

The model uses a mix of numeric and categorical features, including:

- passenger count
- trip distance
- pickup hour
- pickup day of week
- pickup month
- pickup zone ID
- dropoff zone ID
- rate code
- vendor ID
- payment type

Target variable:
- **trip duration in minutes**

---

## Data Cleaning Rules

The pipeline removes clearly invalid records, for example:

- missing timestamps
- negative or zero trip durations
- extremely short or extremely long trips
- negative fares
- invalid trip distances

This makes the workflow more realistic and portfolio-worthy.

---

## ML Approach

Current model:
- **RandomForestRegressor** from Spark MLlib

Evaluation metrics:
- **RMSE**
- **MAE**
- **R²**

You can easily swap the model for:
- `GBTRegressor`
- `LinearRegression`
- `DecisionTreeRegressor`

---

## How to Run

### 1. Create environment

```bash
python -m venv .venv
source .venv/bin/activate   # Linux / macOS
# .venv\Scripts\activate    # Windows
pip install -r requirements.txt
```

### 2. Edit config

Adjust the months in `config/settings.yaml` if needed.

### 3. Run pipeline

```bash
python src/pipeline.py --config config/settings.yaml
```

---

## Example Outputs

After running the project, the following files are created in `output/`:

### `metrics.json`
Contains model evaluation metrics such as:

```json
{
  "rmse": 8.42,
  "mae": 5.11,
  "r2": 0.71,
  "training_rows": 1250000,
  "test_rows": 312000
}
```

### `demand_by_pickup_zone.csv`
A simple aggregated file showing trip counts and average trip duration by pickup zone.

### `sample_predictions.csv`
A small sample of predictions for quick review or dashboard integration.

---

## Why This Project Works Well in a Portfolio

This project demonstrates more than just dashboarding:

- scalable processing with Spark,
- structured data cleaning,
- feature engineering,
- machine learning in a distributed environment,
- export of business-friendly reporting artifacts,
- reproducible project structure.

It positions you beyond basic reporting and shows backend-oriented analytics capability.

---

## Ideas for Next Improvements

You can extend this project further by adding:

- hyperparameter tuning with `CrossValidator`
- MLflow experiment tracking
- Docker support
- Airflow orchestration
- Power BI dashboard on top of generated CSV outputs
- zone lookup joins for human-readable location names
- fare prediction and demand forecasting

---

## Suggested GitHub Description

**PySpark portfolio project using NYC Taxi data for trip-duration prediction, demand analysis, and automated reporting outputs.**

---

## Suggested Resume Bullet

Built an end-to-end PySpark analytics project using NYC Taxi data to clean and transform large-scale trip records, engineer features, train a machine learning model for trip-duration prediction, and generate automated business reporting outputs.
