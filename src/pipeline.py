from __future__ import annotations

import argparse
import os

from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler
from pyspark.ml.regression import RandomForestRegressor
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from feature_engineering import build_zone_demand_summary, clean_and_engineer_features
from utils import build_month_paths, ensure_dir, load_config, save_json


def create_spark(app_name: str) -> SparkSession:
    return (
        SparkSession.builder.appName(app_name)
        .config("spark.sql.session.timeZone", "UTC")
        .getOrCreate()
    )


def build_ml_pipeline(categorical_cols: list[str], numeric_cols: list[str]) -> Pipeline:
    indexers = [
        StringIndexer(inputCol=col, outputCol=f"{col}_idx", handleInvalid="keep")
        for col in categorical_cols
    ]
    encoders = [
        OneHotEncoder(inputCol=f"{col}_idx", outputCol=f"{col}_ohe")
        for col in categorical_cols
    ]
    assembler = VectorAssembler(
        inputCols=[f"{col}_ohe" for col in categorical_cols] + numeric_cols,
        outputCol="features",
    )
    model = RandomForestRegressor(
        featuresCol="features",
        labelCol="trip_duration_minutes",
        predictionCol="prediction",
        numTrees=60,
        maxDepth=10,
        seed=42,
    )
    return Pipeline(stages=indexers + encoders + [assembler, model])


def evaluate_predictions(predictions):
    rmse = RegressionEvaluator(
        labelCol="trip_duration_minutes", predictionCol="prediction", metricName="rmse"
    ).evaluate(predictions)
    mae = RegressionEvaluator(
        labelCol="trip_duration_minutes", predictionCol="prediction", metricName="mae"
    ).evaluate(predictions)
    r2 = RegressionEvaluator(
        labelCol="trip_duration_minutes", predictionCol="prediction", metricName="r2"
    ).evaluate(predictions)
    return rmse, mae, r2


def main(config_path: str) -> None:
    config = load_config(config_path)
    output_dir = config["output_dir"]
    ensure_dir(output_dir)

    spark = create_spark(config["app_name"])

    paths = build_month_paths(config["months"], config["data_dir"])
    raw_df = spark.read.parquet(*paths)

    if config.get("sample_fraction"):
        raw_df = raw_df.sample(
            withReplacement=False,
            fraction=float(config["sample_fraction"]),
            seed=int(config["random_seed"]),
        )

    df = clean_and_engineer_features(
        raw_df,
        min_trip_minutes=int(config["min_trip_minutes"]),
        max_trip_minutes=int(config["max_trip_minutes"]),
        min_trip_distance=float(config["min_trip_distance"]),
        max_trip_distance=float(config["max_trip_distance"]),
    ).cache()

    train_df, test_df = df.randomSplit(
        [float(config["train_ratio"]), 1 - float(config["train_ratio"])],
        seed=int(config["random_seed"]),
    )

    categorical_cols = ["VendorID", "RatecodeID", "PULocationID", "DOLocationID", "payment_type"]
    numeric_cols = [
        "passenger_count",
        "trip_distance",
        "fare_amount",
        "total_amount",
        "pickup_hour",
        "pickup_dayofweek",
        "pickup_month",
        "is_weekend",
    ]

    pipeline = build_ml_pipeline(categorical_cols, numeric_cols)
    model = pipeline.fit(train_df)
    predictions = model.transform(test_df).cache()

    rmse, mae, r2 = evaluate_predictions(predictions)

    metrics = {
        "rmse": round(float(rmse), 4),
        "mae": round(float(mae), 4),
        "r2": round(float(r2), 4),
        "training_rows": int(train_df.count()),
        "test_rows": int(test_df.count()),
        "months": config["months"],
    }
    save_json(metrics, os.path.join(output_dir, "metrics.json"))

    demand_df = build_zone_demand_summary(df)
    demand_df.toPandas().to_csv(os.path.join(output_dir, "demand_by_pickup_zone.csv"), index=False)

    sample_predictions = (
        predictions.select(
            "tpep_pickup_datetime",
            "PULocationID",
            "DOLocationID",
            "trip_distance",
            "trip_duration_minutes",
            F.round("prediction", 2).alias("predicted_trip_duration_minutes"),
        )
        .limit(200)
        .toPandas()
    )
    sample_predictions.to_csv(os.path.join(output_dir, "sample_predictions.csv"), index=False)

    spark.stop()
    print("Pipeline completed successfully.")
    print(metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    args = parser.parse_args()
    main(args.config)
