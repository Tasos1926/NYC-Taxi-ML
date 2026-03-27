from __future__ import annotations

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

REQUIRED_COLUMNS = [
    "VendorID",
    "tpep_pickup_datetime",
    "tpep_dropoff_datetime",
    "passenger_count",
    "trip_distance",
    "RatecodeID",
    "PULocationID",
    "DOLocationID",
    "payment_type",
    "fare_amount",
    "total_amount",
]


def validate_columns(df: DataFrame) -> None:
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def clean_and_engineer_features(
    df: DataFrame,
    min_trip_minutes: int,
    max_trip_minutes: int,
    min_trip_distance: float,
    max_trip_distance: float,
) -> DataFrame:
    validate_columns(df)

    cleaned = (
        df.select(*REQUIRED_COLUMNS)
        .filter(F.col("tpep_pickup_datetime").isNotNull())
        .filter(F.col("tpep_dropoff_datetime").isNotNull())
        .withColumn(
            "trip_duration_minutes",
    (
        F.unix_timestamp("tpep_dropoff_datetime")
        - F.unix_timestamp("tpep_pickup_datetime")
    ) / 60.0,
)
        .filter(F.col("trip_duration_minutes") >= min_trip_minutes)
        .filter(F.col("trip_duration_minutes") <= max_trip_minutes)
        .filter(F.col("trip_distance") >= min_trip_distance)
        .filter(F.col("trip_distance") <= max_trip_distance)
        .filter(F.col("fare_amount") >= 0)
        .filter(F.col("total_amount") >= 0)
        .withColumn("pickup_hour", F.hour("tpep_pickup_datetime"))
        .withColumn("pickup_dayofweek", F.dayofweek("tpep_pickup_datetime"))
        .withColumn("pickup_month", F.month("tpep_pickup_datetime"))
        .withColumn("is_weekend", F.when(F.col("pickup_dayofweek").isin([1, 7]), 1).otherwise(0))
        .na.fill(
            {
                "passenger_count": 0,
                "RatecodeID": 0,
                "PULocationID": 0,
                "DOLocationID": 0,
                "payment_type": 0,
                "VendorID": 0,
            }
        )
    )

    return cleaned


def build_zone_demand_summary(df: DataFrame) -> DataFrame:
    return (
        df.groupBy("PULocationID")
        .agg(
            F.count("*").alias("trip_count"),
            F.avg("trip_duration_minutes").alias("avg_trip_duration_minutes"),
            F.avg("trip_distance").alias("avg_trip_distance"),
            F.avg("total_amount").alias("avg_total_amount"),
        )
        .orderBy(F.desc("trip_count"))
    )
