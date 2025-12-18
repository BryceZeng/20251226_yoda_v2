"""
Simplified feature engineering using direct Spark SQL
Focus only on essential simple features!
"""


def create_simple_features_sql(input_table, start_date=None, end_date=None):
    """
    Generate simple aggregated features using SQL - much cleaner and easier to maintain!

    Args:
        input_table: Name of the input table
        start_date: Optional start date filter
        end_date: Optional end date filter

    Returns:
        SQL query string for simple features
    """

    date_filter = ""
    if start_date:
        date_filter += f" AND tpep_pickup_datetime >= '{start_date}'"
    if end_date:
        date_filter += f" AND tpep_pickup_datetime < '{end_date}'"

    sql = f"""
    WITH hourly_aggregated AS (
        SELECT
            pickup_zip,
            dropoff_zip,
            date_trunc('hour', tpep_pickup_datetime) as rounded_datetime,

            -- Simple aggregations - easy to understand and maintain!
            AVG(fare_amount) as mean_fare_amount,
            COUNT(*) as trip_count,
            AVG(trip_distance) as mean_trip_distance,
            MAX(fare_amount) as max_fare_amount,
            MIN(fare_amount) as min_fare_amount

        FROM {input_table}
        WHERE pickup_zip IS NOT NULL
          AND dropoff_zip IS NOT NULL
          AND fare_amount > 0
          {date_filter}
        GROUP BY
            pickup_zip,
            dropoff_zip,
            date_trunc('hour', tpep_pickup_datetime)
    )
    SELECT
        pickup_zip as zip,
        dropoff_zip,
        rounded_datetime,
        date_format(rounded_datetime, 'yyyy-MM') as yyyy_mm,

        -- Aggregated features
        mean_fare_amount,
        trip_count,
        mean_trip_distance,
        max_fare_amount,
        min_fare_amount,

        -- Time-based features derived from rounded_datetime
        CASE
            WHEN dayofweek(rounded_datetime) IN (1, 7) THEN 1
            ELSE 0
        END as is_weekend,

        HOUR(rounded_datetime) as hour_of_day

    FROM hourly_aggregated
    """

    return sql


def compute_features_fn(input_df, timestamp_column, start_date, end_date):
    """
    Simplified feature computation function - only simple features now!

    This function replaces the complex pickup/dropoff feature approach with
    a single, simple feature computation that's easier to maintain.
    """

    # Create temporary table name
    input_table = f"temp_input_data_{int(time.time())}"

    # Register DataFrame as temporary table
    input_df.createOrReplaceTempView(input_table)

    # Get Spark session
    spark = input_df.sparkSession

    # Generate simple features
    simple_sql = create_simple_features_sql(input_table, start_date, end_date)
    features_df = spark.sql(simple_sql)

    return features_df


def generate_simple_features(
    spark, input_table, output_table, start_date=None, end_date=None
):
    """
    Generate simple features and save to table - one unified function!

    Usage example:
        generate_simple_features(
            spark,
            "pru.pac_mlops.raw_taxi_data",
            "pru.pac_mlops.trip_simple_features",
            start_date="2024-01-01",
            end_date="2024-02-01"
        )
    """

    print("🔄 Generating simple features using SQL...")
    features_sql = create_simple_features_sql(input_table, start_date, end_date)
    features_df = spark.sql(features_sql)

    # Write features to table
    print(f"💾 Writing simple features to {output_table}")
    features_df.write.mode("overwrite").saveAsTable(output_table)

    print("✅ Simple feature generation complete!")
    print(f"   📊 Features: {features_df.count()} rows")

    return features_df


# =============================================================================
# USAGE EXAMPLES - Much simpler approach!
# =============================================================================

# Example 1: Generate features for a specific date range
"""
generate_simple_features(
    spark,
    "pru.pac_mlops.raw_taxi_data",
    "pru.pac_mlops.trip_simple_features",
    start_date="2024-01-01",
    end_date="2024-02-01"
)
"""

# Example 2: Just get features for a specific date
"""
features_sql = create_simple_features_sql(
    "pru.pac_mlops.raw_taxi_data",
    start_date="2024-01-15"
)
features = spark.sql(features_sql)
features.display()
"""

# Add time import for temporary table naming
import time
