# Predicting MBTA Rail On-Time Performance Using Distributed Machine Learning

> **EECE 5645 — Parallel Processing for Data Analytics | Northeastern University | Spring 2026**  
> Nirmalkumar Thirupallikrishnan Kesavan · Pradnyesh Choudhari · Raveendra Sanapala

---

## Table of Contents

- [Project Overview](#project-overview)
- [Architecture](#architecture)
- [Dataset](#dataset)
- [Layer A — Preprocessing](#layer-a--preprocessing)
- [Layer B — Feature Engineering & ML](#layer-b--feature-engineering--ml)
- [Model Results](#model-results)
- [Distribution Shift Analysis](#distribution-shift-analysis)
- [Lag Feature Mitigation](#lag-feature-mitigation)
- [Parallelism Benchmark](#parallelism-benchmark)
- [Parallelism Results](#parallelism-results)
- [Key Findings](#key-findings)
- [Tech Stack](#tech-stack)
- [Repository Structure](#repository-structure)
- [Team Contributions](#team-contributions)

---

## Project Overview

The MBTA operates **7 rail routes** (Red, Blue, Orange, Green-B/C/D/E) serving **~300,000 daily riders**. On-Time Performance (OTP) — the fraction of scheduled trips completed within published tolerance — is the agency's primary reliability metric.

**Two goals:**
1. Predict daily per-route OTP ahead of a service day using a distributed ML pipeline
2. Measure how much parallelism concretely reduces training time across single-node and multi-node clusters

**Scale:** 150M+ raw records across 5 heterogeneous sources, 2020–2024

---

## Architecture

### Medallion Architecture (Bronze → Silver → Gold)

```
┌──────────────────────────────────────────────────────────────────────────┐
│                       RAW DATA SOURCES (S3 Bronze)                       │
│                                                                          │
│   Reliability (1M rows)     Train Events (151M rows)                     │
│   Gated Entries (65M rows)  Service Alerts (3.8M rows)                   │
│   Weather / NOAA (2,269 rows)                                            │
└────────────────────────────────┬─────────────────────────────────────────┘
                                 │
                    s3a://mbta-reliability-project/raw
                                 │
                                 ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                        BRONZE → SILVER (Layer A)                         │
│                                                                          │
│  Gated Entries:                                                          │
│    Load GSE_2020–GSE_2024.csv, unionByName across years                  │
│    Map 131 station names → route (Red/Blue/Orange/Green-B/C/D/E)         │
│    Peak flag: 06:30–09:00 + 15:30–18:29 Eastern time                     │
│    Pivot ON_PEAK / OFF_PEAK → 2 demand columns per route-day             │
│                                                                          │
│  Service Alerts:                                                         │
│    multiLine=True fixes 3.8M → 34M inflation from embedded newlines      │
│    Filter: valid date regex, non-null route_id, severity 1–10            │
│    Compute 3 severity indices: Tiered, Quadratic, DBS                    │
│                                                                          │
│  Train Events:                                                           │
│    3 different schemas across 2020–2025                                  │
│    Dec 2021: extra _c0 column (pandas index leaked into CSV) — dropped   │
│    2024–2025: ISO timestamp format, converted to epoch seconds           │
│    Filter arrivals only (event_type == "ARR")                            │
│    Window function: headway per stop/direction/route/date                │
│    Filter: 30s < headway < 7200s                                         │
│    UTC → Eastern (America/New_York, DST-aware)                           │
│    Aggregate peak/off-peak mean + max headway per route per day          │
│                                                                          │
│  Reliability:                                                            │
│    Parse "2022/10/07 04:00:00+00" → date type                            │
│    Filter: Rail mode, otp_denominator > 0, 2020–2024                     │
│    otp_ratio = otp_numerator / otp_denominator                           │
│    Assign split: Rail test = Jan–May 2024                                │
│                                                                          │
│  Weather: NOAA GHCN Boston Logan — DATE, PRCP, SNOW, TAVG, TMIN, TMAX    │ 
│                                                                          │
│  Written as Parquet → s3a://mbta-reliability-project/processed           │
└────────────────────────────────┬─────────────────────────────────────────┘
                                 │
                                 ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                         GOLD LAYER — Master Join                         │
│                                                                          │
│  Base: df_reliability filtered to Rail mode (7 routes)                   │
│  Join key: (service_date, route_id)                                      │
│                                                                          │
│  Join 1: Weather       — date only (same weather all routes)             │
│  Join 2: Gated entries — date + gated_join_key                           │
│            Green branches → "Green Line", Red → "Red Line", etc.         │
│  Join 3: Events        — date + route_id (per-branch match)              │
│  Join 4: Alerts        — date + route_id → alert_line                    │
│                                                                          │
│  fillna(0): alerts (no alerts = no disruption)                           │
│  fillna(0): headways (no events data = no service)                       │
│                                                                          │
│  Result: 19,229 rows × 20 columns · 0 nulls                              │
└──────────────────────────────────────────────────────────────────────────┘
```

### Cluster Configuration

```
┌──────────────────────────────────────────────────────────────┐
│                SINGLE-NODE (AWS m5.xlarge)                   │
│                 4 vCPU · 16 GB RAM                           │
│                                                              │
│  CrossValidator(parallelism=N), N ∈ {1, 2, 4}                │
│  All concurrent jobs share the same 4 cores                  │
│  Intra-model parallelism competes with cross-CV parallelism  │
└──────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────┐
│           MULTI-NODE (Databricks m5d.2xlarge workers)        │
│                 8 vCPU · 32 GB RAM per worker                │
│                                                              │
│  CrossValidator(parallelism=N), N ∈ {1, 2, 4}                │
│  Each concurrent job lands on its own dedicated worker       │
│  No core contention between concurrent training jobs         │
│                                                              │
│  Example N=4:                                                │
│    Worker 1 → Job 1  (8 dedicated cores)                     │
│    Worker 2 → Job 2  (8 dedicated cores)                     │
│    Worker 3 → Job 3  (8 dedicated cores)                     │
│    Worker 4 → Job 4  (8 dedicated cores)                     │
└──────────────────────────────────────────────────────────────┘
```

---

## Dataset

| Source | Rows | Contribution |
|--------|------|-------------|
| Reliability (MBTA Open Data) | 1M | OTP numerator/denominator by route — **target variable** |
| Train Events (MBTA Open Data) | 151M | Arrival timestamps → headway statistics |
| Gated Entries (MBTA Open Data) | 65M | Fare gate tap counts → passenger demand |
| Service Alerts (MBTA Open Data) | 3.8M | Disruption severity scores (1–10) |
| Weather (NOAA GHCN — BOS Logan) | 2,269 | Daily climate measurements |

**Scope:** Rail mode only · Red, Blue, Orange, Green-B/C/D/E · 2020–2024

### Master Table Statistics

| Statistic | Value |
|-----------|-------|
| Total rows | 19,229 |
| Training rows | 17,469 (2020–2023) |
| Test rows | 1,760 (Jan–May 2024) |
| Columns | 20 |
| Null values | 0 (after fill) |
| OTP mean (std) | 0.849 (0.086) |
| OTP range per route | 0.775 (Green-D) → 0.951 (Blue) |

---

## Layer A — Preprocessing

### S3 Paths

```python
S3_BRONZE  = "s3a://mbta-reliability-project/raw"
S3_SILVER  = "s3a://mbta-reliability-project/processed"
S3_RESULTS = "s3a://mbta-reliability-project/results"

def read_csv(path, **kw):
    return spark.read.csv(path, header=True, inferSchema=True, quote='"', **kw)
```

### Gated Entries

```python
# Load all years and union
gse_2020 = read_csv(S3_GATED_ENTRIES_BRONZE + "/GSE_2020.csv")
gse_2021 = read_csv(S3_GATED_ENTRIES_BRONZE + "/GSE_2021.csv")
gse_2022 = read_csv(S3_GATED_ENTRIES_BRONZE + "/GSE_2022.csv")
gse_2023 = read_csv(S3_GATED_ENTRIES_BRONZE + "/GSE_2023.csv")
gse_2024 = read_csv(S3_GATED_ENTRIES_BRONZE + "/GSE_2024.csv")
gse_all = gse_2020.unionByName(gse_2021).unionByName(gse_2022) \
                  .unionByName(gse_2023).unionByName(gse_2024)

# Map station names to routes then classify peak/off-peak
gse_with_line = gse_with_line.withColumn(
    "ON_PEAK",
    F.when(
        (F.col("time_clean").between("06:30:00", "08:59:00")) |
        (F.col("time_clean").between("15:30:00", "18:29:00")),
        "ON_PEAK"
    ).otherwise("OFF_PEAK")
)

# Aggregate and pivot
gse_final = gse_with_line \
    .withColumn("gated_entries_num", F.col("gated_entries").cast("double")) \
    .groupBy("service_date", "route_or_line") \
    .pivot("ON_PEAK", ["ON_PEAK", "OFF_PEAK"]) \
    .agg(F.round(F.sum("gated_entries_num"), 2)) \
    .withColumnRenamed("ON_PEAK",  "gse_sum_on_peak") \
    .withColumnRenamed("OFF_PEAK", "gse_sum_off_peak")
```

### Service Alerts — Three Severity Indices

```python
# multiLine=True is the critical fix: without it 3.8M rows → 34M due to embedded \n
df_alerts_raw = spark.read.csv(
    f"{S3_BRONZE}/alerts/20*/*.csv",
    header=True, inferSchema=True,
    quote='"', escape='"', multiLine=True
)

df_alerts_daily = (
    df_alerts_valid
    .groupBy("alert_date", "alert_line")
    .agg(
        F.count("*").alias("alert_count"),
        F.sum("severity").alias("severity_sum"),

        # Index 1 — Tiered: low(1-4)×1 + med(5-7)×2 + high(8-10)×3
        (
            F.sum(F.when(F.col("severity") <= 4, 1).otherwise(0)) * 1 +
            F.sum(F.when((F.col("severity") >= 5) &
                         (F.col("severity") <= 7), 1).otherwise(0)) * 2 +
            F.sum(F.when(F.col("severity") >= 8, 1).otherwise(0)) * 3
        ).alias("alert_index_tiered"),

        # Index 2 — Quadratic: SUM(sev²) / 10  (penalises catastrophic events)
        F.round(F.sum(F.pow("severity", 2)) / 10, 2).alias("alert_index_squared")
    )
    # Index 3 — DBS: SUM(sev) × log(1 + count)  (rewards volume AND intensity)
    .withColumn("alert_index_dbs",
        F.round(F.col("severity_sum") * F.log1p(F.col("alert_count")), 2))
    .drop("alert_count", "severity_sum")
)
```

### Train Events — Multi-Year Schema Handling

```python
KEEP_COLS = ["service_date", "route_id", "direction_id",
             "stop_id", "event_type", "event_time"]

# 2021 December: extra _c0 column (pandas row index leaked into CSV)
df_dec_hr = read_csv(f"{S3}/events/2021/2021-12_HREvents.csv") \
    .drop("_c0").select(KEEP_COLS)

# 2024-2025: timestamp/ISO format — normalize to epoch seconds
def normalize_event_time(df, year):
    et_type = dict(df.dtypes)["event_time"]
    if et_type in ("string", "int", "long", "bigint"):
        df_test = df.withColumn("_et_test", F.col("event_time").cast("long"))
        null_count = df_test.filter(F.col("_et_test").isNull()).count()
        total = df_test.count()
        if null_count / total < 0.01:  # already epoch seconds
            return df.withColumn("event_time_epoch",
                                 F.col("event_time").cast("long"))
        else:  # ISO/timestamp string e.g. "2025-10-02T07:01:44Z"
            return df.withColumn("event_time_epoch",
                F.unix_timestamp(
                    F.trim(F.regexp_replace(
                        F.col("event_time").cast("string"), "[TZ]", " "))))
    elif et_type == "timestamp":
        return df.withColumn("event_time_epoch", F.unix_timestamp("event_time"))
```

### Headway Computation (Spark Window Function)

```python
# Window: per stop/direction/route, ordered by epoch time
w = Window.partitionBy(
    "service_date", "route_id", "stop_id", "direction_id"
).orderBy("event_time_epoch")

df_headways = (
    df_arrivals
    .withColumn("prev_time", F.lag("event_time_epoch").over(w))
    .withColumn("headway_sec", F.col("event_time_epoch") - F.col("prev_time"))
    .filter(F.col("headway_sec").isNotNull())
    .filter((F.col("headway_sec") > 30) & (F.col("headway_sec") < 7200))
    # > 30s  removes duplicate-logged events
    # < 7200s removes end-of-service gaps
)

# UTC → Eastern + peak classification
df_headways_et = (
    df_headways
    .withColumn("timestamp_utc", F.from_unixtime("event_time_epoch"))
    .withColumn("timestamp_et",
        F.from_utc_timestamp("timestamp_utc", "America/New_York"))
    .withColumn("time_decimal",
        F.hour("timestamp_et") + F.minute("timestamp_et") / 60.0)
    .withColumn("peak_offpeak",
        F.when(
            ((F.col("time_decimal") >= 6.5) & (F.col("time_decimal") < 9.0)) |
            ((F.col("time_decimal") >= 15.5) & (F.col("time_decimal") < 18.5)),
            "ON_PEAK"
        ).otherwise("OFF_PEAK"))
)

# Aggregate to daily per-route
df_events_final = (
    df_headways_et.repartition(16, "service_date", "route_id")
    .groupBy("service_date", "route_id")
    .agg(
        F.round(F.mean(F.when(F.col("peak_offpeak") == "ON_PEAK",
            F.col("headway_sec"))), 2).alias("headway_mean_peak"),
        F.round(F.mean(F.when(F.col("peak_offpeak") == "OFF_PEAK",
            F.col("headway_sec"))), 2).alias("headway_mean_offpeak"),
        F.round(F.mean("headway_sec"), 2).alias("headway_mean_total"),
        F.max("headway_sec").alias("headway_max"),
    )
)
```

### Reliability Cleaning

```python
df_reliability = (
    df_rel
    .withColumn("service_date",
        F.to_date(F.substring("service_date", 1, 10), "yyyy/MM/dd"))
    .filter(F.col("service_date").isNotNull())
    .filter(F.col("otp_denominator") > 0)
    .filter(F.col("service_date") >= "2020-01-01")
    .filter(F.col("service_date") <= "2025-11-30")
    .withColumn("otp_ratio",
        F.col("otp_numerator") / F.col("otp_denominator"))
    .withColumn("split",
        F.when(
            (F.col("mode_type") == "Rail") &
            (F.col("service_date") >= "2024-01-01"), "test"
        ).otherwise("train"))
    .withColumnRenamed("gtfs_route_id", "route_id")
    .select("service_date", "route_id", "mode_type", "peak_offpeak_ind",
            "otp_numerator", "otp_denominator", "otp_ratio", "split")
)
```

### Master Join

```python
# Green branches all map to "Green Line" in gated entries
df_base = (
    df_reliability
    .filter(F.col("mode_type") == "Rail")
    .withColumn("gated_join_key",
        F.when(F.col("route_id").startswith("Green"), "Green Line")
         .otherwise(F.concat(F.col("route_id"), F.lit(" Line"))))
)

df_master = (
    df_base
    .join(df_weather,
        df_base.service_date == df_weather.weather_date, "left")
    .join(df_gated,
        (df_base.service_date == df_gated.service_date) &
        (df_base.gated_join_key == df_gated.route_or_line), "left")
    .join(df_events,
        (df_base.service_date == df_events.service_date) &
        (df_base.route_id == df_events.route_id), "left")
    .join(df_alerts,
        (df_base.service_date == df_alerts.alert_date) &
        (df_base.route_id == df_alerts.alert_line), "left")
)

df_master = df_master.fillna(0, subset=[
    "alert_index_tiered", "alert_index_squared", "alert_index_dbs",
    "headway_mean_peak", "headway_mean_offpeak",
    "headway_mean_total", "headway_max"
])

df_master.write.parquet(f"{S3_SILVER}/master/", mode="overwrite")
```

---

## Layer B — Feature Engineering & ML

### Feature Set (28 dimensions)

```python
FEATURE_COLS = [
    # Weather (5)
    "TAVG", "TMIN", "TMAX", "PRCP", "SNOW",
    # Demand (2)
    "gse_sum_on_peak", "gse_sum_off_peak",
    # Headways (4)
    "headway_mean_peak", "headway_mean_offpeak",
    "headway_mean_total", "headway_max",
    # Alerts (3)
    "alert_index_tiered", "alert_index_squared", "alert_index_dbs",
    # Temporal (6) — extracted from service_date
    "day_of_week", "month", "day_of_year", "is_weekend", "year", "peak_binary",
    # Route one-hot vector (7 values inside one column)
    "route_vec",
]
```

### Temporal Features

```python
df_fe = (
    df_master
    .withColumn("day_of_week", F.dayofweek("service_date"))  # 1=Sun, 7=Sat
    .withColumn("month", F.month("service_date"))
    .withColumn("day_of_year", F.dayofyear("service_date"))
    .withColumn("is_weekend",
        F.when(F.col("day_of_week").isin(1, 7), 1).otherwise(0))
    .withColumn("year", F.year("service_date"))
    .withColumn("peak_binary",
        F.when(F.col("peak_offpeak_ind") == "PEAK", 1).otherwise(0))
)
```

### Route Encoding + Pipeline

```python
# StringIndexer: "Red" → 0.0, "Blue" → 1.0, etc.
route_indexer = StringIndexer(
    inputCol="route_id",
    outputCol="route_idx",
    handleInvalid="keep"  # unknown routes → special index, not crash
)

# OneHotEncoder: 0.0 → [1,0,0,0,0,0,0]
route_encoder = OneHotEncoder(
    inputCol="route_idx",
    outputCol="route_vec",
    dropLast=False  # keep all 7 columns
)

assembler = VectorAssembler(
    inputCols=FEATURE_COLS,
    outputCol="features_raw",
    handleInvalid="error",
)

# withMean=False preserves sparsity of route_vec
# withStd=True gives LR unit variance (required for fair regularisation)
scaler = StandardScaler(
    inputCol="features_raw",
    outputCol="features",
    withMean=False,
    withStd=True,
)

prep_pipeline = Pipeline(stages=[
    route_indexer,  # Stage 1: string → index   (learned from train)
    route_encoder,  # Stage 2: index → one-hot  (learned from train)
    assembler,      # Stage 3: cols → vector     (no learning)
    scaler,         # Stage 4: vector → scaled   (learned from train)
])

# CRITICAL: fit only on training data to prevent leakage
prep_model = prep_pipeline.fit(df_train_raw)
df_train = prep_model.transform(df_train_raw).select("features", "otp_ratio", "route_id")
df_test  = prep_model.transform(df_test_raw).select("features", "otp_ratio", "route_id")
df_train.cache()
df_test.cache()
```

### Four Models

```python
SEED = 42

# Model 1: Linear Regression
lr = LinearRegression(
    featuresCol="features", labelCol="otp_ratio",
    maxIter=100, regParam=0.01,
    elasticNetParam=0.0,      # pure L2/ridge
    standardization=False,    # pipeline already scaled
)

# Model 2: Decision Tree
dt = DecisionTreeRegressor(
    featuresCol="features", labelCol="otp_ratio",
    maxDepth=10, maxBins=32, seed=SEED,
)

# Model 3: Random Forest
rf = RandomForestRegressor(
    featuresCol="features", labelCol="otp_ratio",
    numTrees=100, maxDepth=10,
    featureSubsetStrategy="auto",  # 1/3 features per split
    subsamplingRate=1.0, seed=SEED,
)

# Model 4: Gradient Boosted Trees
gbt = GBTRegressor(
    featuresCol="features", labelCol="otp_ratio",
    maxIter=50,      # 50 sequential boosting rounds
    maxDepth=5,      # shallow trees (weak learners)
    stepSize=0.1,    # learning rate η
    subsamplingRate=1.0, seed=SEED,
)
```

### Train/Test Split

```
Time-Based Split (Primary — honest real-world forecasting):
  Train: 2020–2023  →  17,469 rows (90.8%)
  Test:  Jan–May 2024  →  1,760 rows (9.2%)
  df_train_raw = df_fe.filter(F.col("split") == "train")
  df_test_raw  = df_fe.filter(F.col("split") == "test")

Random 90/10 Split (IID control — eliminates drift):
  df_train_raw_rand, df_test_raw_rand = df_fe.randomSplit([0.9, 0.1], seed=42)
  Fresh pipeline fitted on random-split train (recomputes scaler stats)
```

---

## Model Results

### Time-Based Split (Honest Test)

| Model | Train Time | RMSE | MAE | R² |
|-------|-----------|------|-----|-----|
| **Linear Regression** | **5.8s** | **0.0628** | **0.0466** | **0.540** |
| Random Forest | 33.7s | 0.0678 | 0.0508 | 0.462 |
| GBT | 91.1s | 0.0683 | 0.0519 | 0.455 |
| Decision Tree | 3.6s | 0.0798 | 0.0561 | 0.255 |

> **Unexpected:** LR wins. Normal order is GBT > RF > DT > LR.

### Random 90/10 Split (IID Control)

| Model | Train Time | RMSE | MAE | R² |
|-------|-----------|------|-----|-----|
| **GBT** | **91.5s** | **0.0352** | **0.0248** | **0.833** |
| Random Forest | 28.1s | 0.0359 | 0.0251 | 0.826 |
| Decision Tree | 2.0s | 0.0395 | 0.0265 | 0.790 |
| Linear Regression | 0.7s | 0.0522 | 0.0369 | 0.632 |

> Normal ordering restored. Models are not broken — the data is the difference.

### R² Gap — Distribution Shift Cost

| Model | Time R² | Random R² | Drift cost |
|-------|---------|-----------|-----------|
| GBT | 0.455 | 0.833 | **−0.378** |
| Random Forest | 0.462 | 0.826 | **−0.364** |
| Decision Tree | 0.255 | 0.790 | **−0.535** |
| Linear Regression | 0.540 | 0.632 | **−0.092** ← barely affected |

---

## Distribution Shift Analysis

### What shifted in the 2024 test period

```python
# Feature drift audit — train vs test mean ratio
feats = ["TAVG", "TMIN", "TMAX", "PRCP", "SNOW",
         "gse_sum_on_peak", "gse_sum_off_peak",
         "headway_mean_peak", "headway_mean_offpeak",
         "alert_index_tiered", "alert_index_squared", "alert_index_dbs"]

for f in feats:
    train_mean = df_master.filter(F.col("split") == "train") \
        .agg(F.mean(f)).collect()[0][0]
    test_mean = df_master.filter(F.col("split") == "test") \
        .agg(F.mean(f)).collect()[0][0]
    ratio = test_mean / train_mean
    # ratio < 0.5 or > 2.0 → flagged as DRIFT
```

Key findings:
- Temperature (`TAVG`): test ~20% lower (only winter/spring months)
- Ridership (`gse_sum_on_peak`): test 11–17% higher (post-pandemic recovery)
- `alert_index_squared`: test 62% lower (no 2022 Orange Line shutdown equivalent)

### Why tree models fail, LR survives

```
Tree model (piecewise-constant):
  Learned: "if alert_index_squared > 200 → predict OTP = 0.81"
  2024 test: alert_index_squared = 50
  → Rule inactive → stale leaf prediction  ✗

Linear Regression (smooth global function):
  OTP = w1×TAVG + w2×headway_mean_peak + w3×alerts + ...
  2024 test: features shift → predictions shift proportionally
  No dead zones. Smooth extrapolation.  ✓
```

### Hybrid Architecture (failed)

```python
# ŷ(x) = f_LR(x) + α · f_GBT_residual(x),   α ∈ [0, 1.5]

ALPHA_SWEEP = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5]

for alpha in ALPHA_SWEEP:
    test_sweep = test_with_both.withColumn(
        "hybrid_prediction",
        F.col("lr_prediction") + alpha * F.col("gbt_residual_prediction")
    )
    rmse = RegressionEvaluator(
        labelCol="otp_ratio", predictionCol="hybrid_prediction", metricName="rmse"
    ).evaluate(test_sweep)
```

| α | RMSE | R² |
|---|------|-----|
| **0.0** | **0.0628** | **0.540** |
| 0.1 | 0.0631 | 0.535 |
| 0.3 | 0.0638 | 0.524 |
| 0.5 | 0.0647 | 0.510 |
| 1.0 | 0.0679 | 0.461 |
| 1.5 | 0.0720 | 0.394 |

Optimal α = 0.0 (pure LR). Every bit of GBT correction made things worse. Even a weakened residual GBT encodes 2020–2023 training-era patterns that do not transfer to 2024.

---

## Lag Feature Mitigation

### Why it works

Original features frozen at training-era boundaries:
```
"is temperature below 45°F?"         ← breaks when 2024 temperatures shift
"is alert_index_squared above 200?"  ← inactive in 2024 (index is 62% lower)
```

Lag features are self-calibrating:
```
"is this route's 30-day avg OTP above 0.85?"  ← threshold always relative to current state
```

### Implementation

```python
# STEP 1: Build per-route daily OTP series
df_daily = (df_master
    .groupBy("route_id", "service_date")
    .agg(F.mean("otp_ratio").alias("otp_daily"))
)

# STEP 2: Convert to days-since-epoch for rangeBetween
df_daily = df_daily.withColumn(
    "date_days", F.datediff("service_date", F.lit("1970-01-01"))
)

# STEP 3: Strictly backward-only rolling windows
# rangeBetween(-n_days, -1) = "up to ONE day before this row"
# The -1 is what prevents leakage — today's OTP never enters its own features

def lag_window(n_days):
    return (Window
            .partitionBy("route_id")
            .orderBy("date_days")
            .rangeBetween(-n_days, -1))  # STRICT past-only

df_lags = (df_daily
    .withColumn("otp_lag_7d",        F.avg("otp_daily").over(lag_window(7)))
    .withColumn("otp_lag_30d",       F.avg("otp_daily").over(lag_window(30)))
    .withColumn("otp_lag_90d",       F.avg("otp_daily").over(lag_window(90)))
    .withColumn("otp_lag_count_7d",  F.count("otp_daily").over(lag_window(7)))
    .drop("date_days")
)

# STEP 4: Cold-start fill (Jan 2020 rows have no history)
# Use training-period global mean — computed from train rows only, no leakage
train_mean_otp = (df_master
    .filter(F.col("split") == "train")
    .agg(F.mean("otp_ratio"))
    .collect()[0][0]
)
df_lags = df_lags.fillna(train_mean_otp, subset=["otp_lag_7d", "otp_lag_30d",
                                                   "otp_lag_90d"])
```

### Results after adding lag features

| Model | R² before | R² after | Gain | RMSE drop |
|-------|-----------|----------|------|-----------|
| Linear Regression | 0.540 | **0.686** | +0.146 | −16% |
| Decision Tree | 0.255 | 0.431 | +0.176 | −13% |
| Random Forest | 0.462 | 0.638 | +0.176 | −18% |
| GBT | 0.455 | 0.620 | +0.165 | −17% |

Lag features close ~50% of the drift gap in one intervention. LR still wins on the time-split (0.686 vs GBT 0.620) because residual drift in weather and alert features continues to favour smooth extrapolation.

---

## Parallelism Benchmark

### What is one "job"

One job = train one model, on one fold of data, with one hyperparameter value → produces one RMSE score. Jobs are completely independent of each other.

```
Stage 1: 2 param values × 3 folds = 6 jobs per CV run
Stage 2: 4 param values × 3 folds = 12 jobs per CV run

N ∈ {1, 2, 4}  (CrossValidator parallelism parameter)

Speedup    = T(N=1) / T(N=x)          ideal = x
Efficiency = Speedup / N               ideal = 1.0
```

### Wave Structure

```
Stage 1, N=4 (6 jobs):              Stage 2, N=4 (12 jobs):
┌─────┬─────┬─────┬─────┐          ┌─────┬─────┬─────┬─────┐
│ J1  │ J2  │ J3  │ J4  │ Wave 1   │ J1  │ J2  │ J3  │ J4  │ Wave 1
└─────┴─────┴─────┴─────┘          └─────┴─────┴─────┴─────┘
┌─────┬─────┬─────┬─────┐          ┌─────┬─────┬─────┬─────┐
│ J5  │ J6  │IDLE │IDLE │ Wave 2   │ J5  │ J6  │ J7  │ J8  │ Wave 2
└─────┴─────┴─────┴─────┘          └─────┴─────┴─────┴─────┘
  2 workers idle = wasted           ┌─────┬─────┬─────┬─────┐
  parallel time                     │ J9  │ J10 │ J11 │ J12 │ Wave 3
                                    └─────┴─────┴─────┴─────┘
                                      Zero idle workers ✓
```

### Benchmark Code

```python
# Stage 1: GRID_SIZE=2, Stage 2: GRID_SIZE=4
NUM_FOLDS = 3
GRID_SIZE = 2   # bump to 4 for Stage 2
PARALLELISM_LEVELS = [1, 2, 4]

def build_grid(estimator, name, size):
    if name == "LinearRegression":
        values = [0.001, 0.01, 0.1, 0.5][:size]
        return ParamGridBuilder().addGrid(estimator.regParam, values).build()
    elif name == "DecisionTree":
        values = [4, 6, 8, 12][:size]
        return ParamGridBuilder().addGrid(estimator.maxDepth, values).build()
    elif name == "RandomForest":
        values = [20, 40, 60, 100][:size]
        return ParamGridBuilder().addGrid(estimator.numTrees, values).build()
    elif name == "GBT":
        values = [15, 25, 40, 60][:size]
        return ParamGridBuilder().addGrid(estimator.maxIter, values).build()

for model_name, estimator in models_to_benchmark:
    param_grid = build_grid(estimator, model_name, GRID_SIZE)
    for par in PARALLELISM_LEVELS:
        cv = CrossValidator(
            estimator=estimator,
            estimatorParamMaps=param_grid,
            evaluator=eval_rmse,
            numFolds=NUM_FOLDS,
            parallelism=par,   # ← the variable under study
            seed=SEED,
        )
        t0 = time.time()
        cv_model = cv.fit(df_train)
        elapsed = time.time() - t0
        best_rmse = eval_rmse.evaluate(cv_model.bestModel.transform(df_test))
```

### Speedup Computation

```python
for dataset in ["time_split", "random_split"]:
    for model_name in model_names:
        t1 = subset[subset["parallelism"] == 1]["time_sec"].iloc[0]
        t2 = subset[subset["parallelism"] == 2]["time_sec"].iloc[0]
        t4 = subset[subset["parallelism"] == 4]["time_sec"].iloc[0]

        speedup_4    = round(t1 / t4, 2)        # ideal = 4.0
        efficiency_4 = round(speedup_4 / 4, 2)  # ideal = 1.0
```

---

## Parallelism Results

### Single-Node — Stage 1 (6 jobs, N=4, time-split)

| Model | Speedup | Efficiency |
|-------|---------|------------|
| Linear Regression | 1.36× | 0.34 |
| Decision Tree | 1.25× | 0.31 |
| Random Forest | 1.22× | 0.30 |
| GBT | 1.13× | 0.28 |

Why so low: each training job already uses all 4 cores internally. N=4 concurrent jobs split the same 4 cores 4 ways. Each job runs at quarter speed. Benefit and cost nearly cancel.

### Single-Node — Stage 2 (12 jobs, N=4, time-split)

| Model | Stage 1 | Stage 2 | Note |
|-------|---------|---------|------|
| Linear Regression | 1.36× | **1.51×** | Overhead fraction shrank |
| Random Forest | 1.22× | 1.32× | Overhead fraction shrank |
| Decision Tree | 1.25× | 1.19× | Noise — trains in 4s |
| GBT | 1.13× | 1.15× | Bottleneck is internal, not overhead |

### Multi-Node — Stage 1 (6 jobs, N=4, time-split)

| Model | Speedup | vs SN Stage 1 |
|-------|---------|--------------|
| Linear Regression | 1.65× | +0.29 |
| Random Forest | 1.53× | +0.31 |
| Decision Tree | 1.49× | +0.24 |
| GBT | 1.34× | +0.21 |

Core contention removed — each job has dedicated 8 cores. Still limited: wave 2 has only 2 jobs, leaving 2 workers idle (50% waste).

### Multi-Node — Stage 2 (12 jobs, N=4) — Best Configuration

| Model | Time-split | Random-split | Efficiency (random) |
|-------|-----------|-------------|---------------------|
| Random Forest | **2.26×** | 2.19× | 0.55 |
| Linear Regression | 2.12× | 1.83× | 0.46 |
| Decision Tree | 1.91× | 1.89× | 0.47 |
| GBT | 1.84× | **2.73× ← ALL-TIME PEAK** | **0.68** |

### Single-Node vs Multi-Node Stage 2 at N=4

| Model | Single-Node | Multi-Node | Gain |
|-------|------------|------------|------|
| Linear Regression | 1.51× | 2.12× | +0.61 |
| Decision Tree | 1.19× | 1.91× | +0.72 |
| Random Forest | 1.32× | 2.26× | +0.94 |
| **GBT** | **1.15×** | **2.73×** | **+1.58** |

### GBT: Worst → Best (the headline finding)

```
Single-node GBT had TWO simultaneous problems:

  Problem A — Sequential boosting rounds (algorithm constraint):
    Round N cannot start until Round N-1 finishes.
    F_m(x) = F_{m-1}(x) + η · h_m(x)
    h_m is trained on residuals of F_{m-1} — strict dependency.
    This is baked into GBT mathematics. No hardware removes it.

  Problem B — Core contention (hardware constraint):
    On a 4-vCPU machine, N=4 concurrent GBT jobs share 4 cores.
    Each job needs cores for data scans + tree building per round.
    Each job gets ~1 core instead of 4 → runs at quarter speed.
    Both A and B compound simultaneously → 1.13× speedup.

  Multi-node removes Problem B:
    Job 1 gets all 8 cores on Worker 1 to itself.
    Job 2 gets all 8 cores on Worker 2 to itself.
    Zero contention. Both run at full speed simultaneously.
    Problem A still limits each job internally.
    But A no longer interacts with B.
    → 2.73× speedup (best of entire benchmark)

  GBT was never a bad parallel model.
  It was a victim of the wrong hardware.
  The algorithm didn't change. Only the infrastructure changed.
```

### Why nobody reaches ideal 4×

```
1. Amdahl's Law — serial Spark driver overhead
   Job submission, broadcast variables, result collection,
   model serialisation, driver bookkeeping.
   All run on one thread regardless of worker count.
   If serial fraction s = 0.15 → max speedup = 1/0.15 = 6.7×.
   No worker count beats this ceiling.

2. Uneven workload — idle workers in last wave
   12 jobs ÷ N=4 = 3 exact waves (lucky here).
   Real grids with 6, 9, 15 jobs always have a partial final wave.
   Idle workers in last wave = wasted parallel slots.

3. Shuffle and network cost
   Each CV fold needs its data slice serialised and sent over
   the network to its worker before training can start.
   N=4 concurrent transfers share the same network bandwidth.
   Each transfer slightly slower → each job waits slightly longer.
```

### Correctness guarantee

> Best RMSE found by each model is **identical across all N values to 5 decimal places**.  
> Parallelism changes only *how fast* the answer is reached — never *what* the answer is.  
> This was verified explicitly for every model, both splits, both stages.

---

## Key Findings

**1. Distribution shift silently breaks tree models in real-world forecasting.**
Under honest time-based evaluation, LR beats GBT (R²=0.540 vs 0.455). The 2024 test window has systematically different temperature, ridership, and alert distributions. Tree model rules go inactive. LR extrapolates smoothly.

**2. Feature engineering beats model complexity for drift.**
Hybrid LR+GBT failed (optimal α=0). Adding 4 rolling lag features recovered R² gains of +0.146 to +0.176 across all models. Self-calibrating features that move with the current distribution beat architectural complexity.

**3. Dedicated compute per job is the parallelism unlock.**
Single-node peak: 1.51×. Multi-node peak: 2.73×. The lever was not N or grid size — it was eliminating core contention. GBT went from worst (+0.02 in Stage 1→2 on SN) to biggest beneficiary (+1.58 SN→MN).

**4. Practical deployment recipe.**
```
Lag features  +  Multi-node CV  +  Time-based holdout
     ↓                ↓                    ↓
  Fix drift      Train fast          Honest evaluation
     
→ Train all 4 families concurrently via CrossValidator
→ Pick winner on rolling time-based holdout
→ Retrain frequently to keep training distribution current
```

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Distributed processing | Apache Spark (PySpark) |
| ML framework | PySpark MLlib |
| Single-node cluster | AWS m5.xlarge (4 vCPU, 16 GB) |
| Multi-node cluster | Databricks m5d.2xlarge (8 vCPU, 32 GB per worker) |
| Storage | AWS S3 — Parquet (Medallion architecture) |
| Data sources | MBTA Open Data Portal, NOAA GHCN |
| Language | Python 3 |
| Visualisation | Matplotlib, Seaborn, Pandas |

---

## Repository Structure

```
mbta-otp-prediction/
│
├── README.md
│
├── Layer_A_Preprocessing/
│   └── Layer_A_Preprocessing.py    # Complete Bronze → Silver → Gold pipeline
│                                    # Gated entries (5-year union, peak flag)
│                                    # Service alerts (multiLine fix, 3 indices)
│                                    # Train events (3-schema handling,
│                                    #   event_time normalisation, headway window fn)
│                                    # Reliability (date parse, OTP ratio, split)
│                                    # Weather (NOAA GHCN)
│                                    # Master join (4 left joins, fillna)
│                                    # EDA + health checks
│
├── Layer_B_ML_and_Parallelism/
│   └── Machine_Learning_and_Parallelism.py
│                                    # Feature engineering (temporal + route OHE)
│                                    # Pipeline: StringIndexer + OHE +
│                                    #   VectorAssembler + StandardScaler
│                                    # 4 models: LR, DT, RF, GBT
│                                    # Time-based + random split evaluation
│                                    # Feature drift audit (train vs test)
│                                    # Hybrid LR + residual GBT (α-sweep 0→1.5)
│                                    # Lag features (7d/30d/90d, past-only)
│                                    # CrossValidator parallelism benchmark:
│                                    #   Stage 1 (GRID_SIZE=2, 6 jobs)
│                                    #   Stage 2 (GRID_SIZE=4, 12 jobs)
│                                    #   Both time-split and random-split
│                                    # Speedup + efficiency analysis
│                                    # Save models + metrics to S3
│
├── reports/
│   ├── EECE5645_Project_Report.pdf
│   └── EECE5645_Final_Presentation.pdf
│
└── results/                         # Written to S3 during execution
    ├── models/                      # Saved PySpark ML models (time-split)
    ├── models_random/               # Saved PySpark ML models (random-split)
    └── metrics/
        ├── time_split_base/
        ├── random_split_base/
        ├── time_vs_random_comparison/
        ├── hybrid_alpha_sweep/
        ├── parallelism_benchmark/
        └── speedup_analysis/
```

---

## Team Contributions

| Member | Contribution |
|--------|-------------|
| Nirmalkumar Thirupallikrishnan Kesavan | Data Preprocessing (Layer A) and ML Models & Parallel Processing (Layer B) |
| Pradnyesh Choudhari | Data Preprocessing (Layer A) |
| Raveendra Sanapala | ML Models & Parallelism (Layer B)|


---

*EECE 5645: Parallel Processing for Data Analytics — Northeastern University — Spring 2026*
