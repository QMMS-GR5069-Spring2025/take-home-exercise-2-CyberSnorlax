# Databricks notebook source
from pyspark.sql import functions as F
from pyspark.sql.functions import col, year, current_date, avg, rank, sum, min, max, when, lit, first
from pyspark.sql.window import Window


# COMMAND ----------

#creating a pipeline
df_laptimes = spark.read.csv('s3://columbia-gr5069-main/raw/lap_times.csv', header=True)
df_drivers = spark.read.csv('s3://columbia-gr5069-main/raw/drivers.csv', header=True)
df_pitstops = spark.read.csv('s3://columbia-gr5069-main/raw/pit_stops.csv', header=True)
df_results = spark.read.csv('s3://columbia-gr5069-main/raw/results.csv', header=True)
df_races = spark.read.csv('s3://columbia-gr5069-main/raw/races.csv', header=True)

# exploratory data analysis
display(df_laptimes)

display(df_drivers)

# COMMAND ----------

df_drivers.count()

# COMMAND ----------

# Create age column for drivers
df_drivers = df_drivers.withColumn('age', year(current_date()) - year(col('dob')))

# COMMAND ----------

df_lap_drivers = df_drivers.select('driverId','driverRef','forename','age').join(df_laptimes, df_drivers.driverId == df_laptimes.driverId)
df_lapdrivers = df_lap_drivers.groupBy('age').agg(avg('milliseconds'))

display(df_lapdrivers)

# COMMAND ----------

# Average time each driver spent at pit stop for each race
df_pitstop_avg = df_pitstops.groupBy('driverId', 'raceId') \
    .agg(avg('milliseconds').alias('avg_pitstop_time')) \
    .join(df_drivers.select('driverId', 'forename', 'surname'), 'driverId')
    
display(df_pitstop_avg)

# COMMAND ----------

# Rank the average time spent at the pit stop in order of race winners
windowSpec = Window.partitionBy('raceId').orderBy('avg_pitstop_time')
df_pitstop_ranked = df_pitstop_avg.join(df_results.select('raceId', 'driverId', 'position'), 
                                       ['raceId', 'driverId']) \
    .filter(col('position') == 1) \
    .withColumn('pitstop_rank', rank().over(windowSpec)) \
    .select('raceId', 'driverId', 'forename', 'surname', 'avg_pitstop_time', 'pitstop_rank')
    
display(df_pitstop_ranked)

# COMMAND ----------

# Insert the missing code (e.g: ALO for Alonso) for drivers based on the 'drivers' dataset
df_drivers_with_code = df_drivers.withColumn('driverCode', 
    when(col('surname') == 'Alonso', 'ALO')
    .when(col('surname') == 'Hamilton', 'HAM')
    .when(col('surname') == 'Verstappen', 'VER')
    # Add more mappings as needed based on your drivers dataset
    .otherwise(lit('UNKNOWN')))
df_lap_drivers = df_drivers_with_code.select('driverId', 'driverRef', 'forename', 'driverCode', 'age') \
    .join(df_laptimes, 'driverId')
    
display(df_lap_drivers)

# COMMAND ----------

# Youngest and oldest driver for each race, create column "age"
# Create DataFrame with age calculated for each race
df_with_age = df_results.join(df_races.select('raceId', 'date'), 'raceId') \
    .join(df_drivers.select('driverId', 'dob', 'forename', 'surname'), 'driverId') \
    .withColumn('age', (datediff(col('date'), col('dob')) / 365.25).cast('double'))

# Define window partitioned by raceId and ordered by age
window_spec = Window.partitionBy('raceId').orderBy(col('age'))

# Add rank for youngest and reverse rank for oldest
df_ranked = df_with_age.withColumn('youngest_rank', rank().over(window_spec)) \
    .withColumn('oldest_rank', rank().over(window_spec.orderBy(col('age').desc())))

# Select youngest (rank 1) and oldest (rank 1) drivers per race
df_youngest_oldest = df_ranked.filter((col('youngest_rank') == 1) | (col('oldest_rank') == 1)) \
    .groupBy('raceId') \
    .agg(
        first(when(col('youngest_rank') == 1, col('driverId'))).alias('youngest_driverId'),
        first(when(col('youngest_rank') == 1, col('age'))).alias('youngest_age'),
        first(when(col('youngest_rank') == 1, col('forename'))).alias('youngest_forename'),
        first(when(col('youngest_rank') == 1, col('surname'))).alias('youngest_surname'),
        first(when(col('oldest_rank') == 1, col('driverId'))).alias('oldest_driverId'),
        first(when(col('oldest_rank') == 1, col('age'))).alias('oldest_age'),
        first(when(col('oldest_rank') == 1, col('forename'))).alias('oldest_forename'),
        first(when(col('oldest_rank') == 1, col('surname'))).alias('oldest_surname')
    )
    
display(df_youngest_oldest)

# COMMAND ----------

# For a given race, which driver has the most wins and losses?
race_id = 13  # Replace with desired raceId
df_wins_losses = df_results.filter(col('raceId') == race_id) \
    .groupBy('driverId') \
    .agg(F.sum(when(col('position') == 1, 1).otherwise(0)).alias('wins'),
         sum(when(col('position') != 1, 1).otherwise(0)).alias('losses')) \
    .join(df_drivers.select('driverId', 'forename', 'surname'), 'driverId') \
    .orderBy(col('wins').desc(), col('losses').desc())

display(df_wins_losses)

# COMMAND ----------

# My own question: Average lap time by driver nationality
df_lap_nationality = df_laptimes.join(df_drivers.select('driverId', 'nationality'), 'driverId') \
    .groupBy('nationality') \
    .agg(avg('milliseconds').alias('avg_lap_time')) \
    .orderBy('avg_lap_time')
display(df_lap_nationality)

# COMMAND ----------

df_pitstop_avg.write.option("header", "true").csv('s3://yw4407-gr5069/processed/inclass/pitstop_avg')
df_pitstop_ranked.write.option("header", "true").csv('s3://yw4407-gr5069/processed/inclass/pitstop_ranked')
df_lap_drivers.write.option("header", "true").csv('s3://yw4407-gr5069/processed/inclass/lap_drivers_with_code')
df_youngest_oldest.write.option("header", "true").csv('s3://yw4407-gr5069/processed/inclass/race_age')
df_wins_losses.write.option("header", "true").csv('s3://yw4407-gr5069/processed/inclass/wins_losses_race_')
df_lap_nationality.write.option("header", "true").csv('s3://yw4407-gr5069/processed/inclass/lap_nationality')

# COMMAND ----------

