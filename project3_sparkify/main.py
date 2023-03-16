from pyspark.sql import SparkSession
from pyspark.sql.window import Window
import pyspark.sql.functions as F
import pyspark.sql.types as T
import pyspark.ml as M

import datetime
import pandas as pd
import numpy as np

def main():
    # create a spark session
    spark = SparkSession.builder.appName('sparkify').getOrCreate()

    # load dataset
    file = 's3n://udacity-dsnd/sparkify/sparkify_event_data.json'
    df = spark.read.json(file)

    # clean dataset (drop missing values)
    df = df.dropna(how='any', subset=['firstName'])

    # convert time format from timestamp to datetime
    time_func = F.udf(lambda x: datetime.datetime.fromtimestamp(x / 1000).strftime('%Y-%m-%d %H:%M:%S'))
    df = df.withColumns({'datetime': time_func('ts'), 'reg_datetime': time_func('registration')})

    # extract year, month, day of month, day of week, hour from datetime
    df = df.withColumns({
        'year': F.year(F.col('datetime')),
        'reg_year': F.year(F.col('reg_datetime')),
        'month': F.month(F.col('datetime')),
        'reg_month': F.month(F.col('reg_datetime')),
        'week_of_year': F.weekofyear(F.col('datetime')),
        'reg_woy': F.weekofyear(F.col('reg_datetime')),
        'day_of_month': F.dayofmonth(F.col('datetime')),
        'reg_dom': F.dayofmonth(F.col('reg_datetime')),
        'day_of_week': F.dayofweek(F.col('datetime')),
        'reg_dow': F.dayofweek(F.col('reg_datetime')),
        'hour': F.hour(F.col('datetime')),
        'reg_hour': F.hour(F.col('reg_datetime')),
    })

    # add label
    churn_user = df.filter(df['page']=='Cancellation Confirmation')\
        .select('userId').distinct()\
        .withColumn('label', F.lit(1))

    nonchurn_user = df.filter(~df['userId'].isin(churn_user.select('userId').rdd.flatMap(lambda x: x).collect()))\
        .select('userId').distinct()\
        .withColumn('label', F.lit(0))

    df_churn = churn_user.union(nonchurn_user)
    df = df.join(df_churn, on=['userId'])

    # number of artists
    g_artist = df.groupBy('label', 'userId').agg(F.count_distinct('artist').alias('num_artists'))

    # length of songs
    g_length = df.groupBy('label', 'userId').agg(
        F.sum('length').alias('sum_length'),
        F.avg('length').alias('mean_length'),
        F.max('length').alias('max_length'),
        F.min('length').alias('min_length'),
        F.expr('percentile_approx(length, 0.5)').alias('median_length')
    )

    # gender
    d_gender = df.select('label', 'userId', 'gender').dropDuplicates()

    # level
    level_func = F.udf(lambda x: int(x=='paid'), T.IntegerType())
    d_level = df.withColumn('is_paid', level_func('level'))\
        .groupBy('label','userId').agg(F.max('is_paid').alias('is_paid'))

    # location
    state_func = F.udf(lambda x: x[-2:])
    d_state = df.withColumn('state', state_func('location'))\
        .select('label','userId','state')\
        .dropDuplicates()

    # page
    g_page = df.filter(~df['page'].isin(['Cancellation Confirmation','Cancel']))\
        .groupBy('label','userId')\
        .pivot('page')\
        .count()\
        .fillna(0)

    # days since registration
    g_lifetime = df.groupBy('label', 'userId').agg(
        F.max('ts').alias('max_dt'),
        F.min('registration').alias('reg_dt')
    ).withColumn('lifetime_day', (F.col('max_dt') - F.col('reg_dt'))/1000/60/60/24)\
    .select('label','userId','lifetime_day')

    # number of songs
    g_songs = df.groupBy('label','userId').agg(
        F.count('song').alias('num_songs'),
        F.count_distinct('song').alias('num_unique_songs')
    )

    # average times per song
    g_song_times = df.groupBy('label','userId','song').count()\
        .groupBy('label','userId')\
        .agg(F.avg('count').alias('avg_times_per_song'))


    # pct of pages a users hit each hour
    g_hour = df.groupBy('label','userId','hour').agg(F.count('page').alias('num_pages'))\
        .groupBy('label','userId').pivot('hour').sum('num_pages').fillna(0)\

    g_page_ = df.groupBy('label','userId').agg(F.count('page'))
    g_hour = g_hour.join(g_page_, on=['label','userId'])

    for i in range(24):
        g_hour = g_hour.withColumn(str(i), (F.col(str(i))/F.col('count(page)')))\
            .withColumnRenamed(str(i), f'num_pages_hr{i}')

    g_hour = g_hour.drop('count(page)')

    # device
    device_func = F.udf(lambda x: x.split('(')[1].replace(';',' ').split(' ')[0])
    g_device = df.withColumn('device', device_func('userAgent'))\
            .select('label','userId','device')\
            .dropDuplicates()

    # join all features
    df_features = g_artist.join(g_artist, on=['label','userId'])\
            .join(g_length, on=['label','userId'])\
            .join(d_gender, on=['label','userId'])\
            .join(d_level, on=['label','userId'])\
            .join(d_state, on=['label','userId'])\
            .join(g_page, on=['label','userId'])\
            .join(g_lifetime, on=['label','userId'])\
            .join(g_songs, on=['label','userId'])\
            .join(g_song_times, on=['label','userId'])\
            .join(g_hour, on=['label','userId'])\
            .join(g_device, on=['label','userId'])

    # pipeline
    str_cols = ['gender','state','device']
    str_indexer = M.feature.StringIndexer(
        inputCols=str_cols,
        outputCols=[f'{col}_idx' for col in str_cols]
    )

    assembler_cols = df_features.columns
    assembler_cols = list(set(assembler_cols).difference(set(str_cols)).difference(set(['label','userId']))) + [f'{col}_idx' for col in str_cols]
    assembler = M.feature.VectorAssembler(inputCols=assembler_cols, outputCol='features')

    features_pipeline = M.Pipeline(stages=[
        str_indexer,
        assembler
    ])

    df_model = features_pipeline.fit(df_features).transform(df_features)

    # train test split
    train, test = df_model.randomSplit([.8,.2], seed=2022)

    clf = M.classification.LinearSVC()
    param_grid = M.tuning.ParamGridBuilder() \
        .addGrid(clf.maxIter, [50, 100, 500]) \
        .addGrid(clf.regParam, [0.01, 0.05, 0.1]) \
        .build()

    crossval = M.tuning.CrossValidator(
        estimator=M.Pipeline(stages=[clf]),
        estimatorParamMaps=param_grid,
        evaluator=M.evaluation.MulticlassClassificationEvaluator(metricName='f1'),
        numFolds=3
    )

    model = crossval.fit(train)
    results = model.transform(test)

    # accuracy
    acc_evaluator = M.evaluation.MulticlassClassificationEvaluator(metricName='accuracy')
    accuracy = acc_evaluator.evaluate(results.select(F.col('label'), F.col('prediction')))

    # f1 score
    f1_evaluator = M.evaluation.MulticlassClassificationEvaluator(metricName='f1')
    f1_score = f1_evaluator.evaluate(results.select(F.col('label'), F.col('prediction')))

    print(f'Support Vector Machine\n\taccuracy on test set: {accuracy}\n\tf1 score on test set: {f1_score}')

if __name__ == '__main__':
    main()