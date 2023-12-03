package com.example;

// WineQualityPredictionApp.java

import org.apache.spark.ml.PipelineModel;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

public class WineQualityPredictionApp {
    public static void main(String[] args) {
        if (args.length != 1) {
            System.out.println("Usage: WineQualityPredictionApp <test_data_path>");
            System.exit(1);
        }

        String testDataPath = args[0];

        SparkSession spark = SparkSession.builder().appName("WineQualityPredictionApp")
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
        .getOrCreate();

        // Load test data
        Dataset<Row> testData = spark.read().option("header", "true").option("delimiter", ";").csv(testDataPath);

        // Perform feature engineering on test data (if needed)
        // ...
        String[] featureColumns = {
            "fixed acidity",
            "volatile acidity",
            "citric acid",
            "residual sugar",
            "chlorides",
            "free sulfur dioxide",
            "total sulfur dioxide",
            "density",
            "pH",
            "sulphates",
            "alcohol"
        };

        String target = "quality";

        for (String col : featureColumns) {
            testData = testData.withColumn(col, testData.col(col).cast("double"));
        }
 
        testData = testData.withColumn(target, testData.col(target).cast("int"));

        // Transform test data using the trained model
        PipelineModel trainedModel = PipelineModel.load("s3a://wineprediction-aws-project/model/wine_quality_logistic_regression_model");
        Dataset<Row> predictions = trainedModel.transform(testData);

        predictions.show();

        // Stop Spark session
        spark.stop();
    }
}
