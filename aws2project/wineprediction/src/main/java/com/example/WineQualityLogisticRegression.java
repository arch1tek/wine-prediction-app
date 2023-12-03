package com.example;

import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.classification.LogisticRegression;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;

public class WineQualityLogisticRegression {
    public static void main(String[] args) {
        SparkSession spark = SparkSession.builder().appName("WineQualityLogisticRegression").getOrCreate();

        // Load training data
        Dataset<Row> trainingData = spark.read().option("header", "true").option("delimiter", ";").csv("s3://wineprediction-aws-project/data/TrainingDataset.csv");
        // Dataset<Row> trainingData = spark.read().option("header", "true").csv("s3://wineprediction-aws-project/data/TrainingDataset.csv");
        trainingData.show();

        // Perform feature engineering
        // ...

        // Create feature vector
        
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
            trainingData = trainingData.withColumn(col, trainingData.col(col).cast("double"));
        }

        trainingData = trainingData.withColumn(target, trainingData.col(target).cast("int"));


        VectorAssembler assembler = new VectorAssembler().setInputCols(featureColumns).setOutputCol("features");

        // Create and train the model
        System.out.println("Training model");
        LogisticRegression lr = new LogisticRegression().setLabelCol("quality").setFeaturesCol("features");
        Pipeline pipeline = new Pipeline().setStages(new org.apache.spark.ml.PipelineStage[]{assembler, lr});
        PipelineModel trainedModel = pipeline.fit(trainingData);
        System.out.println("Model trained");

        // Load validation data
        Dataset<Row> validationData = spark.read().option("header", "true").option("delimiter", ";").csv("s3://wineprediction-aws-project/data/ValidationDataset.csv");
        // Dataset<Row> validationData = spark.read().option("header", "true").csv("s3://wineprediction-aws-project/data/ValidationDataset.csv");

        validationData.show();
        // Perform feature engineering on validation data
        // ...

        for (String col : featureColumns) {
            validationData = validationData.withColumn(col, validationData.col(col).cast("double"));
        }

        validationData = validationData.withColumn(target, validationData.col(target).cast("int"));

        // Transform validation data using the trained model


        Dataset<Row> predictions = trainedModel.transform(validationData);

        predictions.show();

        // Evaluate the model using F1 score
        MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
                .setLabelCol("quality")  // Set the correct label column with double quotes
                .setPredictionCol("prediction")
                .setMetricName("f1");

        double f1 = evaluator.evaluate(predictions);
        System.out.println("F1 Score on Validation Data: " + f1);

        // Save the model
        try {
            trainedModel.write().overwrite().save("s3://wineprediction-aws-project/model/wine_quality_logistic_regression_model");
            System.out.println("Model stored in S3");
        } catch (Exception e) {
            e.printStackTrace();
        }

        spark.stop();
    }
}
