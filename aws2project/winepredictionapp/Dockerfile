# ARG JRE_VERSION=11-jre

# Create a builder instance with the desired platform
FROM amazoncorretto:11 AS base


# Define default Spark version
ARG SPARK_VERSION_DEFAULT=3.4.1
# Define default Hadoop version
ARG HADOOP_VERSION_DEFAULT=3
# Define default Hadoop aws jar version
ARG HADOOP_AWS_VERSION_DEFAULT=3.2.0
# Define default aws sdk jar version
ARG AWS_SDK_BUNDLE_VERSION_DEFAULT=1.11.375
# Define default GCS connector jar version
ARG GCS_CONNECTOR_VERSION_DEFAULT=hadoop3-2.2.0

# Define ENV variables
ENV SPARK_VERSION=${SPARK_VERSION_DEFAULT}
ENV HADOOP_VERSION=${HADOOP_VERSION_DEFAULT}
ENV HADOOP_AWS_VERSION=${HADOOP_AWS_VERSION_DEFAULT}
ENV AWS_SDK_BUNDLE_VERSION=${AWS_SDK_BUNDLE_VERSION_DEFAULT}
ENV GCS_CONNECTOR_VERSION=${GCS_CONNECTOR_VERSION_DEFAULT}


RUN yum update \
    && yum install -y bash tini libc6 libpam-modules krb5-user libnss3 procps curl tar

FROM base AS spark-base

# Download and extract Spark
COPY spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz .
RUN tar -xvzf spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz 
RUN mv spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION} /opt/spark 

COPY entrypoint.sh /opt/spark

RUN chmod a+x /opt/spark/entrypoint.sh

FROM spark-base AS sparkbuilder

# Set SPARK_HOME
ENV SPARK_HOME=/opt/spark

# Extend PATH environment variable
ENV PATH=${PATH}:${SPARK_HOME}/bin

# Create the application directory
RUN mkdir -p /app

FROM sparkbuilder AS spark-with-s3-gcs

# Download S3 and GCS jars
COPY hadoop-aws-${HADOOP_AWS_VERSION}.jar ${SPARK_HOME}/jars/ 
COPY aws-java-sdk-bundle-${AWS_SDK_BUNDLE_VERSION}.jar ${SPARK_HOME}/jars/

FROM spark-with-s3-gcs AS spark-with-jar

WORKDIR /app
# Add application jar in /app
ADD target/winepredictionapp-1.0-SNAPSHOT.jar /app

RUN echo "spark.hadoop.fs.s3.impl=org.apache.hadoop.fs.s3a.S3AFileSystem" >> $SPARK_HOME/conf/spark-defaults.conf 
RUN echo "spark.hadoop.fs.s3a.aws.credentials.provider=org.apache.hadoop.fs.s3a.AnonymousAWSCredentialsProvider" >> $SPARK_HOME/conf/spark-defaults.conf 



USER root

# ENTRYPOINT [ "/opt/spark/entrypoint.sh" ]
CMD ["spark-submit", "--class", "com.example.WineQualityPredictionApp", "--master", "local", "/app/winepredictionapp-1.0-SNAPSHOT.jar", "s3a://wineprediction-aws-project/data/ValidationDataset.csv"]
# CMD ["ls"]




# FROM openjdk:8-jdk-alpine

# ENV S3_DATASET_URL=s3://wineprediction-aws-project/data/ValidationDataset.csv

# WORKDIR /app

# COPY target/winepredictionapp-1.0-SNAPSHOT.jar /app/winepredictionapp-1.0-SNAPSHOT.jar

# ENV SPARK_VERSION 3.2.1
# ENV HADOOP_VERSION 3.3.0

# # Install curl and wget using apk
# RUN apk --update add curl wget

# # Copy the pre-downloaded Spark binary file
# COPY spark-3.2.1-bin-hadoop2.7.tgz .

# # Decompress the Spark binary file
# RUN tar -xzf spark-3.2.1-bin-hadoop2.7.tgz

# # Move the extracted Spark directory
# RUN mv spark-3.2.1-bin-hadoop2.7 spark

# # Explicitly set the PATH environment variable
# ENV PATH="/app/spark/bin:$PATH"

# # Set the JAR file as the entry point
# CMD ["bash", "-c", "export PATH='/app/spark/bin:$PATH' && spark-submit --class com.example.WineQualityPredictionApp --master local /app/winepredictionapp-1.0-SNAPSHOT.jar $S3_DATASET_URL"]
