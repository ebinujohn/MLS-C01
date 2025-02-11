# AWS Services to Study for Machine Learning - Specialty Exam

This document outlines the key AWS services and concepts to focus on when preparing for the AWS Certified Machine Learning - Specialty exam.

## Table of Contents

1.  [Data Engineering](#data-engineering)
2.  [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
3.  [Modeling](#modeling)
4.  [Machine Learning Implementation and Operations](#machine-learning-implementation-and-operations)
5.  [Machine Learning Frameworks and Algorithms](#machine-learning-frameworks-and-algorithms)
6.  [General AWS Knowledge](#general-aws-knowledge)

## Data Engineering

- [ ] **Amazon S3**
  - [ ] Bucket policies and access control
  - [ ] Versioning
  - [ ] Lifecycle rules
  - [ ] Encryption (SSE-S3, SSE-KMS, SSE-C)
  - [ ] Storage classes (Standard, Intelligent-Tiering, Glacier, etc.)
  - [ ] Transfer Acceleration
  - [ ] S3 Select and Glacier Select
- [ ] **AWS Glue**
  - [ ] Data Catalog
  - [ ] ETL jobs (Spark, Python Shell)
  - [ ] Crawlers
  - [ ] Triggers
  - [ ] Workflows
  - [ ] Development endpoints
- [ ] **Amazon Kinesis**
  - [ ] Kinesis Data Streams
  - [ ] Kinesis Data Firehose
  - [ ] Kinesis Data Analytics (SQL and Apache Flink)
  - [ ] Kinesis Video Streams
- [ ] **Amazon EMR**
  - [ ] Hadoop ecosystem (HDFS, Hive, Pig, etc.)
  - [ ] Spark
  - [ ] Cluster management and scaling
  - [ ] Security configurations
- [ ] **AWS Data Pipeline**
  - [ ] Scheduling and orchestration
  - [ ] Data transformation activities
- [ ] **Amazon Redshift**
  - [ ] Data warehousing concepts
  - [ ] Distribution styles and sort keys
  - [ ] Redshift Spectrum
- [ ] **Amazon Athena**
  - [ ] Serverless querying of S3 data
  - [ ] Presto query engine
  - [ ] Integration with Glue Data Catalog
- [ ] **AWS Lake Formation**
  - [ ] Building and managing data lakes
  - [ ] Fine-grained access control
- [ ] **AWS Database Migration Service (DMS)**
- [ ] **AWS Glue DataBrew**
- [ ] **Amazon QuickSight**

## Exploratory Data Analysis (EDA)

- [ ] **Amazon SageMaker Data Wrangler**
  - [ ] Data import from various sources
  - [ ] Data transformation and feature engineering
  - [ ] Data visualization
  - [ ] Exporting data and code
- [ ] **Amazon SageMaker Studio**
  - [ ] Notebooks
  - [ ] Integration with other SageMaker features
- [ ] **Amazon SageMaker Notebook Instances**
- [ ] **Jupyter Notebooks**
- [ ] **Pandas**
- [ ] **Matplotlib**
- [ ] **Seaborn**
- [ ] **AWS Glue DataBrew**

## Modeling

- [ ] **Amazon SageMaker**
  - [ ] Built-in algorithms (linear learner, XGBoost, image classification, etc.)
    - [ ] Common hyperparameters for each algorithm
    - [ ] Input/output data formats
  - [ ] Training jobs
    - [ ] Instance types and scaling
    - [ ] Hyperparameter tuning
    - [ ] Automatic Model Tuning (HPO)
    - [ ] Distributed training
  - [ ] Model hosting
    - [ ] Endpoints
    - [ ] Endpoint configurations
    - [ ] Variant configurations
    - [ ] Auto scaling
    - [ ] A/B testing
  - [ ] Batch transform
  - [ ] SageMaker Debugger
  - [ ] SageMaker Model Monitor
  - [ ] SageMaker Clarify
  - [ ] SageMaker Experiments
  - [ ] SageMaker Autopilot
  - [ ] SageMaker Canvas
  - [ ] SageMaker Ground Truth
  - [ ] SageMaker Edge Manager
- [ ] **Scikit-learn**
  - [ ] Common algorithms and techniques
  - [ ] Model evaluation metrics
- [ ] **Deep Learning Concepts**
  - [ ] Neural networks
  - [ ] Activation functions
  - [ ] Loss functions
  - [ ] Optimization algorithms (gradient descent, Adam, etc.)
  - [ ] Regularization techniques
- [ ] **Reinforcement Learning Concepts**

## Machine Learning Implementation and Operations

- [ ] **Amazon SageMaker Pipelines**
  - [ ] Building and managing ML workflows
  - [ ] Pipeline steps and parameters
  - [ ] Integration with other SageMaker features
- [ ] **Amazon SageMaker Projects**
  - [ ] MLOps templates for common use cases
- [ ] **AWS Step Functions**
  - [ ] Orchestrating complex workflows
  - [ ] Integration with SageMaker and other AWS services
- [ ] **AWS CodePipeline**
  - [ ] CI/CD for ML models
- [ ] **AWS CodeBuild**
- [ ] **AWS CodeDeploy**
- [ ] **AWS CodeCommit**
- [ ] **Amazon CloudWatch**
  - [ ] Monitoring metrics and logs
  - [ ] Alarms
  - [ ] Events
- [ ] **AWS CloudFormation**
  - [ ] Infrastructure as code for ML resources
- [ ] **Amazon ECR**
- [ ] **Amazon ECS**
- [ ] **Amazon EKS**
- [ ] **Amazon EventBridge**

## Machine Learning Frameworks and Algorithms

- [ ] **SageMaker Built-in Algorithms** (Detailed understanding)
  - [ ] Linear Learner
  - [ ] XGBoost
  - [ ] K-Means
  - [ ] Principal Component Analysis (PCA)
  - [ ] Factorization Machines
  - [ ] Image Classification
  - [ ] Object Detection
  - [ ] Semantic Segmentation
  - [ ] Seq2Seq
  - [ ] BlazingText
  - [ ] DeepAR
  - [ ] And others!
- [ ] **TensorFlow** (on AWS)
  - [ ] Using TensorFlow within SageMaker
  - [ ] TensorFlow Serving
- [ ] **PyTorch** (on AWS)
  - [ ] Using PyTorch within SageMaker
- [ ] **MXNet** (on AWS)
  - [ ] Using MXNet within SageMaker
- [ ] **Scikit-learn** (on AWS)
  - [ ] Using Scikit-learn within SageMaker
- [ ] **Amazon Comprehend**
- [ ] **Amazon Rekognition**
- [ ] **Amazon Translate**
- [ ] **Amazon Polly**
- [ ] **Amazon Transcribe**
- [ ] **Amazon Lex**
- [ ] **Amazon Personalize**
- [ ] **Amazon Forecast**

## General AWS Knowledge

- [ ] **AWS Global Infrastructure**
- [ ] **AWS Shared Responsibility Model**
- [ ] **AWS Well-Architected Framework**
- [ ] **IAM** (Roles, Policies, Permissions)
- [ ] **Basic Networking Concepts** (VPC, subnets, security groups, etc.)
