# Useful Links for AWS Certified Machine Learning - Specialty Exam

This document contains a curated list of resources that can be helpful for preparing for the AWS Certified Machine Learning - Specialty exam. The links are categorized by topic for easier navigation.

## Table of Contents

1.  [Official AWS Resources](#official-aws-resources)
2.  [Data Engineering](#data-engineering)
3.  [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
4.  [Modeling](#modeling)
5.  [Machine Learning Implementation and Operations](#machine-learning-implementation-and-operations)
6.  [Machine Learning Frameworks](#machine-learning-frameworks)
    - [SageMaker Built-in Algorithms](#sagemaker-built-in-algorithms)
    - [TensorFlow on AWS](#tensorflow-on-aws)
    - [PyTorch on AWS](#pytorch-on-aws)
    - [MXNet on AWS](#mxnet-on-aws)
7.  [Blogs, Articles, and Whitepapers](#blogs-articles-and-whitepapers)
8.  [Practice Exams and Questions](#practice-exams-and-questions)
9.  [Video Courses and Tutorials](#video-courses-and-tutorials)
10. [Other Useful Resources](#other-useful-resources)

## Official AWS Resources

- **Exam Readiness: AWS Certified Machine Learning - Specialty (MLS-C01):** [https://explore.skillbuilder.aws/learn/courses/27/exam-readiness-aws-certified-machine-learning-specialty-mls-c01](https://explore.skillbuilder.aws/learn/courses/27/exam-readiness-aws-certified-machine-learning-specialty-mls-c01) - Start with this AWS Builder course to get an understanding of what topics are included in the exam and what to study.
- **AWS Certified Machine Learning - Specialty Exam Guide:** [https://d1.awsstatic.com/training-and-certification/docs-ml/AWS-Certified-Machine-Learning-Specialty_Exam-Guide.pdf](https://d1.awsstatic.com/training-and-certification/docs-ml/AWS-Certified-Machine-Learning-Specialty_Exam-Guide.pdf) - The official exam guide outlining the exam domains, objectives, and sample questions.
- **AWS Certified Machine Learning - Specialty Sample Questions:** [https://d1.awsstatic.com/training-and-certification/docs-ml/AWS-Certified-Machine-Learning-Specialty_Sample-Questions.pdf](https://d1.awsstatic.com/training-and-certification/docs-ml/AWS-Certified-Machine-Learning-Specialty_Sample-Questions.pdf) - Get a feel for the type of questions on the exam.
- **AWS Machine Learning Documentation:** [https://docs.aws.amazon.com/machine-learning/index.html](https://docs.aws.amazon.com/machine-learning/index.html) - Comprehensive documentation on all AWS ML services and features.
- **AWS Well-Architected Framework (Machine Learning Lens):** [https://docs.aws.amazon.com/wellarchitected/latest/machine-learning-lens/index.html](https://docs.aws.amazon.com/wellarchitected/latest/machine-learning-lens/index.html)
- **AWS Architecture Best Practices for Machine Learning:** [https://aws.amazon.com/architecture/machine-learning/](https://aws.amazon.com/architecture/machine-learning/)
- **AWS Machine Learning Blog:** [https://aws.amazon.com/blogs/machine-learning/](https://aws.amazon.com/blogs/machine-learning/) - Latest news, updates, and deep dives on AWS ML.
- **AWS Training and Certification:** [https://aws.amazon.com/training/](https://aws.amazon.com/training/) - Official AWS training courses, including ML-related courses.
- **Machine Learning on AWS:** [https://aws.amazon.com/machine-learning/](https://aws.amazon.com/machine-learning/) - General overview of AWS ML services.

## Data Engineering

- **Amazon S3 Documentation:** [https://docs.aws.amazon.com/s3/index.html](https://docs.aws.amazon.com/s3/index.html) - For storing and retrieving data.
- **AWS Glue Documentation:** [https://docs.aws.amazon.com/glue/index.html](https://docs.aws.amazon.com/glue/index.html) - For data integration, ETL, and data cataloging.
- **Amazon Kinesis Documentation:** [https://docs.aws.amazon.com/kinesis/index.html](https://docs.aws.amazon.com/kinesis/index.html) - For real-time data streaming.
- **Amazon EMR Documentation:** [https://docs.aws.amazon.com/emr/index.html](https://docs.aws.amazon.com/emr/index.html) - For big data processing using Spark and Hadoop.
- **AWS Data Pipeline Documentation:** [https://docs.aws.amazon.com/data-pipeline/index.html](https://docs.aws.amazon.com/data-pipeline/index.html) - For orchestrating data movement and processing.
- **Amazon Redshift Documentation:** [https://docs.aws.amazon.com/redshift/index.html](https://docs.aws.amazon.com/redshift/index.html) - For data warehousing.
- **AWS Lake Formation Documentation:** [https://docs.aws.amazon.com/lake-formation/index.html](https://docs.aws.amazon.com/lake-formation/index.html)
- **Amazon Athena Documentation:** [https://docs.aws.amazon.com/athena/index.html](https://docs.aws.amazon.com/athena/index.html)

## Exploratory Data Analysis (EDA)

- **Amazon SageMaker Data Wrangler Documentation:** [https://docs.aws.amazon.com/sagemaker/latest/dg/data-wrangler.html](https://docs.aws.amazon.com/sagemaker/latest/dg/data-wrangler.html) - For importing, transforming, and analyzing data within SageMaker.
- **Amazon SageMaker Studio Documentation:** [https://docs.aws.amazon.com/sagemaker/latest/dg/studio.html](https://docs.aws.amazon.com/sagemaker/latest/dg/studio.html) - IDE for machine learning in SageMaker.
- **Pandas Documentation:** [https://pandas.pydata.org/docs/](https://pandas.pydata.org/docs/) - Widely used library for data manipulation and analysis in Python.
- **Matplotlib Documentation:** [https://matplotlib.org/stable/contents.html](https://matplotlib.org/stable/contents.html) - Popular library for data visualization in Python.
- **Seaborn Documentation:** [https://seaborn.pydata.org/](https://seaborn.pydata.org/) - Statistical data visualization library based on Matplotlib.

## Modeling

- **Amazon SageMaker Documentation:** [https://docs.aws.amazon.com/sagemaker/index.html](https://docs.aws.amazon.com/sagemaker/index.html) - Comprehensive service for building, training, and deploying ML models.
- **SageMaker Autopilot Documentation:** [https://docs.aws.amazon.com/sagemaker/latest/dg/autopilot.html](https://docs.aws.amazon.com/sagemaker/latest/dg/autopilot.html)
- **SageMaker Debugger Documentation:** [https://docs.aws.amazon.com/sagemaker/latest/dg/debugger.html](https://docs.aws.amazon.com/sagemaker/latest/dg/debugger.html)
- **SageMaker Model Monitor Documentation:** [https://docs.aws.amazon.com/sagemaker/latest/dg/model-monitor.html](https://docs.aws.amazon.com/sagemaker/latest/dg/model-monitor.html)
- **Scikit-learn Documentation:** [https://scikit-learn.org/stable/](https://scikit-learn.org/stable/) - Popular ML library for Python.

## Machine Learning Implementation and Operations

- **Amazon SageMaker Pipelines Documentation:** [https://docs.aws.amazon.com/sagemaker/latest/dg/pipelines.html](https://docs.aws.amazon.com/sagemaker/latest/dg/pipelines.html) - For creating and managing ML workflows.
- **Amazon SageMaker Projects Documentation:** [https://docs.aws.amazon.com/sagemaker/latest/dg/projects.html](https://docs.aws.amazon.com/sagemaker/latest/dg/projects.html)
- **AWS Step Functions Documentation:** [https://docs.aws.amazon.com/step-functions/index.html](https://docs.aws.amazon.com/step-functions/index.html) - For coordinating distributed applications and microservices.
- **Amazon CloudWatch Documentation:** [https://docs.aws.amazon.com/cloudwatch/index.html](https://docs.aws.amazon.com/cloudwatch/index.html) - For monitoring and logging.
- **AWS CloudFormation Documentation:** [https://docs.aws.amazon.com/cloudformation/](https://docs.aws.amazon.com/cloudformation/) - For infrastructure as code.

## Machine Learning Frameworks

### SageMaker Built-in Algorithms

- **SageMaker Built-in Algorithms Documentation:** [https://docs.aws.amazon.com/sagemaker/latest/dg/algos.html](https://docs.aws.amazon.com/sagemaker/latest/dg/algos.html) - Learn about the various algorithms provided by SageMaker.
- **XGBoost:** [https://docs.aws.amazon.com/sagemaker/latest/dg/xgboost.html](https://docs.aws.amazon.com/sagemaker/latest/dg/xgboost.html)
- **Linear Learner:** [https://docs.aws.amazon.com/sagemaker/latest/dg/linear-learner.html](https://docs.aws.amazon.com/sagemaker/latest/dg/linear-learner.html)
- **K-Means:** [https://docs.aws.amazon.com/sagemaker/latest/dg/k-means.html](https://docs.aws.amazon.com/sagemaker/latest/dg/k-means.html)
- **Principal Component Analysis (PCA):** [https://docs.aws.amazon.com/sagemaker/latest/dg/pca.html](https://docs.aws.amazon.com/sagemaker/latest/dg/pca.html)
- **And more!** - Explore the full list in the documentation.

### TensorFlow on AWS

- **TensorFlow on AWS:** [https://aws.amazon.com/tensorflow/](https://aws.amazon.com/tensorflow/) - Resources for running TensorFlow on AWS.
- **TensorFlow Documentation:** [https://www.tensorflow.org/](https://www.tensorflow.org/) - Official TensorFlow documentation.

### PyTorch on AWS

- **PyTorch on AWS:** [https://aws.amazon.com/pytorch/](https://aws.amazon.com/pytorch/) - Resources for running PyTorch on AWS.
- **PyTorch Documentation:** [https://pytorch.org/](https://pytorch.org/) - Official PyTorch documentation.

### MXNet on AWS

- **MXNet on AWS:** [https://aws.amazon.com/mxnet/](https://aws.amazon.com/mxnet/) - Resources for running MXNet on AWS.
- **MXNet Documentation:** [https://mxnet.apache.org/](https://mxnet.apache.org/) - Official MXNet documentation.

## Blogs, Articles, and Whitepapers

- **AWS Machine Learning Blog:** [https://aws.amazon.com/blogs/machine-learning/](https://aws.amazon.com/blogs/machine-learning/) - A great source for articles and tutorials.
- **Towards Data Science:** [https://towardsdatascience.com/](https://towardsdatascience.com/) - A Medium publication with numerous articles on ML and AWS.
- **KDnuggets:** [https://www.kdnuggets.com/](https://www.kdnuggets.com/) - A leading site on AI, Analytics, Big Data, Data Mining, Data Science, and Machine Learning.

## Practice Exams and Questions

- **Tutorials Dojo AWS Certified Machine Learning - Specialty Practice Exams:** [https://tutorialsdojo.com/courses/aws-certified-machine-learning-specialty-practice-exams/](https://tutorialsdojo.com/courses/aws-certified-machine-learning-specialty-practice-exams/)
- **Whizlabs AWS Certified Machine Learning - Specialty Practice Tests:** [https://www.whizlabs.com/aws-certified-machine-learning-specialty/](https://www.whizlabs.com/aws-certified-machine-learning-specialty/)
- **A Cloud Guru AWS Certified Machine Learning - Specialty Practice Exam Questions:** [https://acloudguru.com/](https://acloudguru.com/)

## Video Courses and Tutorials

- **A Cloud Guru - AWS Certified Machine Learning - Specialty:** [https://acloudguru.com/course/aws-certified-machine-learning-specialty](https://acloudguru.com/course/aws-certified-machine-learning-specialty)
- **Udemy - AWS Certified Machine Learning Specialty:** [https://www.udemy.com/topic/aws-certified-machine-learning-specialty/](https://www.udemy.com/topic/aws-certified-machine-learning-specialty/)

## Other Useful Resources

- **Papers with Code:** [https://paperswithcode.com/](https://paperswithcode.com/) - Browse machine learning papers with code implementations.
- **Kaggle:** [https://www.kaggle.com/](https://www.kaggle.com/) - A platform for data science competitions and learning.
