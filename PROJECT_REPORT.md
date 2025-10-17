# Movie Recommender — Final Year Project Report

Version: 1.0

Author(s): Project team

Date: October 2025

---

## Executive summary

This project implements an end-to-end movie recommendation system using collaborative filtering (Alternating Least Squares, ALS) implemented with Apache Spark (MLlib) and Scala. The system is trained and validated on the MovieLens 100k dataset, and demonstrates how to preprocess rating data, train a latent-factor recommendation model, persist model artifacts, and generate recommendations for users and items.

The deliverables include: production-ready Scala code for training and generating recommendations, a PySpark notebook version for demonstration, and a reproducible build configuration using Maven (`pom.xml`). This report documents the full system design, implementation choices, evaluation, reproducibility, limitations, and opportunities for future work.

## Project objectives

- Design and implement a scalable recommendation system for movies using collaborative filtering.
- Demonstrate data ingestion, preprocessing, model training, persistence, and serving (recommendation generation).
- Justify technology and architecture choices for a production-style pipeline.
- Evaluate model outputs and provide guidance for improvements and extensions.

## Motivation and scope

Recommender systems are critical for modern online services (streaming platforms, e-commerce, social media). The aim of this project is to build a compact, understandable pipeline that shows the core components used in industry: data preparation, model training, artifact storage, and runtime recommendation generation. We focus on collaborative filtering because it directly models user-item interactions and is effective when interaction data (ratings) are available.

Scope exclusions:
- This project does not implement advanced production components such as real-time feature stores, A/B testing infrastructure, online model updates, or full UX integration. These are listed as future work.

## Dataset

- Dataset used: MovieLens 100k (u.data / u.item). This dataset contains 100,000 ratings from 943 users on 1,682 movies.
- Key files used:
  - `u.data`: ratings with (userID, itemID, rating, timestamp).
  - `u.item`: movie metadata (movieID | title | release | ...).

Why MovieLens 100k:
- Small and well-known: easy to download, experiment with and reproduce results.
- Contains realistic rating behaviour suitable for collaborative filtering experiments.

## Model and algorithms

We use collaborative filtering implemented by the Alternating Least Squares (ALS) algorithm from Spark MLlib (RDD-based API). Key reasons:

- ALS (matrix factorization) efficiently models latent user/item factors from sparse rating matrices.
- It scales to large datasets when used in Spark's distributed environment.
- The MLlib implementation is mature, well-documented, and suitable for batch training jobs.

Model hyperparameters used (configurable in code):
- rank: number of latent factors (e.g., 5 or 10)
- iterations: number of ALS iterations (e.g., 20)
- lambda: regularization parameter (e.g., 0.1)

Training objective: minimize regularized squared-error on observed ratings while alternating updates between user and item factor matrices.

## System design and architecture

High-level components:

1. Data ingestion and preprocessing (Scala Spark job):
   - Read `u.data` as text, parse and map into MLlib `Rating(user, item, rating)` objects.
2. Model training (Scala Spark job `RecommenderTrain`):
   - Train ALS using the ratings RDD, checkpointing as needed to avoid deep lineage.
   - Persist the trained `MatrixFactorizationModel` to storage (local or HDFS path).
3. Recommendation generation (Scala Spark job `Recommend`):
   - Load saved model and `u.item` movie titles.
   - Generate top-N recommendations per user (`model.recommendProducts(user, n)`) or top-N users for an item (`model.recommendUsers(item, n)`).
4. Demonstration / notebook (PySpark Databricks notebook):
   - A notebook version demonstrates the same workflow using Python for easier exploration and visualization.

Storage and deployment considerations:
- Models are saved as Spark `MatrixFactorizationModel` objects. In a production setup, store artifacts in HDFS/S3 and version them with a metadata store.
- For local testing the code uses `file://` paths; for cluster deployment switch to HDFS (e.g., `hdfs:///user/<you>/movie/ALSmodel`).

## Technology choices — rationale

1. Apache Spark
   - Rationale: scalable distributed data processing framework with built-in MLlib algorithms (including ALS). It simplifies parallel training and data processing and integrates well with HDFS and common big-data ecosystems.

2. Scala
   - Rationale: Native language for Spark — Spark APIs are first-class in Scala. Using Scala provides compact code and avoids serialization/performance pitfalls sometimes encountered with other languages.

3. Spark MLlib (RDD-based ALS)
   - Rationale: Mature matrix factorization implementation suitable for batch training on distributed data. For this project MLlib's ALS provides straightforward APIs for training and recommending. For advanced use, consider the DataFrame-based API (`org.apache.spark.ml.recommendation.ALS`) which integrates with pipelines.

4. Maven (`pom.xml`)
   - Rationale: Project uses Maven for reproducible builds and artifact management. Maven is widely used in Java/Scala ecosystems and fits well with CI/CD pipelines.

5. MovieLens dataset
   - Rationale: Standard benchmarking dataset familiar to ML and recommender communities. Small, reproducible, and sufficient for demonstrating core ideas.

## Implementation details

- Language: Scala (primary). PySpark notebook included for demonstration.
- Build: Maven (`pom.xml`) — produces `target/MovieRecommender-1.0.jar`.
- Main programs:
  - `RecommenderTrain.scala`: initializes SparkContext, reads ratings, trains ALS, saves model.
  - `Recommend.scala`: reads movie metadata, loads model, and prints recommendations.

Key implementation notes:
- The code includes checkpointing for RDDs to avoid deep DAGs and potential stack-overflow issues.
- The Scala code uses simple, explicit path strings for quick testing. For production, the code should accept CLI arguments or a configuration file to specify input, model and checkpoint paths.

## How to reproduce (recommended commands)

1. Build the project (Maven):

```bash
mvn -DskipTests package
```

This produces `target/MovieRecommender-1.0.jar`.

2. Train a model locally using Spark (example):

```bash
spark-submit --master local[*] --class RecommenderTrain target/MovieRecommender-1.0.jar
```

3. Generate recommendations for a movie (example):

```bash
spark-submit --master local[*] --class Recommend target/MovieRecommender-1.0.jar --M 5
```

4. Generate recommendations for a user (example):

```bash
spark-submit --master local[*] --class Recommend target/MovieRecommender-1.0.jar --U 13
```

Notes:
- Ensure Spark and Java are installed and `spark-submit` is available on PATH. For cluster runs change `--master` and paths to HDFS as required.

## Evaluation and results

This project demonstrates correct training and inference pipelines. For objective evaluation:

- Split the ratings into train/test and compute RMSE on held-out ratings. The codebase currently uses the whole dataset for demonstration; adding a train/test split and RMSE computation is recommended.
- Use ranking metrics (Precision@K, NDCG) to evaluate top-N recommendation quality.

Suggested quick evaluation (not currently in code):
1. Randomly split ratings RDD into train/test (e.g., 80/20).
2. Train model on train set, predict on user-item pairs in test set and compute RMSE.
3. For ranking, compute Precision@K by comparing top-K recommendations against items in the test set for each user.

## Limitations

- Cold-start: ALS-based collaborative filtering requires historical ratings. New users or items with no ratings cannot be recommended accurately.
- Model interpretability: Latent factors are not directly interpretable without additional item features.
- RDD-based MLlib ALS lacks tight DataFrame integration and pipeline utilities compared to `org.apache.spark.ml`.
- Hard-coded paths in the Scala source make it less flexible for deployment; parameterization is needed for production.

## Future work and extensions

1. Parameterize paths and hyperparameters via CLI or configuration file.
2. Migrate to DataFrame-based Spark ML `ALS` for better integration with Spark SQL and Pipelines.
3. Add train/test split and automated evaluation (RMSE, Precision@K, recall, NDCG).
4. Implement item/user cold-start mitigation: hybrid model with content-based features (movie metadata) or side-information using factorization machines or deep learning.
5. Add CI to run builds and unit tests, and create a release artifact (fat JAR) automatically.
6. Build a simple web API (REST) that loads the saved model and serves recommendations for users (e.g., using Play Framework, Akka HTTP or a lightweight Python Flask service that calls Spark job or precomputed recommendations).

## Ethical considerations

- Recommendations can introduce filter bubbles or inadvertently amplify biases present in historical data. For production systems, monitor for fairness and diversity, and consider algorithms or post-processing steps that improve diversity and reduce bias.

## Conclusion

This project demonstrates a complete, reproducible pipeline for training and serving collaborative-filtering based recommendations on MovieLens 100k. The system uses industry-grade technologies (Spark, Scala, Maven) that scale to much larger datasets. The codebase is a solid foundation for further improvements such as evaluation, parameterization, cold-start handling, and production deployment.

---

If you want, I can also:
- Produce a slide-deck summarizing this report for presentations.
- Parameterize the Scala code (CLI arguments) and add a small test harness to run training + recommendation automatically.
- Run `mvn package` and execute `spark-submit` locally to produce example output for your report.
