package paristech

import org.apache.spark.SparkConf
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{CountVectorizer, CountVectorizerModel, HashingTF, IDF, OneHotEncoder, OneHotEncoderEstimator, RegexTokenizer, StopWordsRemover, StringIndexer, VectorAssembler}
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit}
import org.apache.spark.sql.SparkSession


object Trainer {

  def main(args: Array[String]): Unit = {

    val conf = new SparkConf().setAll(Map(
      "spark.scheduler.mode" -> "FIFO",
      "spark.speculation" -> "false",
      "spark.reducer.maxSizeInFlight" -> "48m",
      "spark.serializer" -> "org.apache.spark.serializer.KryoSerializer",
      "spark.kryoserializer.buffer.max" -> "1g",
      "spark.shuffle.file.buffer" -> "32k",
      "spark.default.parallelism" -> "12",
      "spark.sql.shuffle.partitions" -> "12",
      "spark.driver.maxResultSize" -> "2g"
    ))

    val spark = SparkSession
      .builder
      .config(conf)
      .appName("TP Spark : Trainer")
      .getOrCreate()


    /*******************************************************************************
      *
      *       TP 3
      *
      *       - lire le fichier sauvegarder précédemment
      *       - construire les Stages du pipeline, puis les assembler
      *       - trouver les meilleurs hyperparamètres pour l'entraînement du pipeline avec une grid-search
      *       - Sauvegarder le pipeline entraîné
      *
      *       if problems with unimported modules => sbt plugins update
      *
      ********************************************************************************/

    println("hello world ! from Trainer")
    println("Step 1: prepare the data")

    // Load the parquet data
    val df = spark.read.parquet("datasets/prepared_trainingset")

    // Stage1 : seperate the text file into words/tokens
    val regexTokenizer = new RegexTokenizer()
      .setPattern("\\W+")
      .setGaps(true)
      .setInputCol("text")
      .setOutputCol("tokens")

    // Stage2 : drop all the stop words
    val remover = new StopWordsRemover()
      .setInputCol("tokens")
      .setOutputCol("filtered")

    // Stage3 and 4 : calculate the TF-IDF
    val tf = new CountVectorizer()
      .setInputCol("filtered")
      .setOutputCol("vect_filter")

    val idf = new IDF()
      .setInputCol("vect_filter")
      .setOutputCol("tfidf")

    // Stage 5 and 6: convert country2 currency2 to numeric
    val indexer1 = new StringIndexer()
      .setInputCol("country2")
      .setOutputCol("country_indexed")

    val indexer2 = new StringIndexer()
      .setInputCol("currency2")
      .setOutputCol("currency_indexed")

    // Stage 7 and 8: one-hot encode
    val encoder = new OneHotEncoderEstimator()
      .setInputCols(Array("country_indexed","currency_indexed"))
      .setOutputCols(Array("country_onehot","currency_onehot"))

    // Stage 9: assemble all features to a vector
    val assembler = new VectorAssembler()
      .setInputCols(Array("tfidf", "days_campaign", "hours_prepa", "goal", "country_onehot", "currency_onehot" ))
      .setOutputCol("features")

    // Stage 10: logistic regression
    val lr = new LogisticRegression()
      .setElasticNetParam(0.0)
      .setFitIntercept(true)
      .setFeaturesCol("features")
      .setLabelCol("final_status")
      .setStandardization(true)
      .setPredictionCol("predictions")
      .setRawPredictionCol("raw_predictions")
      .setThresholds(Array(0.7, 0.3))
      .setTol(1.0e-6)
      .setMaxIter(20)

    // Create Pipeline
    println("Step 2: create Pipeline")
    val pipeline = new Pipeline()
      .setStages(Array(regexTokenizer, remover, tf, idf, indexer1, indexer2,
      encoder, assembler, lr))

    // Train a model for the predictions
    println("Step 3: Model training")

    // 1. split the dataset : training and test by 0.9 : 0.1
    val Array(training, test) = df.randomSplit(Array(0.9, 0.1), seed = 42)

    // 2. train the model
    val rawModel = pipeline.fit(training)
    rawModel.write.overwrite().save("models/lr_simple")

    // 3. predictions on test dataset
    val dfWithSimplePredictions = rawModel.transform(test)

    //4. show the results
    dfWithSimplePredictions.groupBy("final_status", "predictions").count.show()

    //5. show the f1 score

    // Evaluator F1 Score
    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("final_status")
      .setPredictionCol("predictions")
      .setMetricName("f1")

    val f1Score = evaluator.evaluate(dfWithSimplePredictions)
    println("Logistic Regression F1 Score : " + f1Score)

    // Tuning two parameters:

    // lr.regParam : Parameter for regularization parameter (>= 0).
    // tf.minDF : minimum number (or fraction if < 1.0) of documents
    // where a term must appear in to be included in the vocabulary

    // lr.threshold : cutoff value
    // If the probability is greater than this threshold value,
    // the event is predicted to happen otherwise it is predicted not to happen.

    println("Step 4: Tuning")

    // set parameters' grid
    val paramGrid = new ParamGridBuilder()
      .addGrid(lr.regParam, Array(10e-8, 10e-6, 10e-4, 10e-2))
      .addGrid(tf.minDF, Array(55.0, 75.0, 95.0))
      .build()

    // model selection using trainValidationSplit which splits data once for each parameter point
    val trainValidationSplit = new TrainValidationSplit()
      .setEstimator(pipeline)
      .setEvaluator(evaluator)
      .setEstimatorParamMaps(paramGrid)
      .setTrainRatio(0.7)
      .setParallelism(2)

    // train the model with grid search
    val model = trainValidationSplit.fit(training)
    model.write.overwrite().save("models/lr_model_grid_search")

    // predictions on test dataset
    val dfWithPredictions = model.transform(test)

    //4. show the results
    dfWithPredictions.groupBy("final_status", "predictions").count.show()

    val f1Score_tune = evaluator.evaluate(dfWithPredictions)
    println("Logistic Regression F1 Score after tuning : " + f1Score_tune)

    dfWithPredictions.write.parquet("results/Trainer")

  }
}
