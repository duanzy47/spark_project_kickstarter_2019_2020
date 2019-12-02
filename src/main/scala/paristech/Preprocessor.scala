package paristech

import org.apache.spark.SparkConf
import org.apache.spark.sql.{Column, DataFrame, Dataset, Row, SparkSession}
import org.apache.spark.sql.functions.{concat, datediff, from_unixtime, lit, lower, round, to_date, udf}

object Preprocessor {

  def main(args: Array[String]): Unit = {

    // Des réglages optionnels du job spark. Les réglages par défaut fonctionnent très bien pour ce TP.
    // On vous donne un exemple de setting quand même
    val conf = new SparkConf().setAll(Map(
      "spark.scheduler.mode" -> "FIFO",
      "spark.speculation" -> "false",
      "spark.reducer.maxSizeInFlight" -> "48m",
      "spark.serializer" -> "org.apache.spark.serializer.KryoSerializer",
      "spark.kryoserializer.buffer.max" -> "1g",
      "spark.shuffle.file.buffer" -> "32k",
      "spark.default.parallelism" -> "12",
      "spark.sql.shuffle.partitions" -> "12"
    ))

    // Initialisation du SparkSession qui est le point d'entrée vers Spark SQL (donne accès aux dataframes, aux RDD,
    // création de tables temporaires, etc., et donc aux mécanismes de distribution des calculs)
    val spark = SparkSession
      .builder
      .config(conf)
      .appName("TP Spark : Preprocessor")
      .getOrCreate()
    // It should be imported after the SparkSession is created.
    import spark.implicits._
    /*******************************************************************************
      *
      *       TP 2
      *
      *       - Charger un fichier csv dans un dataFrame
      *       - Pre-processing: cleaning, filters, feature engineering => filter, select, drop, na.fill, join, udf, distinct, count, describe, collect
      *       - Sauver le dataframe au format parquet
      *
      *       if problems with unimported modules => sbt plugins update
      *
      ********************************************************************************/
    val df: DataFrame = spark
        .read
        .option("header", true)
        .option("inferSchema", true)
        .csv("datasets/original/train_clean.csv")

    println(s"Nombre de lignes : ${df.count}")
    println(s"Nombre de colonnes : ${df.columns.length}")

    df.show()

    val dfCasted: DataFrame = df
      .withColumn("goal", $"goal".cast("Int"))
      .withColumn("deadline" , $"deadline".cast("Int"))
      .withColumn("state_changed_at", $"state_changed_at".cast("Int"))
      .withColumn("created_at", $"created_at".cast("Int"))
      .withColumn("launched_at", $"launched_at".cast("Int"))
      .withColumn("backers_count", $"backers_count".cast("Int"))
      .withColumn("final_status", $"final_status".cast("Int"))

    dfCasted.printSchema()

    dfCasted
      .select("goal", "backers_count", "final_status")
      .describe()
      .show

    dfCasted.groupBy("disable_communication").count.orderBy($"count".desc).show(100)
    dfCasted.groupBy("country").count.orderBy($"count".desc).show(100)
    dfCasted.groupBy("currency").count.orderBy($"count".desc).show(100)
    dfCasted.select("deadline").dropDuplicates.show()
    dfCasted.groupBy("state_changed_at").count.orderBy($"count".desc).show(100)
    dfCasted.groupBy("backers_count").count.orderBy($"count".desc).show(100)
    dfCasted.select("goal", "final_status").show(30)
    dfCasted.groupBy("country", "currency").count.orderBy($"count".desc).show(50)

    // Only very very few rows have "disable_communication" values as True
    val df2: DataFrame = dfCasted.drop("disable_communication")

    // Delete the data we can't obtain while predicting
    val dfNoFutur: DataFrame = df2.drop("backers_count", "state_changed_at")

    // It seems that some rows when 'country = False', the 'country' values are in 'currency', we count the numbers
    // In spark SQL "===" which returns a column should be used instead of "==" returning a boolean
    df.filter($"country" === "False")
      .groupBy("currency")
      .count
      .orderBy($"count".desc)
      .show(50)

    // UDF clean the country column: when country has no value (False), fill it with the value in currency.
    def cleanCountry(country: String, currency: String): String = {
      if (country == "False")
        currency
      else
        country
    }
    // UDF clean the currency column: when its value not 3 characters, fill it with null; otherwise keep its origin.
    def cleanCurrency(currency: String): String = {
      if (currency != null && currency.length != 3)
        null
      else
        currency
    }
    // Scala is functional programming language, which means we can pass a function to a function.
    // In the example below, cleanCountry is passed to udf with a anonymous function's parameter '_' which represents
    // the String
    val cleanCountryUdf = udf(cleanCountry _)
    val cleanCurrencyUdf = udf(cleanCurrency _)

    val dfCountry: DataFrame = dfNoFutur
      .withColumn("country2", cleanCountryUdf($"country", $"currency"))
      .withColumn("currency2", cleanCurrencyUdf($"currency"))
      .drop("country", "currency")

    // Find out how many different values in final_status
    dfCountry.groupBy("final_status").count().show(100)

    // Two strategies to process rows with final_status != 0 or 1

    // Strategy 1 : delete these rows
    val dfFinalStatus1: DataFrame = dfCountry.filter($"final_status".isin(0,1))
    //dfFinalStatus1.groupBy("final_status").count().show(100)

    // Strategy 2 : replace them with 0
    def cleanFinal_status(final_status : Integer): Integer = {
      if (final_status == 0 || final_status == 1)
        final_status
      else
        0
    }
    val cleanFinal_statusUDF = udf(cleanFinal_status _)
    val dfFinalStatus2: DataFrame = dfCountry
      .withColumn("final_status2", cleanFinal_statusUDF($"final_status"))
      .drop("final_status")

    // Add a new feature days_campaign = days between launched_at and deadline
    // Add a new feature hours_prepa = hours between launched_at and created_at
    // Add a new feature text who concats (name, desc, and keywords) in lowercase
    val dfAddManip: DataFrame = dfFinalStatus2
      .withColumn("days_campaign", datediff(to_date(from_unixtime($"deadline"))
        ,to_date(from_unixtime($"launched_at"))))
      .withColumn("hours_prepa", round(($"launched_at"-$"created_at")/3600))
      .withColumn("text", concat(lower($"name"),lit(" "),lower($"desc")
        , lit(" "), lower($"keywords")))
      .drop("launched_at","created_at","deadline")

    // Remplacez les valeurs nulles des colonnes days_campaign, hours_prepa, et goal par la valeur -1 et
    // par "unknown" pour les colonnes country2 et currency2.

    def cleanNuls1(num : Integer): Integer = {
      if (num == null)
        -1
      else
        num
    }
    val cleanNulsUDF1 = udf(cleanNuls1 _)

    def cleanNuls2(str : String): String = {
      if (str == null)
        "unknown"
      else
        str
    }
    val cleanNulsUDF2 = udf(cleanNuls2 _)

    val dfValueNul:DataFrame = dfAddManip
      .withColumn("days_campaign2", cleanNulsUDF1($"days_campaign"))
      .withColumn("hours_prepa2", cleanNulsUDF1($"hours_prepa"))
      .withColumn("goal2", cleanNulsUDF1($"goal"))
      .withColumn("country3", cleanNulsUDF1($"country2"))
      .withColumn("currency3", cleanNulsUDF1($"currency2"))

      dfValueNul.write.parquet("results/Preprocessor")
  }
}
