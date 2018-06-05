
import org.apache.spark.SparkContext
import org.apache.spark.mllib.evaluation.{RankingMetrics, RegressionMetrics}
import org.apache.spark.mllib.recommendation.{ALS, Rating}
import org.jblas.DoubleMatrix

object SparkApp {
  def main(args: Array[String]) {
    val sc = new SparkContext("local[2]", "First Spark App")
    //     ./bin/spark-shell --driver-memory 4g
    val PATH = "/Volumes/work/sparkProject"
    val rawData = sc.textFile(PATH + "/ml-100k/u.data") // 用户id 影片id 星级 时间戳
    rawData.first()
    val rawRatings = rawData.map(_.split("\t").take(3))
    val ratings = rawRatings.map { case Array(user, movie, rating) =>
      Rating(user.toInt, movie.toInt, rating.toDouble)
    }
    ratings.first()

    val model = ALS.train(ratings, 50, 10, 0.01)
    model.userFeatures
    model.userFeatures.count
    model.productFeatures.count

    val predictedRating = model.predict(789, 123) // 该推荐模型下 指定用户对指定电影的预测评级
    val userId = 789
    val K = 10
    val topKRecs = model.recommendProducts(userId, K) // 该推荐模型下 对指定用户预测其评级最高的前K项 即 最佳推荐项
    println(topKRecs.mkString("\n"))
    // model = ALS.trainImplicit(ratings, 50, 10, 0.01, 0.01)

    // 检验推荐效果
    val movies = sc.textFile(PATH + "/ml-100k/u.item")
    val titles = movies.map(line => line.split("\\|").take(2)).map(array
    => (array(0).toInt, array(1))).collectAsMap()
    titles(123)

    val moviesForUser = ratings.keyBy(_.user).lookup(789)
    println(moviesForUser.size)
    moviesForUser.sortBy(-_.rating).take(10).map(rating => (titles(rating.product),
      rating.rating)).foreach(println)

    topKRecs.map(rating => (titles(rating.product), rating.rating)).foreach(println)

    // 物品推荐 相似度衡量法：
    // 1、皮尔森相关系数 pearson correlation
    // 2、针对实数向量的余弦相似度 cosine similarity
    // 3、针对二元向量的杰卡德相似系数 jaccard similarity
    // val aMatrix = new DoubleMatrix(Array(1.0, 2.0, 3.0))
    val itemId = 567
    val itemFactor = model.productFeatures.lookup(itemId).head
    val itemVector = new DoubleMatrix(itemFactor)
    cosineSimilarity(itemVector, itemVector)

    val sims = model.productFeatures.map { case (id, factor) =>
      val factorVector = new DoubleMatrix(factor)
      val sim = cosineSimilarity(factorVector, itemVector)
      (id, sim)
    }
    val sortedSims = sims.top(K)(Ordering.by[(Int, Double), Double] { case (id, similarity) => similarity })
    println(sortedSims.take(K).mkString("\n"))
    println(titles(itemId))

    // 去除本物品 结果中补充物品名称 title
    val sortedSims2 = sims.top(K + 1)(Ordering.by[(Int, Double), Double] { case (id, similarity) => similarity })
    sortedSims2.slice(1, 11).map{ case (id, sim) => (titles(id), sim)}.mkString("\n")




    // 量化推荐模型效果
    // 均方差 直接衡量 "用户-物品" 评级矩阵的重建误差。是一些模型里所采用的最小化目标函数，特别是许多矩阵类分解类方法，比如 AL。常用于显式评级的情形
    val actualRating = moviesForUser.take(1)(0)
    val predictedRating2 = model.predict(789, actualRating.product)
    val squaredError = math.pow(predictedRating2 - actualRating.rating, 2.0)

    val usersProducts = ratings.map{ case Rating(user, product, rating) => (user, product) }
    val predictions = model.predict(usersProducts).map{
      case Rating(user, product, rating) => ((user, product), rating)
    }
    val ratingsAndPredictions = ratings.map{
      case Rating(user, product, rating) => ((user, product), rating)
    }.join(predictions)
    val MSE = ratingsAndPredictions.map{
      case ((user, product), (actual, predicted)) => math.pow((actual - predicted), 2)
    }.reduce(_ + _) / ratingsAndPredictions.count
    println("Mean Squared Error = " + MSE)
    val RMSE = math.sqrt(MSE)
    println("Root Mean Squared Error = " + RMSE)

    val predictedAndTrue = ratingsAndPredictions.map{ case ((user, product), (predicted, actual)) => (predicted, actual)}
    val regressionMetrics = new RegressionMetrics(predictedAndTrue)
    println("Mean Squared Error = " + regressionMetrics.meanSquaredError)
    println("Root Mean Squared Error = " + regressionMetrics.rootMeanSquaredError)


    // MAPK 整个数据集上的K值平均准确率(Average Precision at K metric, APK)
    val actualMovices = moviesForUser.map(_.product)
    val predictedMovices = topKRecs.map(_.product)
    val apk10 = avgPrecisionK(actualMovices, predictedMovices, 10)

    val itemFactors = model.productFeatures.map{ case (id, factor) => factor }.collect()
    val itemMatrix = new DoubleMatrix(itemFactors)
    println(itemMatrix.rows, itemMatrix.columns)
    val imBroadcast = sc.broadcast(itemMatrix)
    val allRecs = model.userFeatures.map{ case (userId, array) =>
        val userVector = new DoubleMatrix(array)
        val scores = imBroadcast.value.mmul(userVector)
        val sortedWithId = scores.data.zipWithIndex.sortBy(-_._1)
        val recommendedIds = sortedWithId.map(_._2 + 1).toSeq
      (userId, recommendedIds)
    }
    val userMovies = ratings.map{ case Rating(user, product, rating) => (user, product) }.groupBy(_._1)
    var K2 = 10
    val MAPK = allRecs.join(userMovies).map{ case (useId, (predicted, actualWithIds)) =>
      val actual = actualWithIds.map(_._2).toSeq
        avgPrecisionK(actual, predicted, K2)
    }.reduce(_ + _) / allRecs.count
    println("Mean Average Precision at K = " + MAPK)

    val predictedAndTrueForRanking = allRecs.join(userMovies).map{ case (userId, (predicted, actualWithIds)) =>
        val actual = actualWithIds.map(_._2)
      (predicted.toArray, actual.toArray)
    }
    val rankingMetrics = new RankingMetrics(predictedAndTrueForRanking)
    println("Mean Average Precision = " + rankingMetrics.meanAveragePrecision)
    K2 = 2000
    val MAPK2000 = allRecs.join(userMovies).map{ case (useId, (predicted, actualWithIds)) =>
      val actual = actualWithIds.map(_._2).toSeq
      avgPrecisionK(actual, predicted, K2)
    }.reduce(_ + _) / allRecs.count
    println("Mean Average Precision at K 2000 = " + MAPK2000)
  }

  def cosineSimilarity(vec1: DoubleMatrix, vec2: DoubleMatrix): Double = {
    vec1.dot(vec2) / (vec1.norm2() * vec2.norm2())
  }

  def avgPrecisionK(actual: Seq[Int], predicted: Seq[Int], K: Int): Double = {
    val predK = predicted.take(K)
    var score = 0.0
    var numHits = 0.0
    for ((p, i) <- predK.zipWithIndex) {
      if (actual.contains(p)) {
        numHits += 1.0
        score += numHits / (i.toDouble + 1.0)
      }
    }
    if (actual.isEmpty) {
      1.0
    } else {
      score / scala.math.min(actual.size, K).toDouble
    }
  }
}