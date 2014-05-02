# Playing with Mahout's Spark Shell

## Intro

We'll use an excerpt of a publicly available [dataset about cereals](http://lib.stat.cmu.edu/DASL/Datafiles/Cereals.html). The dataset tells the protein, fat, carbohydrate and sugars (in milligrams) contained in a set of cereals, as well as a customer rating for the cereals. Our aim for this example is to fit a linear model which infers the customer rating from the ingredients.


Name                    | protein | fat | carbo | sugars | rating
:-----------------------|:--------|:----|:------|:-------|:---------
Apple Cinnamon Cheerios | 2       | 2   | 10.5  | 10     | 29.509541
Cap'n'Crunch            | 1       | 2   | 12    | 12     | 18.042851  
Cocoa Puffs             | 1       | 1   | 12    | 13     | 22.736446
Froot Loops             | 2       |	1   | 11    | 13     | 32.207582  
Honey Graham Ohs        | 1       |	2   | 12    | 11     | 21.871292
Wheaties Honey Gold     | 2       | 1   | 16    |  8     | 36.187559  
Cheerios                | 6       |	2   | 17    |  1     | 50.764999
Clusters                | 3       |	2   | 13    |  7     | 40.400208
Great Grains Pecan      | 3       | 3   | 13    |  4     | 45.811716  


## Installing Mahout & Spark on your local machine

We describe how to do a quick toy setup of Spark & Mahout on your local machine, so that you can run this example and play with the shell.

 1. Download [Apache Spark 0.9.1](http://d3kbcqa49mib13.cloudfront.net/spark-0.9.1.tgz) and unpack the archive file
 1. Change to the directory where you unpacked Spark and type ```sbt/sbt assembly``` to build it
 1. Create a directory for Mahout somewhere on your machine, change to there and checkout the current trunk of Apache Mahout from SVN ```svn co https://svn.apache.org/repos/asf/mahout/trunk/ mahout```
 1. Change to the ```mahout``` directory and build mahout using ```mvn -DskipTests clean install```
 
## Starting Mahout's Spark shell

 1. Goto the directory where you unpacked Spark and type ```sbin/start-all.sh``` to locally start Spark
 1. Open a browser, point it to [http://localhost:8080/](http://localhost:8080/) to check whether Spark successfully started. Copy the url of the spark master at the top of the page (it starts with **spark://**)
 1. Define the following environment variables: 
```
export MAHOUT_HOME=<directory into which you checked out Mahout>
export SPARK_HOME=<directory where you unpacked Spark>
export MASTER=<url of the Spark master>
```
 1. Finally, change to the directory where you unpacked Mahout and type ```bin/mahout spark-shell```, you should see the shell starting and get the prompt ```mahout> ```

## Implementation

We'll use the shell to interactively play with the data and incrementally implement our linear regression algorithm. Let's first load the dataset. Usually, we wouldn't need Mahout unless we processed a large dataset stored in a distributed filesystem. But for the sake of this example, we'll use our tiny and "pretend" it was too big to fit onto a single machine.

Mahout's linear algebra DSL has an abstraction called *DistributedRowMatrix (DRM)* which models a matrix that is partitioned by rows and stored in the memory of a cluster of machines. We use ```dense()``` to create a dense in-core matrix from our toy dataset and use ```drmParallelize``` to load it into the cluster, "mimicking" a large, partitioned dataset.



```
val drmData = drmParallelize(dense(
  (2, 2, 10.5, 10, 29.509541),  // Apple Cinnamon Cheerios
  (1, 2, 12,   12, 18.042851),  // Cap'n'Crunch
  (1, 1, 12,   13, 22.736446),  // Cocoa Puffs
  (2, 1, 11,   13, 32.207582),  // Froot Loops
  (1, 2, 12,   11, 21.871292),  // Honey Graham Ohs
  (2, 1, 16,   8,  36.187559),  // Wheaties Honey Gold
  (6, 2, 17,   1,  50.764999),  // Cheerios
  (3, 2, 13,   7,  40.400208),  // Clusters
  (3, 3, 13,   4,  45.811716)), // Great Grains Pecan
  numPartitions = 2);
```

*y = Xβ + ε*

```
val drmX = drmData(::, 0 until 4)
```

```
val y = drmData.collect(::, 4)
```

*β_hat = (X<sup>T</sup>X)<sup>-1</sup> X<sup>T</sup>y*

X<sup>T</sup>X

```
val drmXtX = drmX.t %*% drmX
```

X<sup>T</sup>y
```
val drmXty = drmX.t %*% y
```

```
val XtX = drmXtX.collect
val Xty = drmXty.collect(::, 0)

val betaHat = solve(XtX, Xty)
```

*Xβ_hat*
```
val yFitted = (drmX %*% betaHat).collect(::, 0)
(y - yFitted).norm(2)
```

```
import org.apache.mahout.math.Vector
def ols(drmX: DrmLike[_], y: Vector) = {
  val XtX = (drmX.t %*% drmX).collect
  val Xty = (drmX.t %*% y).collect(::, 0)
  solve(XtX, Xty)
}

def goodnessOfFit(drmX: DrmLike[Int], beta: Vector, y: Vector) = {
  val fittedY = (drmX %*% beta).collect(::, 0)
  (y - fittedY).norm(2)
}
```

```
val drmXwithBiasColumn = drmX.mapBlock(ncol = drmX.ncol + 1) {
  case(keys, block) =>
    val blockWithBiasColumn = block.like(block.nrow, block.ncol + 1)
    blockWithBiasColumn(::, 0 until block.ncol) := block
    blockWithBiasColumn(::, block.ncol) := 1

    keys -> blockWithBiasColumn
}
```

```
val betaWithBiasTerm = ols(drmXwithBiasColumn, y)
goodnessOfFit(drmXwithBiasColumn, betaWithBiasTerm, y)
```


