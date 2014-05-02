
# Distributed Linear Regression with Mahout's Spark Shell

## Intro

We'll use an excerpt of a publicly available [dataset about cereals](http://lib.stat.cmu.edu/DASL/Datafiles/Cereals.html). The dataset tells the protein, fat, carbohydrate and sugars (in milligrams) contained in a set of cereals, as well as a customer rating for the cereals. Our aim for this example is to fit a linear model which predicts the customer rating from the ingredients.


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
 1. Create a directory for Mahout somewhere on your machine, change to it and checkout the current trunk of Apache Mahout from SVN ```svn co https://svn.apache.org/repos/asf/mahout/trunk/ mahout```
 1. Change to the ```mahout``` directory and build mahout using ```mvn -DskipTests clean install```
 
## Starting Mahout's Spark shell

 1. Goto the directory where you unpacked Spark and type ```sbin/start-all.sh``` to locally start Spark
 1. Open a browser, point it to http://localhost:8080/ to check whether Spark successfully started. Copy the url of the spark master at the top of the page (it starts with **spark://**)
 1. Define the following environment variables: 
```
export MAHOUT_HOME=<directory where you checked out Mahout>
export SPARK_HOME=<directory where you unpacked Spark>
export MASTER=<url of the Spark master>```
 1. Finally, change to the directory where you unpacked Mahout and type ```bin/mahout spark-shell```

## Implementation

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

