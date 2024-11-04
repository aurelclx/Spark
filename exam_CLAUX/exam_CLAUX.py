
import  urllib.request 
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, isnan, isnull, regexp_replace, lower, length, to_date
from pyspark.sql.types import BooleanType
import numpy as np


def download_file(filename):
    r = urllib.request.urlretrieve('https://assets-datascientest.s3.eu-west-1.amazonaws.com/' + filename, filename)


# En utilisant la fonction urlretrieve du module urllib.request, écrire une fonction download_file permettant de télécharger un fichier filename depuis l'adresse globale précédente. Appliquer cette fonction aux fichiers que nous voulons télécharger.
download_file("gps_app.csv")
download_file("gps_user.csv")

####################################################
# CSV to RDD

spark = SparkSession\
        .builder\
        .appName("Cours de Spark")\
        .master("local[*]")\
        .getOrCreate()
sc = spark.sparkContext


raw_app = spark.read.option("header", True)\
                    .option("inferSchema", True)\
                    .option("escape", "\"")\
                    .csv("gps_app.csv")

raw_user = spark.read.option("header", True)\
                     .option("inferSchema", True)\
                     .option("escape", "\"")\
                     .csv("gps_user.csv")


for column in raw_app.columns:
    raw_app = raw_app.withColumnRenamed(column , column.lower().replace(" ", "_"))

for column in raw_user.columns:
    raw_user = raw_user.withColumnRenamed(column , column.lower().replace(" ", "_"))

# Q.2. Dans un premier prétraitement, renommer toutes les colonnes en remplaçant les espaces par des soulignements et les majuscules par des minuscules.
# je cast en integer Reviews qui apparait en String sur le Schema et ca va me permettre une analyse réelle avec le 
raw_app = raw_app.withColumn("reviews", raw_app["reviews"].cast('int'))


# Q.3.1 Remplacer les valeurs manquantes dans la colonne rating par la moyenne ou la médiane. Justifier le choix.
# Le maximum étant bien plus élevé que la moyenne il vaut mieux choisir la médiane, 

reviews_median = raw_app.approxQuantile("rating", [0.5],0.01)[0]
raw_app = raw_app.fillna({"rating" : reviews_median})

 

# Q.3.2 Remplacer la valeur manquante de la colonne type par la valeur la plus logique. Justifier le choix.
# Je regarde l'attribut price de la même ligne pour vérifier s'il y un prix ou non et determiner la valeur de l'attribut type
raw_app.filter(isnan(col('type'))).show()
# il n'y a qu'une valeur, je peux effectuer le fill NA sur tout le dataframe sans problème.
raw_app = raw_app.fillna({"type" : "Free"})


# Q.3.3 Afficher les valeurs uniques prisent par la colonne type. Que remarquez-vous ? 
# Supprimer le problème. Cela réglera aussi la valeur manquante de la colonne content_rating.
# Fillna ne fonctionne pas ici bizarrement
raw_app = raw_app.withColumn('type', when(isnan(col(('type'))), "Free").otherwise(col("type")) )
# Pour supprimer la ligne ou l'insertation est décalé, je selectionne le dataframe ou la valeur de type n'est pas la valeur problématique, ici 0
raw_app = raw_app.filter(col("type") != "0")

# Q.3.4 Remplacer le reste des valeurs manquantes pour la colonne current_ver et 
# la colonne android_ver par leur modalité respective.

modes_current_ver = raw_app.groupby("current_ver").count().sort("count", ascending=False)
mode_ver = modes_current_ver.head()["current_ver"]
raw_app = raw_app.withColumn("current_ver", when(isnan("current_ver") | isnull("current_ver"), mode_ver).otherwise(col("current_ver")))

modes_current_ver = raw_app.groupby("android_ver").count().sort("count", ascending=False)
mode_ver = modes_current_ver.head()["android_ver"]
raw_app = raw_app.withColumn("android_ver", when(isnan("android_ver") | isnull("android_ver"), mode_ver).otherwise(col("android_ver")))


def getMissingValues(dataframe):
  count = dataframe.count()
  columns = dataframe.columns
  nan_count = []
  # we can't check for nan in a boolean type column
  for column in columns:
    if dataframe.schema[column].dataType == BooleanType():
      nan_count.append(0)
    else:
      nan_count.append(dataframe.where(isnan(col(column))).count())
  null_count = [dataframe.where(isnull(col(column))).count() for column in columns]
  return([count, columns, nan_count, null_count])

def missingTable(stats):
  count, columns, nan_count, null_count = stats
  count = str(count)
  nan_count = [str(element) for element in nan_count]
  null_count = [str(element) for element in null_count]
  max_init = np.max([len(str(count)), 10])
  line1 = "+" + max_init*"-" + "+"
  line2 = "|" + (max_init-len(count))*" " + count + "|"
  line3 = "|" + (max_init-9)*" " + "nan count|"
  line4 = "|" + (max_init-10)*" " + "null count|"
  for i in range(len(columns)):
    max_column = np.max([len(columns[i]),\
                        len(nan_count[i]),\
                        len(null_count[i])])
    line1 += max_column*"-" + "+"
    line2 += (max_column - len(columns[i]))*" " + columns[i] + "|"
    line3 += (max_column - len(nan_count[i]))*" " + nan_count[i] + "|"
    line4 += (max_column - len(null_count[i]))*" " + null_count[i] + "|"
  lines = f"{line1}\n{line2}\n{line1}\n{line3}\n{line4}\n{line1}"
  print(lines)

# il ne reste pas de valeurs manquantes 
missingTable(getMissingValues(raw_app))

# Q.4.1 Étudier les valeurs manquantes présents dans ce jeu de données. Les valeurs manquantes (nan) de chaque colonne sont-elles toutes sur les mêmes lignes ?
# On compte autant de Nan value avec la requête suivantes que sur le tableau affiché dans la question. Les valeurs manquantes sont sur les mêmes lignes
raw_user.filter(isnan(col("translated_review"))).filter(isnan(col("sentiment"))).filter(isnan(col("sentiment_polarity"))).filter(isnan(col("sentiment_subjectivity"))).count()

# Q.4.2 Nettoyer les valeurs manquantes.
raw_user = raw_user.filter(~isnan(col("translated_review"))).filter(~isnan(col("sentiment"))).filter(~isnan(col("sentiment_polarity"))).filter(~isnan(col("sentiment_subjectivity")))

# Q.4.3 Vérifier qu'il ne reste plus de valeurs manquantes grâce à l'application de la commande :
missingTable(getMissingValues(raw_user_no_na))


# Q.5.1 Vérifier si il reste des valeurs non numériques dans les colonnes sentiment_polarity et sentiment_subjectivity. Pour se faire on pourra filtrer les lignes pour lesquelles transformer la colonne en double renvoie une valeur manquante.
raw_user = raw_user.withColumn("sentiment_polarity", col("sentiment_polarity").cast("double")).withColumn("sentiment_subjectivity", col("sentiment_subjectivity").cast("double"))

# Pas de nouvelles valeurs manquantes après avoir transformé en double sentiment polarity et sentiment subjectivity
missingTable(getMissingValues(raw_user_double))


# Q.5.2 Convertir les colonnes numériques au format float.
raw_user = raw_user.withColumn("sentiment_polarity", col("sentiment_polarity").cast("float")).withColumn("sentiment_subjectivity", col("sentiment_subjectivity").cast("float"))



# Q.5.3 Remplacer les caractères spéciaux de la colonne translated_review par des espaces. Remplacer ensuite tous les espaces de taille supérieure à 2 par un espace de taille 1. Pour répondre à cette question on pourra utiliser la fonction regexp_replace de la collection pyspark.sql.functions.
# Espace en regex = \s & cractères speciaux = tout ce qui  n'est pas alpha numérique
raw_user = raw_user.withColumn("translated_review", regexp_replace(col("translated_review"), r"[^a-zA-Z0-9\s]", " ")).withColumn("translated_review", regexp_replace(col("translated_review"), r"\s{2,}", " "))
withColumn("translated_review", regexp_replace(col("translated_review"), r"[^a-zA-Z0-9\s]", " "))

# Q.5.4 Minimiser tous les caractères de la colonne translated_review.
raw_user = raw_user_double.withColumn("translated_review", lower(col("translated_review")))

# Q.5.5 Afficher le nombre de commentaires pour chacun des groupes de tailles allant de 1 caractère à 10 caractères.
raw_user.filter(( length(col("translated_review")) >= 1) & (length(col("translated_review")) <=10)).show()

# Q.5.6 Conserver uniquement les lignes dont le commentaire est de taille supérieure ou égale à 3.
raw_user = raw_user.filter( length(col("translated_review")) >= 3)

# Q.5.7 Calculer les 20 mots les plus présents pour les commentaires étant positifs. Pour se faire on pourra passer par l'attribut rdd du DataFrame puis extraire la colonne translated_review 
# je retire les valeurs NULL de mon DataFrame sinon je ne pourrais pas appliquer de replace, lower.
raw_user = raw_user.withColumn("translated_review", when(isnull(col("translated_review")), " ").otherwise(col("translated_review")))
translated_review_map = raw_user.\
                        filter(col("sentiment") == "Positive").rdd.\
                        flatMap(lambda x : x['translated_review'].replace('\'', ' ').lower().split()).\
                        filter(lambda word: len(word) > 1).\
                        map(lambda x : (x ,1))

translated_review_map_reduce = translated_review_map.\
                        reduceByKey(lambda a,b : a+b).\
                        sortBy(lambda x: x[1], False)
                        
translated_review_map_reduce.take(20)

# Q.6.1 Changer le type de la colonne reviews en integer en transformant les lignes problématiques si nécessaire.
# pas de lignes problématiques
raw_app = raw_app.withColumn("reviews", col("reviews").cast("integer"))


# Q.6.2 Nous allons maintenant convertir la colonne installs en integer aussi. Pour se fa va utiliser une regex assez similaire à celles utilisées précédemment afin
# de remplacer tous les caractères n'étant pas des chiffres par un vide. On pourra, avant de remplacer la colonne, s'assurer qu'il n'y a pas de valeurs nulle

raw_app = raw_app.withColumn("installs", regexp_replace(col("installs"), r"[^0-9\s]", " ")).withColumn("installs", regexp_replace(col("installs"), r"\s{1,}", "")).withColumn("installs", col("installs").cast("integer"))

# Q.6.3 Répéter le même type d'opération pour transformer la colonne price en double. Attention ici à bien traiter les nombres à virgule.

raw_app = raw_app.withColumn("price", regexp_replace(col("price"), r"[\$]", " ")).withColumn("price", col("price").cast("double"))


# Q.6.4 En partant du principe que la date de la colonne last_updated est au format MMMM d, yyyy, convertir cette colonne au format date avec la fonction to_date.
raw_app = raw_app.withColumn("last_updated", to_date("last_updated", "MMMM d, yyyy" ))