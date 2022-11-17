from pyspark import SparkContext
from pyspark.streaming import StreamingContext
from pyspark import SparkContext
from pyspark.sql.session import SparkSession
from pyspark.sql.functions import col
import re
import f2,model,test


sc=SparkContext.getOrCreate()
ssc=StreamingContext(sc,1)
spark=SparkSession(sc)
sc.setLogLevel("OFF")

try:
	record=ssc.socketTextStream('localhost',6100)
except Exception as e:
	print(e)

def readstream(rdd):
	if(not rdd.isEmpty()):
		try:
			df=spark.read.json(rdd)
		except Exception as e:
			print(e)
		newcols = [col(column).alias(re.sub('\s*', '', column)) \
		for column in df.columns]
	
		
		try:
			df1=f2.text_preprocess(df)
			#df4.select("combined_F","label").show(truncate=True)
		except Exception as e:
			print(e)
			
		try:
			x,y,a,b=model.split(df4)
			model.f1(x,y,a,b)
			#model.cluster(x,y,a,b)
		except Exception as e:
			print(e)
			
		'''try:
			test.f1(df4)
		except Exception as e:
			print(e)'''
		
		
		
		
record.foreachRDD(lambda x:readstream(x))


		
	

ssc.start()             
ssc.awaitTermination()  
