from pyspark.sql.functions import length,lower,col
import pyspark.sql.types as T
import pyspark.sql.functions as F
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType,ArrayType
from pyspark.ml.feature import StringIndexer,Tokenizer,StopWordsRemover,HashingTF,VectorAssembler
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem import WordNetLemmatizer
from pyspark.ml import Pipeline
import numpy as np
import re
import ast



def f3(x):
	x=ast.literal_eval(str(x))
	res=[n.strip() for n in x]
	return res
	
ra=F.udf(f3,T.ArrayType(T.StringType()))


url  =  r"((http://)[^ ]*|(https://)[^ ]*|(www\.)[^ ]*)"
up = '@[^\s]+'
hashtagPattern=  r'\B#\S+'
apattern = "[^a-zA-Z0-9]"
seqpattern = r"(.)\1\1+"
replacepattern = r"\1\1"



remove_url =lambda  s:re.sub(url,'URL',s) #replace all URLs with URL
remove_handler = lambda s :re.sub(up,'',s) #Replace all @USERNAME to USER
remove_hashtag =lambda  s:re.sub(hashtagPattern,'',s)  #remove  hashtags
neat_sequence  =lambda s :re.sub(seqpattern ,replacepattern,s) #Replace 3 or more consecutive letters by 2 letters
remove_extra_spaces=lambda s:re.sub(r'\s+',' ',s,flags=re.I) #Substituting multiple spaces with single space
remove_qoutes =lambda s:re.sub(r'"','',s) #remove quotation marks
neat_alpha =lambda  x:'  '.join(re.findall(r'\w+',x)) #remove all special characters

def  f1(df):
	df1=df.withColumn('cleaned_message',lower(col('message'))).select('spam','cleaned_message','subject')
	df1=df1.withColumn('cleaned_message',udf(remove_url,StringType( ))('cleaned_message')).select('spam','cleaned_message','subject')
	df1=df1.withColumn('cleaned_message',udf(remove_handler,StringType( ))('cleaned_message')).select('spam','cleaned_message','subject')
	df1=df1.withColumn('cleaned_message',udf(remove_hashtag,StringType( ))('cleaned_message')).select('spam','cleaned_message','subject')
	df1=df1.withColumn('cleaned_message',udf(neat_sequence,StringType( ))('cleaned_message')).select('spam','cleaned_message','subject')
	df1=df1.withColumn('cleaned_message',udf(remove_qoutes,StringType( ))('cleaned_message')).select('spam','cleaned_message','subject')
	df1=df1.withColumn('cleaned_message',udf(neat_alpha,StringType( ))('cleaned_message')).select('spam','cleaned_message','subject')
	df1=df1.withColumn('cleaned_message',udf(remove_extra_spaces,StringType( ))('cleaned_message')).select('spam','cleaned_message','subject')
	df1=df1.withColumn('cleaned_subject',lower(col('subject'))).select('cleaned_message','cleaned_subject','spam')
	df1=df1.withColumn('cleaned_subject',udf(remove_url,StringType( ))('cleaned_subject')).select('cleaned_message','cleaned_subject','spam')
	df1=df1.withColumn('cleaned_subject',udf(remove_handler,StringType( ))('cleaned_subject')).select('cleaned_message','cleaned_subject','spam')
	df1=df1.withColumn('cleaned_subject',udf(remove_hashtag,StringType( ))('cleaned_subject')).select('cleaned_message','cleaned_subject','spam')
	df1=df1.withColumn('cleaned_subject',udf(neat_sequence,StringType( ))('cleaned_subject')).select('cleaned_message','cleaned_subject','spam')
	df1=df1.withColumn('cleaned_subject',udf(remove_qoutes,StringType( ))('cleaned_subject')).select('cleaned_message','cleaned_subject','spam')
	df1=df1.withColumn('cleaned_subject',udf(neat_alpha,StringType( ))('cleaned_subject')).select('cleaned_message','cleaned_subject','spam')
	df1=df1.withColumn('cleaned_subject',udf(remove_extra_spaces,StringType( ))('cleaned_subject')).select('cleaned_message','cleaned_subject','spam')
	return df1
	

	
def f2(df):
	indexer = StringIndexer(inputCol="spam", outputCol="label")
	tokenizer1 = Tokenizer(inputCol="cleaned_message", outputCol="token_message")
	stopremove1 = StopWordsRemover(inputCol='token_message',outputCol='stop_tokens1')
	tokenizer2 = Tokenizer(inputCol="cleaned_subject", outputCol="token_subject")
	stopremove2 = StopWordsRemover(inputCol='token_subject',outputCol='stop_tokens2')
	data_prep_pipe = Pipeline(stages=[indexer,tokenizer1,tokenizer2,stopremove1,stopremove2])
	cleaner = data_prep_pipe.fit(df)
	clean_data = cleaner.transform(df)
	return clean_data

def lemma(df):
	wnl=WordNetLemmatizer()
	lemmatizer=udf(lambda tokens: [wnl.lemmatize(token) for token in tokens],StringType())
	df=df.withColumn("lemmatized1",lemmatizer("stop_tokens1"))
	df=df.withColumn("lemmatized2",lemmatizer("stop_tokens2"))
	
	return df
	
def text_preprocess(df):
		df1=lemma(f2(df))
		df2=df1.withColumn("lemmatized1",ra(F.col("lemmatized1")))
		df2=df1.withColumn("lemmatized2",ra(F.col("lemmatized2")))
			
		hashingTF1 = HashingTF(inputCol="lemmatized1", outputCol="feature1", numFeatures=2**17)
		hashingTF2 = HashingTF(inputCol="lemmatized2", outputCol="feature2", numFeatures=2**17)
		assembler=VectorAssembler(inputCols=['feature1','feature2'],outputCol="combined_F")
		data_prep_pipe = Pipeline(stages=[hashingTF1,hashingTF2,assembler])
		cleaner = data_prep_pipe.fit(df)
		clean_data = cleaner.transform(df)
	
		return clean_data
	
