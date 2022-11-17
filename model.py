import numpy as np
import pickle
import ast
from operator import attrgetter
from scipy.sparse import vstack,csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier,PassiveAggressiveClassifier,Perceptron
from sklearn.cluster import MiniBatchKMeans

clf=MultinomialNB(alpha=0.01)
clf2=SGDClassifier(max_iter=5)
clf3=PassiveAggressiveClassifier()
clf4=Perceptron()
clf5=MiniBatchKMeans()


def as_mat(vec):
 	data,ind=vec.values,vec.indices
 	shape=1,vec.size
 	return csr_matrix((data,ind,np.array([0,vec.values.size])),shape)
 	
def split(df):
	X=df.select('combined_F')
	y=df.select('label')
	y=np.array(y.collect()).T
	y=np.array(y[0])
	
	feat=X.rdd.map(attrgetter("combined_F"))
	mats=feat.map(as_mat)
	mat=mats.reduce(lambda x,y:vstack([x,y]))
	mat=np.array(mat.todense())
	
	
	X_train,X_test,y_train,y_test=train_test_split(mat,y,test_size=0.1,random_state=42)
	
	return X_train,y_train,X_test,y_test
	



def f1(X_train,y_train,X_test,y_test):

	clf.partial_fit(X_train, y_train, classes=[0.0,1.0])
	clf2.partial_fit(X_train,y_train,classes=[0.0,1.0])
	clf3.partial_fit(X_train,y_train,classes=[0.0,1.0])
	clf4.partial_fit(X_train,y_train,classes=[0.0,1.0])
	a=clf.predict(X_test)
	b=clf2.predict(X_test)
	c=clf3.predict(X_test)
	d=clf4.predict(X_test)
	print("Report MultinomialNB\n:",classification_report(a,y_test,labels=[0.0,1.0]))
	print("Report SGDC\n:",classification_report(b,y_test,labels=[0.0,1.0]))
	print("Report PAC\n:",classification_report(c,y_test,labels=[0.0,1.0]))
	print("Report Perceptron\n:",classification_report(d,y_test,labels=[0.0,1.0]))
	'''print("accuracy using MultinomialNB is:",accuracy_score(a,y_test))
	print("accuracy using SGDClassifier is:",accuracy_score(b,y_test))
	print("accuracy using PassiveAggressiveClassifier is:",accuracy_score(c,y_test))
	print("accuracy using Perceptron() is:",accuracy_score(d,y_test))'''
	
	try:
		with open('MNB','wb') as file:
			pickle.dump(clf,file)
		with open('SGDC.pkl','wb') as file:
			pickle.dump(clf2,file)
		with open('SGDC.pkl','wb') as file:
			pickle.dump(clf2,file)
		with open('PAC.pkl','wb') as file:
			pickle.dump(clf3,file)
		with open('percep.pkl','wb') as file:
			pickle.dump(clf4,file)
			
		#print("DONE!")
		
	except Exception as e:
		print(e)
		
	return X_test,y_test
	
	
def cluster(X_train,y_train,X_test,y_test):
	
	clf5.partial_fit(X_train,y_train)
	y_pred=clf5.predict(X_test)
	print("Accuracy using clustering: ",accuracy_score(y_pred,y_test))
		
	
	
	
	
		
	
	
	
	
	
	
	

