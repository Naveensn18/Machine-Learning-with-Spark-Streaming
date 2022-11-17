from sklearn.metrics import accuracy_score,classification_report
import pickle
import model




def f1(df):
	X_test,y_test=model.f1(df)
	m=pickle.load(open('MNB','rb'))
	y_pred=m.predict(X_test)
	print(accuracy_score(y_test,y_pred))
	m1=pickle.load(open('PAC.pkl','rb'))
	y_pred1=m1.predict(X_test)
	print("PAC: ",accuracy_score(y_test,y_pred1))
	m2=pickle.load(open('percep.pkl','rb'))
	y_pred2=m2.predict(X_test)
	print("percep: ",accuracy_score(y_test,y_pred2))
	m3=pickle.load(open('SGDC.pkl','rb'))
	y_pred3=m3.predict(X_test)
	print("SGDC: ",accuracy_score(y_test,y_pred3))
	
	
