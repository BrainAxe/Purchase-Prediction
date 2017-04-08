from pyspark import SparkContext
from pyspark.mllib.stat import Statistics
import math
from pyspark.mllib.linalg import Vectors
from pyspark.sql import SQLContext
from pyspark.ml.regression import LinearRegression
import pandas as pd 
import numpy as np 
from datetime import datetime
import time 
import csv
from sklearn import metrics 
import matplotlib.pyplot as plt
from matplotlib import style


def roc_graph():
	style.use('ggplot')

	x,y = np.loadtxt('result4.csv', unpack = True, delimiter = ',')


	auc = np.trapz(y,x)

	plt.plot(x, y,color='darkorange',label='ROC curve (area = %0.2f)' % auc)
	plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('Receiver operating characteristic(ROC)')
	plt.legend(loc="lower right")
	plt.show()




def nsession():
	with open('tdata.csv') as csvfile:
	reader = csv.reader(csvfile)
	c = 0
	pre_id = ''
	for row in reader:
		u_id = row[0]
		
		if u_id != pre_id:
			c+=1
			pre_id = u_id
			

	return c




def check_result():
	s = []
	with open("result.csv") as csvfile:
		reader = csv.reader(csvfile)
		for row in reader:
			u_id = row[0]
			score = float(row[3])
			if score >= .05:
				if u_id not in s:
					print "{} {}".format(u_id,score)
					s.append(u_id)


	c = 0

	with open("solution-sort.csv") as rfile:
		reader = csv.reader(rfile)
		for row in reader:
			if row[0] in s:
				c+=1


	l = len(s)		


	n = nsession()
	tp = c
	fp = l-c
	fn = 3285-c
	tn = n - tp - fp - fn

	print "TP: ",c
	print "TN: ",tn 
	print "FP: ",l-c
	print "FN: ",3285-c


	fpr = fp/float(fp+tn)
	tpr = tp/float(tp+fn)
	precision = tp/float(tp+fp)
	recall = tp/float(tp+fn)
	accuracy = (tp+tn)/float(n) 

	print "FPR: ",fpr
	print "TPR: ",tpr 
	print "Precision: ",precision
	print "Recall: ",recall
	print "accuracy: ",accuracy






def calculate(r):

	with open('tdata.csv') as csvfile:
		reader = csv.reader(csvfile)
		with open('result.csv', 'w') as wfile:
			writer = csv.writer(wfile)
			pre_id = ''
			for row in reader:
				u_id = row[0]
				time = float(row[1])
				p_id = row[2]
				click = int(row[3])
				same = int(row[4])


				if u_id == pre_id:
					t_diff = time - s_time



					result = r[0] + float(r[1])*t_diff + float(r[2])*click + float(r[3])*same

					writer.writerow([u_id,t_diff,p_id,result])

				else:
					s_time = time 
					t_diff = 0
 

					result = r[0] + float(r[1])*t_diff + float(r[2])*click + float(r[3])*same

					writer.writerow([u_id,t_diff,p_id,result])
					pre_id = u_id





def ptest_data():
	df = pd.read_csv('test_split.csv')
	li = []
	with open('tdata.csv', 'w') as wfile:
		writer = csv.writer(wfile)
		pre_id = ''
		for idx, row in df.iterrows():
			u_id = row['user_id']
			time = datetime.strptime(row['timestamp'],'%Y-%m-%dT%H:%M:%S.%fZ')
			time = float(time.strftime("%s.%f"))
			p_id = row['item_id']
			


			if u_id == pre_id:
				t_diff = time - s_time
				c1 = 0
				li.append(p_id)
				for i in li:
					if i == p_id:
						c1+=1
				c+=1 
				#print "{} {} {} {}".format(u_id,t_diff,c,c1)
				writer.writerow([u_id,t_diff,p_id,c,c1])

			else:
				s_time = time 
				t_diff = 0
				del li[:] 
				li = []
				li.append(p_id)
				c = 1
				c1 = 1
				#print "{} {} {} {}".format(u_id,t_diff,c,c1)
				writer.writerow([u_id,t_diff,p_id,c,c1])
				pre_id = u_id




def linear_reg():
	sc = SparkContext()

	#Load the csv file into a RDD
	autoData = sc.textFile('data.csv')
	autoData.cache()

	#Use default for average HP

	def transformToNumeric( inputStr) :
	    attList=inputStr.split(",")
	     
	    #Filter out columns not wanted at this stage
	    values= Vectors.dense([ float(attList[0]), \
	                     float(attList[1]), \
	                     float(attList[2]), \
	                     float(attList[3]), \
	                     float(attList[4]), \
	                     float(attList[5])
	                     ])
	    return values

	autoVectors = autoData.map(transformToNumeric)


	sqlContext = SQLContext(sc)

	def transformToLabeledPoint(inStr) :
	    lp = ( float(inStr[5]), Vectors.dense([inStr[1],inStr[3],inStr[4]]))
	    return lp
	    
	autoLp = autoVectors.map(transformToLabeledPoint)
	autoDF = sqlContext.createDataFrame(autoLp,["label", "features"])
	autoDF.select("label","features").show(10)



	lr = LinearRegression()
	lrModel = lr.fit(autoDF)
	r = []
	r.append(float(lrModel.intercept))
	r.append(str(lrModel.coefficients))
	# print "Coefficients: " + str(lrModel.coefficients)
	# print "Intercept: " + str(lrModel.intercept)
	return r 






def ptraining_data():
	df = pd.read_csv('sort_file.csv')
	li = []
	with open('data.csv', 'w') as wfile:
		writer = csv.writer(wfile)
		pre_id = ''
		for idx, row in df.iterrows():
			u_id = row['user_id']
			time = datetime.strptime(row['timestamp'],'%Y-%m-%dT%H:%M:%S.%fZ')
			time = float(time.strftime("%s.%f"))
			p_id = row['item_id']
			buy = row['buy']


			if u_id == pre_id:
				t_diff = time - s_time
				#c = fun1(u_id,time)
				c1 = 0
				li.append(p_id)
				for i in li:
					if i == p_id:
						c1+=1
				c+=1 
				print "{} {} {} {}".format(u_id,t_diff,c,c1)
				writer.writerow([u_id,t_diff,p_id,c,c1,buy])

			else:
				s_time = time 
				t_diff = 0
				del li[:] 
				li = []
				li.append(p_id)
				c = 1
				c1 = 1
				print "{} {} {} {}".format(u_id,t_diff,c,c1)
				writer.writerow([u_id,t_diff,p_id,c,c1,buy])
				pre_id = u_id

							
							



if __name__ == '__main__':
	ptraining_data()
	r = linear_reg()
	r0 = []
	r0.append(r[0])
	r1 = r[1].strip('[')
	r2 = r1.strip(']')
	r3 = r2.split(',')
	result = r0+r3
	print result
	#result = [-0.04792350124887441, '3.25103615086e-06', '0.00238898088314', '0.0452184563619']
	#ptest_data()
	calculate(result)
	check_result()
	roc_graph()
