#Roxanne Dewing
#Gareth Marriott
#compile with:
#python3 NaiveBayes.py

import fileinput
import math

def extractVocab(list1):
	finlist = []
	for x in list1:
		temp = x.split()
		for word in temp:
			if word not in finlist:
				finlist.append(word)
	return finlist

def countDocsInClass(trainlabels, c):
	count = 0
	for x in trainlabels:
		if x == c:
			count+=1
	return count
	
def concatText(traindata, trainlabels, c):
	inc = 0
	finlist = []
	for x in traindata:
		if trainlabels[inc]==c:
			temp = x.split()
			for y in temp:
				finlist.append(y)
		inc+=1
	return finlist	

def sumAllTerms(V, fullvocab):
	finsum = 0
	for term in V:
		finsum +=fullvocab.count(term)
	return finsum
			
def trainMultinomialNM(traindata, trainlabels):
	V = extractVocab(traindata)
	N = len(traindata)
	prior = {}
	C = extractVocab(trainlabels)
	condprob = [[0 for x in range(len(C))] for y in range(len(V))] 
	y = 0
	dict = {}
	for c in C:
		T_ct = []
		Nc = countDocsInClass(trainlabels, c)
		prior.update({c:Nc/N})
		fullvocab = concatText(traindata, trainlabels, c)
		fullsum = sumAllTerms(V, fullvocab)+len(V)
		for term in V:
			counttokens = fullvocab.count(term)+1
			dict.update({(term, c): counttokens })		
	return V, prior, dict

def extractTokensFromDoc(V, d):
	finlist = []
	for x in d.split(" "):
		if x in V:
			if x not in finlist:
				finlist.append(x)
	return finlist
		
def ApplyMultinomialNB(C, V, prior, dict, d):
	W = extractTokensFromDoc(V, d)
	score = {}
	x = 0
	for c in C:
		temp = math.log(prior.get(c))
		for term in W:
			temp+= math.log(dict.get((term, c)))
		score.update({c:temp})
	classlabel = max(score, key=score.get) 
	return classlabel

def checkResults(data, label, C, V, prior, dict):
	x = 0
	correct = 0
	for d in data:
		if ApplyMultinomialNB(C, V, prior, dict, d) == label[x]:
			correct+=1
		x+=1
	print("Correct are: ")
	print(correct)
	print("Out of: ")
	print(len(label))
	return
		
def main():
	testdata = []
	testlabels = []
	traindata = []
	trainlabels = []
	
	f1 = open("testdata.txt", "r")
	for line in f1:
  		testdata.append(line.strip())
	f1.close()	
  
	f2 = open("testlabels.txt", "r")
	for line in f2:
  		testlabels.append(line.strip())
	f2.close()
  	
	f3 = open("traindata.txt", "r")
	for line in f3:
		traindata.append(line.strip())
	f3.close()
  	
	f4 = open("trainlabels.txt", "r")
	for line in f4:
		trainlabels.append(line.strip())
	f4.close()	
	
	print("Training on traindata.txt")
	
	#Call to train
	V, prior, dict = trainMultinomialNM(traindata, trainlabels)


	C = extractVocab(testlabels)
	print("Testing on testdata.txt")
	#Call to Apply Multionmial
	checkResults(testdata, testlabels, C, V, prior, dict)
	

if __name__=="__main__":
	main()
	
	
	
	
	
	
