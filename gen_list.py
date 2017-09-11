import os
import random


#recommand to del list item ,but not remove,generate random index rather than item
def generate_list(files,label,parentFolder):
	for i in range(len(files)):
		files[i]=parentFolder+files[i]+" %s\n"%label
	valNum=len(files)/4
	valSet=random.sample(files, valNum)
	for item in valSet:
		files.remove(item)
	return valSet,files

trainListName="/home/dandan/imgset/train.txt"
valListName="/home/dandan/imgset/val.txt"
imPaths=[("/home/dandan/imgset/amt/hot/","0","amt/hot/"),\
    ("/home/dandan/imgset/amt/cold/","1","amt/cold/"),\
]
finalTrains=[]
finalVals=[]
for path in imPaths:
	imFiles=os.listdir(path[0])
	vals,trains=generate_list(imFiles, path[1],parentFolder=path[2])
	finalVals=finalVals+vals
	finalTrains=finalTrains+trains
	
f=open(trainListName,"w")
f.writelines(finalTrains)
f.close()

f=open(valListName,"w")
f.writelines(finalVals)
f.close()