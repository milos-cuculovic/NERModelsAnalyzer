"""
Created on Fri May 28 08:48:33 2021
@author: chams
"""
import json
import string 
import random
import os


# Convert json to conll
def json_conll(jsonPath,conllPath):
    f = open(conllPath+'/conll.txt','w')
    with open(jsonPath) as fi:
        for jsonObj in fi:
            data = json.loads(jsonObj)
            listLABEL= data['array_agg']
            sentence= data['text']
            longS=len(sentence)
            incre=0
            wordd=""
            f.write("\n")
            while incre<longS:
                deb=incre
                if sentence[incre] in string.punctuation:
                    f.write(sentence[incre])
                    f.write(" ")
                    f.write("O")
                    f.write("\n")
                    incre=incre+1
                deb=incre
                while incre<longS and sentence[incre]!=" " and sentence[incre] not in string.punctuation :
                    wordd=wordd+sentence[incre]   
                    incre=incre+1
    
                if incre!=deb:
                    label="O"
                    for lab in listLABEL:  
                        lab = lab.replace(',', "") 
                        if deb >= int(lab.split()[0]):
                            if incre<= int(lab.split()[1]):
                                label=lab.split()[2]
                    f.write(wordd)
                    f.write(" ")
                    if label=="CONTENT":
                        label="O"
                    f.write(label)
                    f.write("\n")
               
                if incre<longS:
                    if sentence[incre] in string.punctuation:
                        f.write(sentence[incre])
                        f.write(" ")
                        f.write("O")
                        f.write("\n")
                incre=incre+1
                wordd=""
    f.close()


# remove all label from file
def removeLab(lab,file,file2):
    with open(file) as f:
        lines=f.readlines()
    with open(file2, "w") as f2:
        for line in lines:
            if labelConll(line)!=lab:
                f2.write(line)

# remove paragraphe form file never tested NOT SURE IT WORKS
def removeLabParag(lab,file,file2,oneon):
    with open(file) as f:
        lines=f.readlines()
    with open(file2, "w") as f2:
        para=0
        for line in lines:
            if labelConll(line)!=lab:
                f2.write(line)       
            elif para==oneon:
                f2.write(line) 
            if line.strip()=="":
                if para==oneon:
                    para=1
                else :
                    para=para+1
                

# Count number of a certain label in a file
def countLab(lb):
    with open('/home/chams/Documents/mdpi/dataset/train.txt') as f:
        lines = f.readlines()
        lab=0
        for line in lines:
            if labelConll(line)==lb :
                lab=lab+1
        return lab
 
    
 
# Create a list of a paragraph 
def listparagr(file,num):
    with open(file) as f:
        parag=[]
        lines =f.readlines()
        lines.append("\n")
        npar=0
        for line in lines:
            if line=="\n" and npar==num:
                return parag
            if line=="\n":
                npar=npar+1
            elif npar ==num:
                parag.append(line) 
      
  

# UnderSampling by remove label randomly
def randomUndSamp(label,percentage):
    numlab=countLab(label)
    keep=round(numlab*percentage/100)
    with open('/home/chams/Documents/mdpi/dataset/train.txt') as f:  
        lines = f.readlines()
    while numlab!=keep:
         line = random.choice(lines)
         if labelConll(line)==label :
            deli=line
            with open('/home/chams/Documents/mdpi/dataset/train.txt','w') as f: 
                for ligne in lines:
                    if ligne.strip() != deli.strip():
                        f.write(ligne)
                    else :
                        deli="000"
            with open('/home/chams/Documents/mdpi/dataset/train.txt') as f:  
                lines = f.readlines()
        
         numlab=countLab(label)


# idk en vrai
def numpara(file, nline):
    # numero de paragraphe
    with open(file) as f:
        lines=f.readlines()
        currentline=1
        numparag=1
        while currentline<nline :
            if lines[currentline].strip()=="":
                numparag=numparag+1
            currentline=currentline+1
        return numparag
 

# Explicit title
def numberofParaginAfile(file):
    parag=0
    with open(file) as f:
        lines=f.readlines()
        for line in lines:
            if line.strip()=="":
               parag=parag+1 
    return parag       

    
# idk^
def labelConll (line):
    lis = list(line.split(" "))
    length = len(lis)
    lab=lis[length-1]    
    return lab.strip()

#fonction qui met tous les paragraphe dans une autre fonction de maniere aleatoire 
def shuffleFile(file1,file2,file3):
     with open(file1) as f:
        lines=f.readlines()
        linesrest=[]
        for li in lines:
            linesrest.append(li)
        try:
            while True:
                linesrest.remove('\n')
        except ValueError:
            pass
      
        with open(file2) as fb:
            lines2=fb.readlines()
        while linesrest:
             nbPar=numberofParaginAfile(file1)
             rand=random.randint(1, nbPar)
             sentencelist=listparagr(file1,rand )
             with open(file2,"w") as f2:
                if lines2:
                    # try:
                    #    while True:
                    #         lines2.remove('\n')
                    # except ValueError:
                    #     pass
                    
                    for line2 in lines2:
                        f2.write(line2)
                for sentence in sentencelist:
                    if sentencelist.index(sentence)==0:
                        f2.write("\n")
                    if sentence!="\n" :
                        sentence.replace('\n', '')
                        f2.write(sentence)
             parag=0
             with open(file3,"w") as f1:
                 f1.write("\n")
                 for line in lines:
                    if line.strip()=="":
                        parag=parag+1
                    if parag!= rand:
                        f1.write(line)
                 f1.write("\n")
             linesrest=shuffleFile(file3, file2,file3)
        f.close()
        return linesrest 
    

# randomUndSamp("O", 10)        
# json_conll("/home/chams/Documents/mdpi/datasettoconl.json","/home/chams/Documents/mdpi")    
# removeLab("O","/home/chams/Documents/mdpi/conll.txt","/home/chams/Documents/mdpi/conll2.txt")    

removeLabParag("O","/home/chams/Documents/mdpi/conll.txt","/home/chams/Documents/mdpi/conll2.txt",5) 
shuffleFile('/home/chams/Documents/mdpi/datasetxt.txt', '/home/chams/Documents/mdpi/datasetxt2.txt','/home/chams/Documents/mdpi/datasetxt3.txt')
os.remove("/home/chams/Documents/mdpi/datasetxt3.txt")      
# print("TRIG")
# print(countLab("TRIGGER"))        
# print("O")
# print(countLab("O"))             
# print("loc")
# print(countLab("LOCATION"))       
# print("modal")
# print(countLab("MODAL"))       
# print("actio")
# print(countLab("ACTION"))       

