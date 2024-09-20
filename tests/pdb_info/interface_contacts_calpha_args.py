#!/usr/bin/env python2
import linecache
import math
import sys

data = sys.argv[1]
c1 = sys.argv[2]
c2 = sys.argv[3]
cut = float(sys.argv[4])

size = open(data,"r")
lines = len(size.readlines())

n=0
r1 = []
r2 = []
rn1 = []
rn2 = []
x1 = []
x2 = []
y1 = []
y2 = []
z1 = []
z2 = []
ch1 = []
ch2 = []

res1 = ""
res2 = ""

for i in range(1,lines):
	a = linecache.getline(data,i)
	if a[0:4]=="ATOM" and a[21:22]==c1:
		if a[22:26] != res1: 
			if a[13:15]=="CA":
				r1.append(str(a[22:26]))
				rn1.append(str(a[17:20]))
				x1.append(float(str(a[30:38]))) 
				y1.append(float(str(a[38:46])))
				z1.append(float(str(a[46:54])))
				ch1.append(str(a[21:22]))
				res1 = a[22:26]
	if a[0:4]=="ATOM" and a[21:22]==c2:
		if a[22:26] != res2:
			if a[13:15]=="CA":			
				r2.append(str(a[22:26]))
				rn2.append(str(a[17:20]))
				x2.append(float(str(a[30:38]))) 
				y2.append(float(str(a[38:46])))
				z2.append(float(str(a[46:54])))
				ch2.append(str(a[21:22]))			
				res2 = a[22:26]
output = open("contactmap_calpha_"+data[:-4]+"_"+c1+c2+"_"+str(cut),"w")
output.write("res numb".ljust(10)+"res name".ljust(10)+"chain".ljust(8)+"res numb".ljust(10)+"res name".ljust(10)+"chain".ljust(8)+"distance".ljust(15)+"\n")
	
for j in range(0,len(r1)):
	if c1.strip() == c2.strip():	# If monomeric
		for k in range(j,len(r2)):
			dx = math.pow(x1[j]-x2[k],2)
			dy = math.pow(y1[j]-y2[k],2)
			dz = math.pow(z1[j]-z2[k],2)
			dist = math.pow(dx+dy+dz,0.5)		
			if dist <= cut and dist > 0:
				output.write(r1[j].ljust(10)+rn1[j].ljust(10)+ch1[j].ljust(8)+r2[k].ljust(10)+rn2[k].ljust(10)+ch2[k].ljust(8)+str(dist).ljust(15)+"\n")
				n+=1
	else:							# If multimeric
		for k in range(0,len(r2)):	
			dx = math.pow(x1[j]-x2[k],2)
			dy = math.pow(y1[j]-y2[k],2)
			dz = math.pow(z1[j]-z2[k],2)
			dist = math.pow(dx+dy+dz,0.5)		
			if dist <= cut and dist > 0:
				output.write(r1[j].ljust(10)+rn1[j].ljust(10)+ch1[j].ljust(8)+r2[k].ljust(10)+rn2[k].ljust(10)+ch2[k].ljust(8)+str(dist).ljust(15)+"\n")
				n+=1

print "\n\tNumber of interactions found: "+str(n)+"\n"			
print "\n\tFile saved as: "+"contactmap_calpha_"+data[:-4]+"_"+c1+c2+"_"+str(cut)+"\n"



	

	
