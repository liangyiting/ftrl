import os
import sys
pathIn=sys.argv[1]
pathOut=sys.argv[2]
fin=open(pathIn,'r')
fout=open(pathOut,'w')
for i,line in enumerate(fin):
	s=line.split(' ',1)
	y=int(s[0])
	if y<0:
		y=0
	elif y>1:
		y=1
	newline=str(y)+' '+ s[1]
	fout.write(newline)
fin.close()
fout.close()
