
def f_write(fin,fout):
	for i,line in enumerate(fin):
		s=line.strip().split(' ')
		label=s[0]
		#st=str(label)
		st=label
		for t in s[1:]:
			z=t.split(':')
			appid=z[0]
			if int(appid)>300000:
				break
			st+=' '+t
		if len(st)>1:
			fout.write(st+'\n')
	fout.close()
	fin.close()
import sys
import time 
t0=time.clock()
fin=open(sys.argv[1],'r')
fout=open(sys.argv[2],'w')
f_write(fin,fout)
t1=time.clock()
print('t1-t0=%f'%(t1-t0))
