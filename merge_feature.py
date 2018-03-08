import sys

def f_dim(name,dim,outname):
	f1=open(name,'r')
	fout=open(outname,'w')
	for i,line in enumerate(f1):
		s=line.strip().split('\t')
		st=s[0]
		for t in s[1:]:
			t=t.split(':')
			st+='\t'+str(int(t[0])+dim)+':'+t[1]
		st+='\n'
		fout.write(st)
	f1.close()
	fout.close()

def f_merge(a,b,outname):
	f1=open(a,'r')
	f2=open(b,'r')
	fout=open(outname,'w')
	line1=f1.readline()
	line2=f2.readline()
	while line1:
		s2=line2.split('\t',1)
		try:
			st=line1.strip()+'\t'+s2[1]
		except:
			pass
		fout.write(st)
		line1=f1.readline()
		line2=f2.readline()
	f1.close()
	f2.close()
	fout.close()

import sys
def fun(pathIn_1,pathIn_2,pathOut,dim)
if dim==0:
	f_merge(pathIn_1,pathIn_2,pathOut)
else:
	out='/data/liangyiting/gbdt/tmp/log_merge'
	f_dim(pathIn_1,dim,out)
	f_merge(out,pathIn_2,pathOut)
