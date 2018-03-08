# encoding utf-8
import numpy as np
from scipy import sparse
def f_lines(pathIn):
	#计算样例文件的行数
	f=open(pathIn,'r')
	line=f.readline()
	count=0
	while line:
		count+=1
		line=f.readline()
	f.close()
	return(count)

def f_tomat(pathIn):
	#将ftrl输入格式的数据转化为sklearn所需的矩阵格式，并且用稀疏矩阵存储来减小内存消耗
	n=f_lines(pathIn)
	fin=open(pathIn,'r')
	Install=[]#储存每一行包含的app
	dic={}#储存app总的被安装次数
	y=np.zeros((n,))
	for i,line in enumerate(fin):
		s=line.strip().split(' ')
		y[i]=int(s[0])
		apps=set()
		for t in s[1:]:
			appid=int(t.split(':')[0])
			if appid>300000:
				continue
			try:
				dic[appid]+=1
			except:
				dic[appid]=1
			apps.add(appid)
		Install.append(list(apps))

	apps=list(dic.keys());q=len(dic.keys());
	apps_index={}#建立appid和新的特征编号之间的映射关系
	for i in range(q):
		apps_index[apps[i]]=i
	
	xsparse=sparse.lil_matrix((n,q))
	for i in range(n):
		appInstall=Install[i]
		for appid in appInstall:
			xsparse[i,apps_index[appid]]=1
	return(xsparse,y)

if __name__=='__main__':
	import sys
	sys.path.append('/data/liangyiting/gbdt')

	pathIn='/data/liangyiting/gbdt/log'
	x,y=f_tomat(pathIn)#	
	li=np.random.uniform(0,1,len(y));i=np.argsort(li);x=x[i];y=y[i];#打乱样本
	k=int(0.6*len(y));xt=x[:k];yt=y[:k];xv=x[k:];yv=y[k:]#切分训练集和测试集
	import pdb
#	pdb.set_trace()
	from sklearn.metrics import roc_auc_score as auc
	from sklearn.linear_model import SGDRegressor as sgd
	from sklearn.ensemble import GradientBoostingRegressor as gbdt
	clf=gbdt();
	clf.subsample=0.2;clf.max_features=0.05;clf.min_samples_leaf=5;
#	clf.max_leaf_nodes=30;
	clf.n_estimators=200;
	clf.learning_rate=0.03;

	clf.fit(xt,yt);yp_gbdt=clf.predict(xv.toarray());print(auc(yv,yp_gbdt))

	clf1=sgd()
	clf1.fit(xt,yt);yp_sgd=clf1.predict(xv);print(auc(yv,yp_sgd))

	print(auc(yv,yp_gbdt+yp_sgd))
	
