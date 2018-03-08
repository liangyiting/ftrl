# encoding utf-8
import numpy as np
from scipy import sparse
import os
import pdb

temp_dir="/data/liangyiting/gbdt/tmp"
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

def f_tomat(pathIn,indexSet):
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

	n_feature=len(dic.keys())
	x_set=indexSet
	for k in range(len(indexSet)):
		index=indexSet[k]
		start=index[0]
		dim=index[1]
		x_sparse_k=sparse.lil_matrix((dim,n_feature))
		for i in range(dim):
			appInstall=Install[i+start]
			for appid in appInstall:
				x_sparse_k[i,apps_index[appid]]=1
		x_set[k]=x_sparse_k
	return(x_set,y)

def merge_y(ya,yb):
	la=len(ya);lb=len(yb);
	y=np.zeros((la+lb,))
	y[:la]=ya
	y[la:]=yb
	return(y)

def f_ftrl(x,y,path):
	fout=open(path,'w')
	N_sample,N_feature=x.shape
	for i in range(N_sample):
		if i%1000==1:
			print(i)
		string=str(int(y[i]))
		for j in range(N_feature):
			xij=x[i,j]
			if xij==0:
				continue
			else:
				string+=" %d:1"%j
		fout.write(string+'\n')
	fout.close()

def f_ftrl_1(x,y,path,max_leaf_nodes,dim=0,exist_file=0):
	fout=open(path,'w')
	N_sample,N_tree=x.shape
	flag=0
#	try:
#		fin=open(exist_file,'r')
#		flag=1
	for i in range(N_sample):
		if i%10000==9999:
			print(i)
		if flag==0:
			string=str(int(y[i]))
#		else:
#			string=fin.readline().strip()
		for j in range(N_tree):
			xij=int(x[i,j]);indexij=xij+j*max_leaf_nodes+dim;
			string+=' '+str(indexij)+':1'
		fout.write(string+'\n')
	fout.close()
#	if flag==1:
#		fin.close()

def myTransform(x,max_leaf_nodes):
	N_sample,N_feature=x.shape
	N_z=(N_feature)*max_leaf_nodes;
	z=sparse.lil_matrix((N_sample,N_z))
	for i in range(N_sample):
		if i%1000==1:
			print(i)
		for j in range(N_feature):
			indexij=int(x[i,j])+max_leaf_nodes*j
			z[i,indexij]=1
	return(z)



def shuffle(path):
	os.system("cd /data/liangyiting/gbdt")
	os.system("/data/liangyiting/user_interest_model/ftrl/shuffle1.sh "+path \
			+ " /data/liangyiting/gbdt/log.sf")
	os.system("rm -f "+path)
	os.system("mv log.sf "+path)

def read(result_txt):
	y=open(result_txt,'r').readlines()
	y=[float(t.strip()) for t in y]
	y=np.array(y)
	return(y)
	
def f_chouqu(pathIn,index,pathOut):
		fin=open(pathIn,'r')
		fout=open(pathOut,'w')
		start=index[0]
		final=index[-1]+start
		for i,line in enumerate(fin):
			if i<start:
				continue
			if i==final:
				break
			else:
				fout.write(line)
		fin.close()
		fout.close()

from sklearn.ensemble import GradientBoostingClassifier as GBC
from sklearn.preprocessing import OneHotEncoder as OHE
from sklearn.cross_validation import train_test_split 
from sklearn.metrics import roc_auc_score as auc
from sklearn.ensemble import GradientBoostingRegressor as GBDT
from sklearn.linear_model import SGDRegressor as sgd
from scipy.sparse import vstack
#######################################################
def gbdt_lr(para):
	print("gbdt_lr")
	x_train=para[0];x_train_lr=para[1];x_test=para[2];
	y_train=para[3];y_train_lr=para[4];y_test=para[5];
	maxleafnodes=11
	gbc=GBDT(max_leaf_nodes=maxleafnodes-1,n_estimators=600,min_samples_leaf=5,max_depth=3,learning_rate=0.02,subsample=0.2,max_features=0.1)
	gbc.fit(x_train,y_train);
	ohe=OHE();ohe.fit(gbc.apply(x_train)[:,:])
	li=gbc.apply(x_train_lr)[:,:];
	x_train_lr_gbc=ohe.transform(li)
	#x_train_lr_gbc=myTransform(li,max_leaf_nodes=maxleafnodes)
	li=gbc.apply(x_test)[:,:]
	x_test_gbc=ohe.transform(li)
	#x_test_gbc=myTransform(li,max_leaf_nodes=maxleafnodes)
	del(li)
	lr=sgd(n_iter=50)
	lr.fit(x_train_lr_gbc,y_train_lr)
	yp=lr.predict(x_test_gbc)
	print("GBDT+SGD: "+str(auc(y_test,yp)))
	return(gbc,yp)
#############################################
def gbdt_test(para):
	x_train=para[0];x_train_lr=para[1];x_test=para[2];
	import pdb
	pdb.set_trace()
	y_train=para[3];y_train_lr=para[4];y_test=para[5];
	import pdb;
	pdb.set_trace()
	xt=vstack([para[0],para[1]]);yt=merge_y(y_train,y_train_lr)
	#para[0]=0;para[1]=0;
	clf=GBDT();
	clf.subsample=0.1;clf.max_features=0.05;clf.min_samples_leaf=5;
	clf.n_estimators=200;
	clf.learning_rate=0.03;
	clf.fit(xt,yt);yp_gbdt=clf.predict((para[2]).toarray());
	print("GBDT: "+ str(auc(y_test,yp_gbdt)))
	return(clf)
#################################################
#逻辑回归，用随机梯度下降方法
def sgd_test(para):
	x_train=para[0];x_train_lr=para[1];x_test=para[2];y_train=para[3];y_train_lr=para[4];y_test=para[5];
	xt=vstack([x_train,x_train_lr]);yt=merge_y(y_train,y_train_lr)
	clf1=sgd(n_iter=20,eta0=1,alpha=3)
	clf1.fit(xt,yt);yp_sgd=clf1.predict(x_test);
	print("SGD: "+ str(auc(y_test,yp_sgd)))
####################################################
#ftrl模型训练预测，用sklearn的auc函数计算AUC
def ftrl(para,pathIn):
	y_test=para[5]
	N_tr=para[6];N_tr_lr=para[7];N=para[8];Ne=N-N_tr-N_tr_lr;
	f_chouqu(pathIn,[0,N_tr+N_tr_lr],temp_dir + '/train')
	f_chouqu(pathIn,[N_tr+N_tr_lr,Ne],temp_dir+'/test')
	ftrl_train="ftrl_train_wz -f "+temp_dir+'/train'+" -m model/ftrl_model"
	ftrl_predict="ftrl_predict_wz -t "+temp_dir+'/test'+" -m model/ftrl_model -o model/predict.val"
	os.system("rm -f "+temp_dir+"/*.cache")
	os.system("rm -f model/*.cache")
	os.system(ftrl_train)
	os.system(ftrl_predict)
	yp_ftrl=read('model/predict.val')
	print("FTRL: " + str(auc(y_test,yp_ftrl)))
	return(yp_ftrl)
############################################################
#用ftrl训练GBDT产生的特征
def gbdt_ftrl(para,gbc):
	#x_train=para[0];x_train_lr=para[1];x_test=para[2];
	y_train=para[3];y_train_lr=para[4];y_test=para[5];
	te=temp_dir+'/test_gbc'
	tr=temp_dir+'/train_gbc'
	maxleafnodes=50
	f_ftrl_1(gbc.apply(para[1])[:,:],y_train_lr,tr,max_leaf_nodes=maxleafnodes)#,dim=1000000,exist_file=temp_dir+'/train')
	f_ftrl_1(gbc.apply(para[2])[:,:],y_test,te,max_leaf_nodes=maxleafnodes)#,dim=1000000,exist_file=temp_dir+'/test')
	ftrl_train="ftrl_train_wz -f "+temp_dir+'/train_gbc'+" -m model/ftrl_model.gbc"
	ftrl_predict="ftrl_predict_wz -t "+temp_dir+'/test_gbc'+" -m model/ftrl_model.gbc -o model/predict.val.gbc"
	os.system("rm -f "+temp_dir+"/*.cache");os.system(ftrl_train);os.system(ftrl_predict)
	yp_gbc_ftrl=read('model/predict.val.gbc');
	print("GBDT+FTRL: " + str(auc(y_test,yp_gbc_ftrl)))
	return(yp_gbc_ftrl)
