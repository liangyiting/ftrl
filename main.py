#from package import *
import package
import sys
sys.path.append('/data/liangyiting/gbdt')
pathIn='/data/liangyiting/gbdt/log.apl'
N=package.f_lines(pathIn)
N_tr=int(0.3*N);  k1=N_tr
N_tr_lr=int(0.3*N); k2=N_tr+N_tr_lr
Ne=N-k2;
x_set,y=package.f_tomat(pathIn,[[0,N_tr],[k1,N_tr_lr],[k2,Ne]])
x_train=x_set[0];y_train=y[:k1]
x_train_lr=x_set[1];y_train_lr=y[k1:k2]
x_test=x_set[2];y_test=y[k2:]
del(x_set)
############################################################
para=[x_train,x_train_lr,x_test,y_train,y_train_lr,y_test,N_tr,N_tr_lr,N]
#######################################################
import pdb
#pdb.set_trace()
#package.gbdt_test(para)
#############################################
gbc,yp_lr=package.gbdt_lr(para)
####################################################
#ftrl模型训练预测，用sklearn的auc函数计算AUC
yp_ftrl=package.ftrl(para,pathIn)
print(package.auc(y_test,yp_lr+yp_ftrl))
############################################################
#用ftrl训练GBDT产生的特征
pdb.set_trace()
yp_2=package.gbdt_ftrl(para,gbc)
print(package.auc(y_test,yp_ftrl+yp_2))
