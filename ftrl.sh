cd /data/liangyiting/gbdt
root='/data/liangyiting/gbdt'
rm -f *.cache
f="xgboost.data"
sh ../user_interest_model/ftrl/shuffle1.sh $root/$f /data/liangyiting/gbdt/$f.sf
/usr/bin/python ../user_interest_model/ftrl/divide1.py $f.sf tr te 0.5

ftrl_train_wz -f tr -z 700000:8000000 -m model/ftrl_model --l1 0.01 --l2 0.001 --alpha 0.1
ftrl_predict_wz -t te -m model/ftrl_model -o model/predict.val
ftrl_predict_wz -t tr -m model/ftrl_model -o model/predict.val
