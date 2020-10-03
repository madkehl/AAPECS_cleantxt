from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV

def run_each_participant(function, X, y, feat_bool = True):
    '''
    INPUT:
    scikit rf regressor model, x vals, y vals, boolean whether or not to find feature importances
    
    OUTPUT:
    mae scores, mse scores, feature ranking
    '''
    X = X.copy()
    y = y.copy()
    accuracy_model = []
    mse_ = []
    feat_imps = []
    
    for i in range(0,len(X[X.columns[0]])):
        
        X_test = X.loc[i]
        y_test = pd.Series(y[i])
        X_test = X_test.values.reshape(1, -1)
        
        X_train = X.drop(X.index[i])
        y_train =np.delete(y, i)
        model = function.fit(X_train, y_train)
        
        accuracy_model.append(mean_absolute_error(y_test, model.predict(X_test)))
        mse_.append(mean_squared_error(y_test, model.predict(X_test)))
        if feat_bool == True:
            feat_imp = pd.DataFrame({
    
                'labels': X_train.columns,
                'importance':model.feature_importances_
            })

            feat_imp = feat_imp.reset_index(drop = True)
            feat_imp = feat_imp.sort_values(by = 'importance', axis = 0)
            feat_imps.append(feat_imp)
    
    return([accuracy_model, mse_, feat_imps])

def specify_best(model, X, y, typ = 'RF'):
    '''
    INPUT:
    scikit regressor model, x vals, y vals, str telling if it is rf or svr
    
    OUTPUT:
    best parameter values for min samples split, max features, or kernel gamma and C
    '''
    if typ == 'RF':
        params =  {'min_samples_split': np.arange(2, 100, 5), 'max_features': np.arange(2,15,2)}
    elif typ == 'svr':
         params =  {'kernel': ['rbf', 'linear', 'sigmoid'], 'gamma': np.arange(0.001,0.9,0.1), 'C': [0.0001,0.01,1,10,100]}
    clf = GridSearchCV(model, params,cv = 5)
    clf.fit(X, y)
    print(clf.best_params_)
    return(clf.best_params_)

def ext_featEPRF(res2, featn = 15, mean = False):
    '''
    INPUT:
    result from run_each_participant
    
    OUTPUT:
    most important features across models etc.
    '''
    ext_feat = []
    count = 0
    
    for i in res2:
        i = i.reset_index(drop = False)
        i['index'] = [count]*len(i['index'])
        i = i.pivot(index = 'index', columns='labels', values='importance')
        ext_feat.append(i)
        count += 1
    
    ext_featdf = pd.concat(ext_feat, axis = 0)
    ext_featdf1 = ext_featdf.fillna(0)
    ext_featsums = ext_featdf1.mean(axis = 0)
    if mean == False:
        new_ext = ext_featsums.sort_values().iloc[len(ext_featsums)-featn: len(ext_featsums)]
    else:
        new_ext = ext_featsums.sort_values()
        new_ext = new_ext.where(new_ext > np.mean(new_ext.values))
        new_ext = new_ext.dropna()
        
    names = new_ext.index
    ext_featsums = ext_featdf.count(axis = 0)
    new_ext2 = ext_featsums.loc[list(names)]
    
    results = pd.DataFrame({
        'features':names,
        'feat_imps':new_ext.values,
        'count_models': new_ext2.values
    })
    return(results)
    
def print_vals(best_param, rf_res, eprf):
    print('mss: ' + str(best_param.get('min_samples_split')))
    print('max_feat: ' + str(best_param.get('max_features')))
    print('mean_mae: ' + str(np.mean(rf_res[0])))
    print('mean_mse: '+ str(np.mean(rf_res[1])))
    print('\n' + 'frequency: ' + '\n')
    for i in list(eprf['count_models']):
        print(i)  
    print('\n' + 'feat_imp: ' + '\n')
    for i in list(eprf['feat_imps']):
        print(i)
    print('\n' + 'top_ features' + '\n')
    for i in list(eprf['features']):
        print(i)
