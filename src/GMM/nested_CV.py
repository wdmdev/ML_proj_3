from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold
import numpy as np

def my_cv(X, y, models, K_in, K_out, seed=30):
    inner_cv = StratifiedKFold(n_splits=K_in, shuffle=True, random_state=seed)
    outer_cv = StratifiedKFold(n_splits=K_out, shuffle=True, random_state=seed)

    #Inner fold
    best_inner_models = []
    for train_idx, test_idx in outer_cv.split(X,y):
        train = X[train_idx]
        train_y = y[train_idx]
        test = X[test_idx]
        
        inner_test_set_lens = []
        inner_model_scores = []
        for in_train_idx, in_test_idx in inner_cv.split(train,train_y):
                inner_train = train[in_train_idx]
                inner_test = train[in_test_idx]

                inner_test_set_lens.append(len(inner_test))
                inner_score = []
                for m in models:
                        m.fit(inner_train)
                        inner_score.append(-m.score_samples(inner_test).sum())
                
                inner_model_scores.append(inner_score)
        
        inner_model_errors = np.sum(np.asarray(inner_model_scores)
                                *np.transpose([[tl/len(train) for tl in inner_test_set_lens]]), axis=0)
        inner_best_model_idx = np.argmin(inner_model_errors)
        in_best_model = models[inner_best_model_idx]
        best_inner_models.append(in_best_model)

    #Outer fold
    final_errors = []
    final_test_set_lens = []
    for train_idx, test_idx in outer_cv.split(X,y):
        train = X[train_idx]
        train_y = y[train_idx]
        test = X[test_idx]

        final_test_set_lens.append(len(test))
        outer_model_errors = []
        for m in models:
                m.fit(train)
                outer_model_errors.append(-m.score_samples(test).sum())
        
        final_errors.append(outer_model_errors)

    #Select final model
    model_gen_errors = np.sum(np.asarray(final_errors)
                                *np.transpose([[ftl/len(y) for ftl in final_test_set_lens]]), axis=0)
    
    final_best_model_idx = np.argmin(model_gen_errors)
    return models[final_best_model_idx], np.min(model_gen_errors)