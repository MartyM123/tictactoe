def score(models:list)->list:
    '''return the overall score of each model'''
    table=np.zeros((len(models), len(models)))
    for i,model1 in enumerate(models):
        for j, model2 in enumerate(models[i+1:]):
            j+=i+1
            res=fight(model1, model2)
            table[i][j]+=res[0]
            table[j][i]+=res[1]

            res=fight(model2, model1)
            table[i][j]+=res[0]
            table[j][i]+=res[1]

def score_with_random(models:list, n_games:10):
    res=[]
    for Model in models:
        score = 0
        for i in range(n_games):
            score += fight(Model, random_model(9))[0]
        res.append(score)
    return np.array(res)

def choose_parents(models:list, score:list, n=2)->list:
    '''choose two parents with gratest score and return them in list'''
    zipped_sorted = sorted(zip(models,score), key=lambda x:x[1], reverse=True)[:n]
    return [a for a,b in zipped_sorted]

def reproduce(parents, init_model:model, weighted_score:list) -> model:
    w=w*parent.layers[i_layer].weights
    b=b*parent.layers[i_layer].biases
    for i_layer in range(len(init_model.layers)):
        if init_model.layers[i_layer].is_mutateable:
            w=np.ones(init_model.layers[i_layer].weights.shape)
            b=np.ones(init_model.layers[i_layer].biases.shape)
            for i,parent in enumerate(parents):
                w=w*parent.layers[i_layer].weights*weighted_score[i]
                b=b*parent.layers[i_layer].biases*weighted_score[i]
            init_model.layers[i_layer].weights=w**(1/n)
            init_model.layers[i_layer].biases=b*(1/n)
    return init_model

def one_cycle(models:list, n_parent=2)->list:
    n=len(models)
    s = score(models)
    s+=0.001
    parents=choose_parents(models, s/max(s), n_parent)
    models2=[]
    for i in range(n):
        Model = model()
        Model.layers = [layer(9), dense(9)]
        Model.compile()
        Model= reproduce(parents, Model, s)
        models2.append(Model)
    return models2