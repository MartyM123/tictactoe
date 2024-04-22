from model import *

models=generate_random_models(20)
for i in range(50):
    models=one_cycle(models)
s=score(models)
model1=choose_parents(models, s, 1)[0]

models=generate_random_models(10)
for i in range(10):
    models=one_cycle(models)
s=score(models)
model2=choose_parents(models, s, 1)[0]

fight(model1, model2, show=True)