import model

m = model.model([model.layer(9), model.dense(9)])

m.compile()

rm = model.randomModel(9)

model.fight(m, rm)

