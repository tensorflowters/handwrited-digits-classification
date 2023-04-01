from model import Model

model = Model()

model.get_image()
model.get_distribution()
model.get_mean()

model.train(30)
