from exomodel import ExoModel


exomodel = ExoModel()

exomodel.load('mnist_model.h5')
exomodel.evaluate()
exomodel.predict_one_digit()
exomodel.predict_all_digits()