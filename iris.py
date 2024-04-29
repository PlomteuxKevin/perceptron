from pandas.core.indexes.base import PrettyDict
import pitch as pt
import aikiplot as ak
import math
import random


### --------------------------------------------------------------------------------------------
### DATASET : Catégorisation des fleures par rapport à la taille de leur pétale
### --------------------------------------------------------------------------------------------
data = pt.pd.read_csv("datas/iris.csv")
data = data.drop(columns=['sepal_lenght', 'sepal_width'])

x_train, y_train, x_test, y_test = pt.Pitch.pitch_data(data, y_name='class')

print("Prepared Data : ", x_train.shape, y_train.shape, x_test.shape, y_test.shape)

model = pt.Pitch(x_train, y_train, x_test, y_test)
model.verbose(True)
model.train(learning_rate=0.01, epochs=16, metric=True)

regPoint = model.regLine()
ak.scat2cat(data, regLine=regPoint, lim0=True, sp=3, autoName=True, file="data_visualisation.png")

model.save()
model = pt.Pitch.load()

pred_values = model.predict(data)
# pred_reg = model.regLine(x_train)
ak.scat2cat(pred_values, lim0=True, sp=3, xlabel="Work Hours", ylabel="Sleep Hours", title="Exam Result Prediction",file="predict_visualisation.png")
