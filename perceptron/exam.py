from pandas.core.indexes.base import PrettyDict
import pitch as pt
import aikiplot as ak
import math
import random

### --------------------------------------------------------------------------------------------
### DATASET : Catégorisation des résultats d'examens par rapport au temps de sommeil
###           et au travail fournis
### --------------------------------------------------------------------------------------------
data = pt.pd.read_csv("datas/exam.csv")

x_train, y_train, x_test, y_test = pt.Pitch.pitch_data(data, y_name='Result')

### --------------------------------------------------------------------------------------------

print("Prepared Data : ", x_train.shape, y_train.shape, x_test.shape, y_test.shape)

model = pt.Pitch(x_train, y_train, x_test, y_test)
model.verbose(True)
model.train(learning_rate=0.01, epochs=16, metric=True)

regPoint = model.regLine()
ak.scat2cat(data, regLine=regPoint, lim0=True, sp=3, autoName=True, file="graph/data_visualisation.png")

model.save()
model = pt.Pitch.load()

pred_values = model.predict(x_test)
ak.scat2cat(pred_values, lim0=True, sp=3, xlabel="Work Hours", ylabel="Sleep Hours", title="graph/Exam Result Prediction",file="predict_visualisation.png")
