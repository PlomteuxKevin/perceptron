from inspect import _empty
import numpy as np
import pandas as pd
import sys, os
import time
import json

from pandas.core.groupby.groupby import DataFrame

from .perceptron import p
from pitch.vprint import Vprint, vprint
from tqdm import tqdm

class Pitch():
    def __init__(self, train_data, train_labels, test_data, test_labels):
        self.x_train = np.array(train_data)
        self.labels = np.array(train_labels)
        self.x_test = np.array(test_data)
        self.test_labels = np.array(test_labels)
        self.n = p(self.x_train[0])
        self.error = None
        self.plomteux_treshold = None

    def __str__(self):
        return f"Model : weights {self.n.ws} biais {self.n.b}"

### --------------------------------------------------------------------------------------------
### Class Methodes
### --------------------------------------------------------------------------------------------
### load : load existing model parameters
### pitch_data : prepare the data from the dataset (DataFrame)
### --------------------------------------------------------------------------------------------

    ### ----------------------------------------------------------------------------------------
    ### NAME : load
    ### ----------------------------------------------------------------------------------------
    ### Description : Load model parameters (weights and bias)
    ### ----------------------------------------------------------------------------------------
    ### RETURN : None
    ### ----------------------------------------------------------------------------------------
    @classmethod
    def load(cls, file="model.p5"):
        with open(file, 'r') as file:
            model_data = json.load(file)

        model = Pitch((0,0),(0,0),(0,0),(0,0))

        model.n.ws = np.array(model_data['weights'])
        model.n.b = float(model_data['bias'])

        return model

    ### ----------------------------------------------------------------------------------------
    ### NAME : pitch_data
    ### ----------------------------------------------------------------------------------------
    ### data (type : pandas.DataFrame) : data from dataset with x and y
    ### y_index (type : int,) : determine the index position of the y column
    ### y_name (type : str) : determine the name of the y column
    ### --- The parmeters y_index and y_name are applie with XOR logic. So use only 1 the 2
    ### --- paramters
    ### normed (type : bool) : Set to True if you want x_train and x_test normalized data output
    ### seed (type : int) : put a seed if you want the same rand data sample.
    ### RETURN : x_train, y_train, x_test, y_test (numpy array)
    ### ----------------------------------------------------------------------------------------
    @classmethod
    def pitch_data(cls, data:pd.DataFrame, y_index:int=None, y_name:str=None, normed:bool=False, seed:int=None) -> pd.DataFrame:

        # Generate the RandomState for the sample
        rd = np.random.RandomState(seed=seed)

        # create the 80/20 ratio sample for random train and test data
        train_data = data.sample(frac=0.8, random_state=rd)
        test_data = data.drop(train_data.index)

        # create de x(i) and y data
        if y_index and not y_name:
            y_train = train_data.iloc[:, y_index]
            y_test = test_data.iloc[: y_index]
            x_train = train_data.drop(train_data.columns[y_index], axis=1)
            x_test = test_data.drop(test_data.columns[y_index], axis=1)
        elif y_name and not y_index:
            x_train = train_data.drop(y_name, axis=1)
            y_train = train_data[y_name]
            x_test = test_data.drop(y_name, axis=1)
            y_test = test_data[y_name]
        elif not y_index and not y_name:
            sys.stderr.write("ERROR in pitch_class.py : Pitch.pitch_data() missing 1 argument (Expected : y_index OR y_name)\n")
            exit(1)
        else:
            sys.stderr.write("ERROR in pitch_class.py : Pitch.pitch_data() 1 argument except of 2 (Expected : y_index OR y_name)\n")
            exit(1)

        # Data normalization
        if normed:
            x_train = (x_train - x_train.mean()) / x_train.std()
            x_test = (x_test - x_train.mean()) / x_train.std()

        return x_train, y_train, x_test, y_test


### --------------------------------------------------------------------------------------------
### Public Methodes
### --------------------------------------------------------------------------------------------
### predict : predict Y based on x(i) input
### save : save the weight and bias into file
### train : train the model
### verbose : set the model to verbose mode
### --------------------------------------------------------------------------------------------


    ### ----------------------------------------------------------------------------------------
    ### NAME : predict
    ### ----------------------------------------------------------------------------------------
    ### Description : Save the parameters (weights and biais) into a file to be used in
    ###               futher project.
    ### ----------------------------------------------------------------------------------------
    ### file (type : str) : name of the output file
    ### RETURN : np.array, list of input + prediction
    ### ----------------------------------------------------------------------------------------
    def predict(self, input:np.array):
        pred = np.empty((input.shape[0], input.shape[1]+1), dtype=object)
        for i, s_input in enumerate(np.array(input)):
            pred[i] = [*s_input, self.__predict(s_input)]

        return pd.DataFrame(pred, columns=[f'x{i+1}' for i in range(input.shape[1])] + ['pred'])

    ### ----------------------------------------------------------------------------------------
    ### NAME : save
    ### ----------------------------------------------------------------------------------------
    ### Description : Save the parameters (weights and biais) into a file to be used in
    ###               futher project.
    ### ----------------------------------------------------------------------------------------
    ### file (type : str) : name of the output file
    ### RETURN : None
    ### ----------------------------------------------------------------------------------------
    def save(self, file="model.p5"):
        model_data = {'weights': self.n.ws.tolist(), 'bias': self.n.b}
        with open(file, 'w') as file:
            json.dump(model_data, file)

    ### ----------------------------------------------------------------------------------------
    ### NAME : regLine
    ### ----------------------------------------------------------------------------------------
    ### Description : Find the 2 points of the regretion line
    ### --- Use only for 2 dimensions data
    ### ----------------------------------------------------------------------------------------
    ### todo
    ### RETURN : ((x1, x2), (y1, y2)) tuple of 2 tuples
    ### ----------------------------------------------------------------------------------------
    def regLine(self, data=None):
        try:
            w1, w2 = self.n.ws
            m = -w1 / w2
            b = -self.n.b / w2
        except:
            raise ValueError("Model have more then 2 x in input. Impossible to calculate regression line.")

        if data is None:
            try:
                x1 = np.min(self.x_train[:, 0])
                x2 = np.max(self.x_train[:, 0])
            except:
                raise ValueError("Model contain no data, maybe because the model was loaded from file with model.load(). For loaded model, please use model.regLine(data=[Your DataFrame])")
        else:
            if isinstance(data, pd.DataFrame):
                if data.shape[1] == 2:
                    x1 = np.min(data.iloc[:, 0])
                    x2 = np.max(data.iloc[:, 0])
                else:
                    raise ValueError(f"data shape in regLine(date:DataFrame) must be (:, 2) execept of {data.shape}")
            else:
                raise ValueError(f"data in regLine(date:DataFrame) must be DataFrame except of {type(data)}")

        # calculer les points y correspondants
        y1 = m * x1 + b
        y2 = m * x2 + b

        return ((x1, x2), (y1, y2))


    ### ----------------------------------------------------------------------------------------
    ### NAME : train
    ### ----------------------------------------------------------------------------------------
    ### Description : train the model based on x_train and y_train and eval the model with
    ###               x_test and y_test.
    ### ----------------------------------------------------------------------------------------
    ### todo
    ### RETURN : None
    ### ----------------------------------------------------------------------------------------
    def train(self, learning_rate:float, epochs:int, metric:bool=False):
        vprint("*** Start Model Training ***")
        dataLen = len(self.x_train)
        self.error = np.zeros([epochs, dataLen+1])
        start_time = time.time()
        for epoch in range(epochs):
            i = vp = fp = vn = fn = 0
            with tqdm(total=dataLen, desc=f"Epoch {epoch+1}",leave=True, ncols=100) as pbar:
                for x, label in zip(self.x_train, self.labels):
                    self.n.predict(x)
                    error = self.__lost_func(self.n.y, label)
                    i += 1
                    if self.n.y != label:
                        self.n.ws += learning_rate * error * x
                        self.n.b += learning_rate * error
                        self.error[epoch, i] = error

                    if metric:
                        if label:
                            vp += 1 if self.n.y == label else 0
                            fn += 1 if self.n.y != label else 0
                        else:
                            vn += 1 if self.n.y == label else 0
                            fp += 1 if self.n.y != label else 0

                    pbar.update(1)
            if metric:
                loss = self.error[epoch,:].std()
                error = int(np.sum(np.abs(self.error[epoch,:])))/dataLen
                accuracy = vp / (vp + fp)
                sensitivity = vp / (vp + fn)
                score_f1 = (accuracy * sensitivity) / (accuracy + sensitivity)
                fpr = fp / (fp + vn)
                vprint(f"Loss: {self.error[epoch,:].std():.4f}  ", end="")
                vprint(f"Error: {error:.4f}  ", end="")
                vprint(f"Accuracy: {accuracy:.4f}  ", end="")
                vprint(f"Sensitivity: {sensitivity:.4f}  ", end="")
                vprint(f"Score_f1: {score_f1:.4f}  ", end="")
                vprint(f"FPR: {fpr:.4f}")
        end_time = time.time()
        vprint(f"*** Model Trained *** (in time : {end_time - start_time:.2f}s)")

        self.__test(self.x_test, self.test_labels)

    ### ----------------------------------------------------------------------------------------
    ### NAME : verbose
    ### ----------------------------------------------------------------------------------------
    ### val (type : bool) : set the model to versbose (True) or not verbose (False)
    ### --- Not verbose by default
    ### RETURN : None
    ### ----------------------------------------------------------------------------------------
    def verbose(self, val:bool=False):
        Vprint.verbose=val


### --------------------------------------------------------------------------------------------
### Prived Methodes
### --------------------------------------------------------------------------------------------
### lost_fund : Lost Function based on Binary test (1 or 0)
### predict : predict the value based on trained model weights and biais
### pthresh : Plomteux treshold, calculate the critical square to improve accuracy
###           and optimise the model
### test : test the accuracy of the model
### --------------------------------------------------------------------------------------------
    def __lost_func(self, y_pred:int, y_target:int) -> int: # Binary lost function
        # 0 if no error
        # +1 if need to add learning rate
        # -1 if need to substract learning rate
        return y_target - y_pred

    def __predict(self, test_data:np.array) -> int:
        self.n.predict(test_data)
        return self.n.y

    def __test(self, test_data:np.array, test_labels:np.array) -> int:
        nb_test = 0
        nb_true = 0

        for test, label in zip(np.array(test_data), np.array(test_labels)):
            nb_test += 1
            test_pred = self.__predict(test)
            if test_pred == label:
                nb_true += 1

        vprint("Test done : ", nb_test)
        vprint(f"Test Accuracy : {round(nb_true/nb_test*100, 2)}%")
