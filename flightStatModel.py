import keras
# from keras.models import Sequential
# from keras.layers import Dense, Activation
import numpy as np
import datetime
import logging
import flightStatData


class kerasModel:
    def __init__(self, h5file, rawDBfile, dataPropertiesFile, dataTypes):
        'create the connection to the SQLite DB.'

        logFilename = "logs/" + str(datetime.date.today()) + ".log"
        logging.basicConfig(filename=logFilename, level=logging.DEBUG,
                            format="%(levelname)s: %(message)s - %(asctime)s")

        self.model = None
        self.X_train = None
        self.X_test = None
        self.Y_train = None
        self.Y_test = None

        self.dataClass = flightStatData.kerasData(h5file, rawDBfile, dataTypes)
        self.dataClass.load(dataPropertiesFile)

    def getData(self, fromIndex, toIndex, testPercentage):

        data = np.random.permutation(self.dataClass.getData(fromIndex, toIndex))

        Y = data[:, -1]
        X = data[:, 0:-1]

        i = int(X.shape[0] * testPercentage)

        self.X_test = X[0:i]
        self.X_train = X[(i - 1):-1]

        self.Y_test = Y[0:i]
        self.Y_train = Y[(i - 1):-1]

        # logging.info("X_train : %s", str(self.X_train))
        # logging.info("X_test : %s", str(self.X_test))
        # dict = {'X_train': X_train, 'X_test': X_test, 'Y_train': Y_train, 'Y_test': Y_test}

        return 0

    def create_model(self):
        self.model = keras.models.Sequential()

        self.model.add(keras.layers.Dense(units=300,
                                          input_dim=self.dataClass.rowLength - 1,
                                          activation='relu'))
        self.model.add(keras.layers.Dense(units=200, activation='relu'))
        self.model.add(keras.layers.Dense(units=100, activation='relu'))
        self.model.add(keras.layers.Dense(units=50, activation='relu'))
        self.model.add(keras.layers.Dense(units=1, activation='linear'))

        adamOpt = keras.optimizers.adam(lr=0.001, beta_1=.9, beta_2=0.999, decay=0.0)

        self.model.compile(loss='mean_squared_error',
                           metrics=['mae', 'acc'],
                           optimizer=adamOpt)
        return 0

    def train_batch(self, epochs):

        # model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True))

        self.model.fit(self.X_train, self.Y_train, epochs=epochs, batch_size=512)

        loss_and_metrics = self.model.evaluate(self.X_test, self.Y_test, batch_size=128)

        logging.info("loss and metrics : %s", str(loss_and_metrics))
        return 0

    def train_all(self, batchSize, epochs, testPercentage):
        """

        :param batchSize:
        :param testPercentage:
        :return:
        """

        totalCount = self.dataClass.totalCount

        batchIndex = 0

        while (batchIndex < totalCount):
            try:
                self.getData(batchIndex, batchIndex + batchSize, testPercentage)
                self.train_batch(epochs)

            except Exception as e:
                logging.debug("Error in training model: %s", str(e))
                return 0

            batchIndex += batchSize

        return 0

    def predictOne(self, inputRow):
        """
        prediction for one row only
        :param inputRow:
        :return:
        """

        input = self.dataClass.encode(inputRow)
        output = self.model.predict(input, batch_size=128)
        decodedOutput = self.dataClass.decode(output[0][0])

        # logging.info("prediction : %s", str(decodedOutput))

        return decodedOutput

    def predictMany(self, inputList):
        """

        :param inputList:
        :return:
        """
        encodedinput = np.zeros((len(inputList), self.dataClass.rowLength - 1))

        for i in range(len(inputList)):
            encodedinput[i] = self.dataClass.encode(list(inputList[i]))

        output = self.model.predict(encodedinput, batch_size=128)

        logging.info("encoded Output: %s", str(output))

        decodedOutput = []
        for j in range(len(output)):
            decodedOutput.append(self.dataClass.decode(output[j]))

        return decodedOutput

    def spotTest(self, sampleSize):
        """

        :param sampleSize:
        :return:
        """

        randData = list(self.dataClass.rawDB.getRandomData(sampleSize))
        logging.info("data: %s", str(randData))

        Y = []
        X = []

        for i in range(len(randData)):

            if (randData[i][-1] == ''):
                Y.append(0)
            else:
                Y.append(randData[i][-1])
            X.append(randData[i][1:-1])

        YHat = self.predictMany(X)

        difference = [Y[i] - YHat[i][0] for i in range(len(Y))]

        logging.info("Y: %s", str(Y))
        logging.info("YHat: %s", str(YHat))
        logging.info("difference: %s", str(difference))

        return difference

    def save(self, filename):
        """

        :param filename:
        :return:
        """

        try:
            self.model.save(filename)

            return 0

        except Exception as e:
            logging.debug("Error in saving model data: %s", str(e))
            return 0

    def load(self, filename):
        """

        :param filename:
        :return:
        """

        try:
            self.model = keras.models.load_model(filename)
            return 0

        except Exception as e:
            logging.debug("Error in loading model data: %s", str(e))
            return 0


if __name__ == '__main__':

    try:
        dataTypes = [('day', 'categorical'),
                     ('month', 'categorical'),
                     ('year', 'categorical'),
                     ('carrier', 'categorical'),
                     ('origin', 'categorical'),
                     ('destination', 'categorical'),
                     ('departureTime', 'numeric'),
                     ('delay', 'numeric')]

        m = kerasModel(None, 'FlightStatDB.sqlite', 'FlightStatDataProp.pyd', dataTypes)

        # Train

        # m.create_model()
        # m.train_all(500000, 50, .1)
        # m.save("FlightStatModel.keras")
        # logging.info("saved model", str(m.model))

        # predict
        m.load("FlightStatModel.keras")

        o = m.predictOne([1, 5, 16, 'UA', 'LAX', 'SFO', '2030'])
        logging.info("Prediction: %s", str(o))

        #d = m.spotTest(10)
        # logging.info("difference: %s", str(d))

    except Exception as e:
        logging.debug("Error in data processing: %s", str(e))


