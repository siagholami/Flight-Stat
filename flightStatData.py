import h5py
import datetime
import logging
import numpy as np
import pickle
import FlightStatDB

class kerasData:
    def __init__(self, h5Filename,rawDBFilename, dataTypes):
        """
        The data class supporting Flight Stat Model.

        """

        logfilename = "logs/" + str(datetime.date.today()) + ".log"
        logging.basicConfig(filename=logfilename, level=logging.DEBUG, format="%(levelname)s: %(message)s - %(asctime)s")

        try:
            self.h5File = h5Filename
            self.rawDB = FlightStatDB.FlightStatDB(rawDBFilename)
            self.dataTypes = dataTypes
            self.totalCount = self.rawDB.totalSize()

            self.categories = {}
            self.scales = {}
            self.numericIndexes = []

            # lenght of one row of data (input + output)
            self.rowLength = 0

        except Exception as e:
            logging.debug("Error in creating instance: %s", str(e))

    def resetData(self):
        """
        Reset the data pointer 
        :return:
        """

        try:
            a = np.random.random(size=(self.totalCount, self.rowLength))
            h5f = h5py.File(self.h5File, 'w')
            h5f.create_dataset('kerasDataset', data=a, chunks=(10000, self.rowLength))

            h5f.close()
            return 0


        except Exception as e:
            logging.debug("Error in resetting table : %s", str(e))
            return 0



    def getDataProperties(self):
        """
        Initiates self.processedData, gets categories and indexes

        rawDBFilename: sqlite db of raw data
        dataType: [('ColName','date'), ('colName','numeric'), ('colName','categorical'), ...]
        :return:
        """

        nProcessed = 0

        for i in range(len(self.dataTypes)):

            colType = self.dataTypes[i][1]
            colName = self.dataTypes[i][0]

            if colType == "date":
                self.numericIndexes += [nProcessed, nProcessed+1, nProcessed+2]
                nProcessed += 3

            elif colType == "categorical":
                cats = self.rawDB.distinct(colName)
                self.categories[colName] = cats
                nProcessed += len(cats)

            elif colType == "numeric":
                self.numericIndexes.append(nProcessed)
                nProcessed += 1

            else:
                raise("Not supported data type")

        self.rowLength = nProcessed

        return

    def transformRow(self, row):
        """

        :param row: row of data
        :return: row of processed data in numpy format
        """

        pdata = np.zeros((1, self.rowLength))

        j = 0
        for i in range(len(row)):

            colName = self.dataTypes[i][0]
            colType = self.dataTypes[i][1]
            input = row[i]

            if (input == ''):
                input = 0

            if (colType == 'date'):

                day, month, year = self.date_transform(input)
                pdata[0][j] = float(day)
                pdata[0][j+1] = float(month)
                pdata[0][j+2] = float(year)
                j += 3

            elif (colType == 'numeric'):
                pdata[0][j] = float(input)
                j += 1

            elif (colType == 'categorical'):
                categories = self.categories[colName]

                # make dummy var
                plist = [0 for k in categories]
                catIndex = categories.index(input)
                plist[catIndex] = 1

                # copy to output
                for k in plist:
                    pdata[0][j] = k
                    j += 1
            else:
                raise("data type not supported")

        return pdata

    def transformData(self, batchSize):
        """
        Process data and returns a Numpy
        :return: numpy array
        """

        rowIndex = 0

        while (rowIndex<=self.totalCount):

            rawData = self.rawDB.getData(rowIndex, rowIndex+batchSize)

            h5f = h5py.File(self.h5File, 'r+')
            data = h5f['kerasDataset']

            for row in rawData:

                try:
                    # don't need the id
                    pdata = self.transformRow(row[1:])

                    data[rowIndex] = pdata

                    #logging.info("row: %s", str(row))
                    #logging.info("pdata: %s", str(pdata))
                    #logging.info("data[ii] before close: %s", str(data[rowIndex]))
                    #logging.info("data[ii-1]: %s", str(data[ii]))

                except Exception as e:
                    logging.debug("Error in data transformation: %s", str(e))

                rowIndex += 1

            h5f.close()

            logging.info("Processing data: %s th", str(rowIndex))


        return 0

    def getColumn(self, colIndex):
        """

        :param colIndex:
        :return:
        """

        with h5py.File(self.h5File, 'r') as h5f:
            data = h5f['kerasDataset']
            return data[:, colIndex]

    def scaleData(self):
        """
        normalized processed data
        :return:
        """

        for i in self.numericIndexes:

            with h5py.File(self.h5File, 'r+') as h5f:
                data = h5f['kerasDataset']
                col = np.copy(data[:, i])

                range_i = col.max() - col.min()
                mean_i = col.mean()
                self.scales['range'+str(i)] = range_i
                self.scales['mean'+str(i)] = mean_i


                scaledCol = (col-mean_i) / range_i

                #logging.info("col %s; range: %s; mean: %s" % (str(i), str(range_i), str(mean_i)))
                #logging.info("self.scales: %s" % str(self.scales))
                #logging.info("original: %s" % str(col))
                #logging.info("normalized vector: %s" % str(normalizedCol))

                data[:, i] = scaledCol

        return 0


    def normalizeData(self):
        """
        normalized processed data
        :return:
        """

        for i in self.numericIndexes:

            with h5py.File(self.h5File, 'r+') as h5f:
                data = h5f['kerasDataset']
                col = np.copy(data[:, i])

                norm = np.linalg.norm(col)
                self.norms[str(i)] = norm

                normalizedCol = col/norm

                logging.info("col %s; norm: %s" % (str(i), str(norm)))
                logging.info("self.norms: %s" % str(self.norms))
                #logging.info("original: %s" % str(col))
                #logging.info("normalized vector: %s" % str(normalizedCol))

                data[:, i] = normalizedCol

        return 0

    def getData(self, fromIndex, toIndex):
        """

        :param fromIndex: including
        :param toIndex: excluding
        :return:
        """

        with h5py.File(self.h5File, 'r') as h5f:
            data = h5f['kerasDataset']
            return data[fromIndex:toIndex, :]


    def scaleInput(self, input):
        """

        :param input: numpy array shape (n,1)
        :return:
        """
        scaledInput = np.copy(input)

        for i in self.numericIndexes:

            org = scaledInput[0][i]

            range_i = self.scales['range'+str(i)]
            mean_i = self.scales['mean'+str(i)]

            s_i = (org - mean_i)/range_i

            scaledInput[0][i] = s_i

        return scaledInput

    def normalizeInput(self, input):
        """

        :param input: numpy array shape (n,1)
        :return:
        """
        normalizedInput = np.copy(input)

        for i in self.numericIndexes:
            org = normalizedInput[0][i]

            norm = self.norms[str(i)]
            normalized_i = org / norm

            normalizedInput[0][i] = normalized_i

        return normalizedInput

    def cleanupData(self):
        """

        :return:
        """

        delayIndex = self.rowLength - 1

        with h5py.File(self.h5File, 'r+') as h5f:
            data = h5f['kerasDataset']

            for i in range(self.totalCount):
                delay = data[i, delayIndex]

                if (delay < -15):
                    data[i, delayIndex] = 0

                elif (delay > 100):
                    data[i, delayIndex] = 100

                if (i%10000==0):
                    logging.info("Cleaning up data: %s th", str(i))

        return 0

    def encode(self, input):
        """
        encodes the row for NN input
        :param input:
        :return:
        """

        # there is no output - so just adding a dummy output at the last position
        modifiedInput = input + ['0']
        transformedInput = self.transformRow(modifiedInput)

        #scale input
        scaledInput = self.scaleInput(transformedInput)

        #taking the dummy last variable out
        output = scaledInput[:, :-1]

        return output

    def decode(self, input):
        """
        decodes the output of the NN
        :param input:
        :return:
        """

        # output is at last position
        outputIndex = self.rowLength - 1
        range_i = self.scales['range'+str(outputIndex)]
        mean_i = self.scales['mean' + str(outputIndex)]

        output = input * range_i + mean_i

        return output

    def save(self, filename):
        """

        :param filename:
        :return:
        """
        try:
            file = open(filename, mode='wb')

            picklableDic = {}

            picklableDic['dataTypes'] = self.dataTypes
            picklableDic['categories'] = self.categories
            picklableDic['numericIndexes'] = self.numericIndexes
            picklableDic['scales'] = self.scales
            picklableDic['rowLength'] = self.rowLength

            pickledDic = pickle.dump(picklableDic, file)
            file.close()
            return 0

        except Exception as e:
            logging.debug("Error in saving data: %s", str(e))
            return 0

    def load(self, filename):
        """

        :param filename:
        :return:
        """
        try:
            file = open(filename, mode='rb')

            picklableDic = pickle.load(file)

            self.dataTypes = picklableDic['dataTypes']
            self.categories = picklableDic['categories']
            self.numericIndexes = picklableDic['numericIndexes']
            self.scales = picklableDic['scales']
            self.rowLength = picklableDic['rowLength']

            file.close()
            return 0

        except Exception as e:
            logging.debug("Error in loading data: %s", str(e))
            return 0


if __name__ == '__main__':

    dataTypes = [('day','categorical'),
                 ('month','categorical'),
                 ('year','categorical'),
                 ('carrier','categorical'),
                 ('origin', 'categorical'),
                 ('destination', 'categorical'),
                 ('departureTime','numeric'),
                 ('delay', 'numeric')]
    try:
        d = kerasData('kerasData2.h5', 'FlightStatDB.sqlite', dataTypes=dataTypes)

        #d.getDataProperties()
        #d.save('FlightStatData.hdf5')
        #d.resetData()
        #d.transformData(10000)

        #d.load('FlightStatData.hdf5')
        #d.cleanupData()
        #d.scaleData()
        #d.save('FlightStatData.hdf5')


        d.load('FlightStatData.hdf5')

        r = [5, 1, 16, 'HA', 'LAX', 'HNL', 830]
        e = d.encode(r)
        f = d.decode(.9)
        logging.info("e: %s, f: %s" % (str(e), str(f)))

    except Exception as e:
        logging.debug("Error in data processing: %s", str(e))




