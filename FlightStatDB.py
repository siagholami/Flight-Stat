import sqlite3
import datetime
import logging
import csv

class FlightStatDB:
    def __init__(self, DBFilename):
        """
        SQL db class.
        
        :param DBFilename: 
        """

        logFilename = "logs/" + str(datetime.date.today()) + ".log"
        logging.basicConfig(filename=logFilename, level=logging.DEBUG, format="%(levelname)s: %(message)s - %(asctime)s")

        try:
            self.dbConnection = sqlite3.connect(DBFilename)

        except Exception as e:
            logging.debug("Error in connection %d: %s", e)

    def initDB(self):
        """
        Creates the database tables and initialize tables
        
        :return: 
        """

        # create tables
        self.createFlightsTable()

        return

    def createFlightsTable(self):
        """
        Creates flights table
        :return: 
        """

        cursor = self.dbConnection.cursor()

        try:
            # Create Stocks table
            cursor.execute('''CREATE TABLE IF NOT EXISTS flights (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            day INTEGER NOT NULL,
                            month INTEGER NOT NULL,
                            year INTEGER NOT NULL,
                            carrier TEXT NOT NULL,
                            origin TEXT NOT NULL,
                            destination TEXT NOT NULL,
                            time INTEGER NOT NULL,
                            delay INTEGER NOT NULL)''')

            # Save (commit) the changes
            self.dbConnection.commit()
            return 0


        except Exception as e:
            self.dbConnection.rollback()
            logging.debug("Error in creating table : %s", str(e))
            return 0

    def date_transform(self, inputDate):
        """
        returns day, month and year of the inputDate
        :param inputDate:
        :return:
        """

        d = inputDate.split("/")

        return d[1], d[0], d[2]

    def addFlight(self, date, carrier, origin, destination, time, delay):
        """
        Adds a new flight
        :param date:
        :param carrier:
        :param origin:
        :param destination:
        :param time:
        :param delay:
        :return:
        """

        d, m, y = self.date_transform(date)

        if (delay == ''):
            delay = 0

        cursor = self.dbConnection.cursor()
        #logging.info("FlightStatDB: artitleTitle type: %s", type(articleTitle))

        try:
            # insert
            sql = '''INSERT INTO flights(day, month, year, carrier, origin, destination, time, delay) VALUES (?, ? , ?, ?, ?, ?, ?, ? )'''
            # print(sql)
            cursor.execute(sql, (d, m, y, carrier, origin, destination, time, delay))
            self.dbConnection.commit()
            # print("Entered: " + symbol + " , " + name)
            return 0

        except Exception as e:
            self.dbConnection.rollback()
            logging.debug(
                '''FlightDB: Error in adding flight; error: %s''', str(e))
            return 0

    def addFlightsFromFile(self, filename):
        """
        Adding flights from file.
        :param filename:
        :return:
        """

        with open(filename, mode='r') as csvFile:
            reader = csv.DictReader(csvFile)

            i = 0

            for row in reader:
                self.addFlight(row['FL_DATE'], row['UNIQUE_CARRIER'], row['ORIGIN'],row['DEST'],row['CRS_DEP_TIME'],row['DEP_DELAY'])

                i += 1
                # logging.info('Added Stock: %s' % row['Symbol'])
                if (i % 10000 == 0):
                    logging.info("reading data: %s th", str(i))

        csvFile.close()
        return

    def querySQL(self, sql):
        """
        print the result of this SQL statement.
        :param sql:
        :return:
        """

        cursor = self.dbConnection.cursor()
        cursor.execute(sql)
        for row in cursor:
            print(row)

        return

    def distinct(self, field):
        """
        print the result of this SQL statement.
        :param field:
        :return:
        """

        cursor = self.dbConnection.cursor()

        sql = '''select distinct(%s) from flights''' % field

        #print(sql)
        cursor.execute(sql)

        distinct_value = []
        for row in cursor:
            distinct_value.append(row[0])

        return distinct_value

    def totalSize(self):
        """
        Returns the total number of data points.
        :return:
        """
        cursor = self.dbConnection.cursor()

        sql = '''select count(*) from flights'''

        cursor.execute(sql)

        row = cursor.fetchone()

        return row[0]

    def getData(self, fromIndex, toIndex):
        """

        :param fromIndex: and including
        :param toIndex: and including
        :return:
        """

        cursor = self.dbConnection.cursor()

        sql = '''SELECT * FROM flights WHERE id BETWEEN ? AND ? '''
        # print(sql)
        cursor.execute(sql, (fromIndex, toIndex))

        return cursor.fetchall()

    def getRandomData(self, numberOfRows):
        """

        :param numberOfRows:
        :return:
        """

        cursor = self.dbConnection.cursor()

        sql = '''SELECT * FROM flights ORDER BY RANDOM() LIMIT ? '''
        # print(sql)
        cursor.execute(sql, (numberOfRows, ))

        return cursor.fetchall()

    def queryToCSV(self,sql, headers, filename):
        """

        :param sql:
        :param filename:
        :return:
        """

        cursor = self.dbConnection.cursor()

        cursor.execute(sql)

        with open(filename, 'wb') as myfile:
            wr = csv.writer(myfile)

            if headers != 0:
                wr.writerow(headers)

            for row in cursor:

                wr.writerow(row)

        return 0


if __name__ == '__main__':

    fdb = FlightStatDB("FlightStatDB.sqlite")

    '''fdb.initDB()

    for i in range(2,14):
        fdb.addFlightsFromFile(str(i) + '.csv')'''

    f = fdb.getData(0,5)
    t = fdb.totalSize()

    #f = fdb.getRandomData(20)
    #print(f)

    '''fdb.queryToCSV("select date, carrier, origin, destination, departureTime, delay from flights where date LIKE '1/%'",
                   ['month', 'day', 'carrier', 'origin', 'destination', 'departureTime', 'delay'], 'w1.csv')
    fdb.queryToCSV("select date, carrier, origin, destination, departureTime, delay from flights where date LIKE '2/%'",
                   0, 'w2.csv')
    fdb.queryToCSV("select date, carrier, origin, destination, departureTime, delay from flights where date LIKE '3/%'",
                   0, 'w3.csv')
    fdb.queryToCSV("select date, carrier, origin, destination, departureTime, delay from flights where date LIKE '4/%'",
                   0, 'w4.csv')
    fdb.queryToCSV("select date, carrier, origin, destination, departureTime, delay from flights where date LIKE '5/%'",
                   0, 'w5.csv')
    fdb.queryToCSV("select date, carrier, origin, destination, departureTime, delay from flights where date LIKE '6/%'",
                   0, 'w6.csv')
    fdb.queryToCSV("select date, carrier, origin, destination, departureTime, delay from flights where date LIKE '7/%'",
                   0, 'w7.csv')
    fdb.queryToCSV("select date, carrier, origin, destination, departureTime, delay from flights where date LIKE '8/%'",
                   0, 'w8.csv')
    fdb.queryToCSV("select date, carrier, origin, destination, departureTime, delay from flights where date LIKE '9/%'",
                   0, 'w9.csv')
    fdb.queryToCSV("select date, carrier, origin, destination, departureTime, delay from flights where date LIKE '10/%'",
                   0, 'w10.csv')
    fdb.queryToCSV("select date, carrier, origin, destination, departureTime, delay from flights where date LIKE '11/%'",
                   0, 'w11.csv')
    fdb.queryToCSV("select date, carrier, origin, destination, departureTime, delay from flights where date LIKE '12/%'",
                   0, 'w12.csv')'''

    headers = ['id', 'day', 'month', 'year', 'carrier', 'origin',
               'destination', 'departureTime', 'delay']

    sql = "select distinct(carrier) from flights"

    fdb.queryToCSV(sql , 0,'result.csv')

    print("The end")
