import numpy as np
from .. import cytomine
from threading import Thread, RLock

class SampleReader:

    def __init__(self, filename, cytomine_host, cytomine_public_key, cytomine_private_key, base_path = '/api/', working_path = '/tmp/',type="binary",square_size=1000):
        """
        Parameters
        ----------
        filename : the file name in which the coordonates will be read.
        type : the type of file that is read can be either 'binary' or 'text'
        square_size : the square side size (in pixel)
        """
        self.resetSample()
        self.filename = filename
        self.cytomine_host = cytomine_host
        self.cytomine_public_key = cytomine_public_key
        self.cytomine_private_key = cytomine_private_key
        self.base_path = base_path
        self.working_path = working_path
        if square_size <= 0:
            self.square_size = 1
        else:
            self.square_size = square_size

        self.dataCoordonate = {}
        if type == "text":
            self.readFromTextFile()
        elif type == "binary":
            self.readFromBinaryFile()
        else:
            raise ValueError("error: expected \"binary\" or \"text\"")

    def readFromTextFile(self):
        file = open(self.filename, "r")
        try:
            #readlines
            lines = file.readlines()
            rlist = {}

            beginning = True
            for line in lines:
                splits = [int(val) for val in line.rstrip('\n\r').split(" ")]
                if len(splits) == 4: #rectangle
                    rlist[tuple(splits)] = True
                elif len(splits) == 2: #single point
                    self.dataCoordonate[tuple(splits)] = True
                elif beginning and len(splits) == 3: #image group, image width and height at the beginning of the file
                    beginning = False
                    self.imageGroup = splits[0]
                    self.imageWidth = splits[1]
                    self.imageHeight = splits[2]
            self.dataCoordonate.update(rectangleToPoint(rlist))
        finally:
            file.close()

    def readFromBinaryFile(self):
        import struct as st

        file = open(self.filename, "rb")
        try:
            header = file.read(1) #file type (1 byte) and
            if len(header) != 1:
                raise IOError("uncorrectly formated file (header)")
            header = st.Struct("<B").unpack(header)

            blocksize = 0
            if header[0] == 0: #other type could be implemented here
                header = file.read(8) # image group (4 bytes), image width (2 bytes),image height (2 bytes)
                if len(header) != 8:
                    raise IOError("uncorrectly formated file (header)")
                header = st.Struct("<IHH").unpack(header)
                self.imageGroup = header[0]
                self.imageWidth = header[1]
                self.imageHeight = header[2]
                s = st.Struct("<HH")
                blocksize = 4
            else:
                raise IOError("unsupported file type")

            while True:
                record = file.read(blocksize)
                if len(record) != blocksize:
                    if len(record) != 0:
                        raise IOError("uncorrectly formated file (body): block of {} bytes expected.".format(blocksize))
                    break;
                self.dataCoordonate[tuple(s.unpack(record))] = True

        finally:
            file.close()

    def writeToBinaryFile(self, filename, type=0):
        import struct as st
        file = open(filename, "wb")
        type = 0
        try:
            s = None
            if type == 0: #other type could be implemented here
                #file type (1 byte), image group (4 bytes), image width (2 bytes),image height (2 bytes)
                file.write(st.Struct("<BIHH").pack(type,self.imageGroup,self.imageWidth,self.imageHeight))
                s = st.Struct("<HH")
                blocksize = 4
            else:
                raise IOError("unsupported file type")

            for point in self.dataCoordonate:
                file.write(s.pack(*point))


        finally:
            file.close()
    def test(self):
        t = {}
        k = {(0,0): True}

        for i in range(10):
            conn = cytomine.Cytomine(self.cytomine_host, self.cytomine_public_key, self.cytomine_private_key, base_path = self.base_path, working_path = self.working_path, verbose= False)
            t[i] = Asker(self.imageGroup,k,conn)
            t[i].start()
        for i in range(10):
            t[i].join()
        print("fini")



    def sampleF(self, thread=1):
        """
        Returns
        -------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The input data.
        y : array-like, shape (n_samples,)
            The target values.
        """
        import random as rd
        import sys
        import time
        if thread <= 0:
            thread = 1

        size = len(self.dataCoordonate)
        Y = np.zeros(2*size)
        Y[:size] = 1
        i = 0
        first = True
        nThread = 0
        threads = {}
        splitlist = {}
        conn = {}
        try:
            for point in self.dataCoordonate:
                if i >= size - ((thread-nThread-1)*(size/thread)):
                    conn[nThread] = cytomine.Cytomine(self.cytomine_host, self.cytomine_public_key, self.cytomine_private_key, base_path = self.base_path, working_path = self.working_path, verbose= False)
                    threads[nThread] = Asker(self.imageGroup,splitlist,conn[nThread])
                    threads[nThread].start()
                    nThread += 1
                    splitlist = {}
                splitlist[point] = True
                i += 1

            conn[nThread] = cytomine.Cytomine(self.cytomine_host, self.cytomine_public_key, self.cytomine_private_key, base_path = self.base_path, working_path = self.working_path, verbose= False)
            threads[nThread] = Asker(self.imageGroup,splitlist,conn[nThread])
            threads[nThread].start()
            nThread += 1
            nppoint = 0

            for i in range(thread):
                while threads[i].is_alive():
                    count = 0
                    for j in range(thread):
                        with threads[j].lock:
                            sys.stdout.write("\rCount {}: {}             ".format(j,threads[j].count))
                            sys.stdout.flush()
                            count += threads[j].count
                        time.sleep(1)
                    sys.stdout.write("\rWaiting {} / {}".format(count,size))
                    sys.stdout.flush()
                    time.sleep(1)
                if first:
                    X = np.zeros((size,threads[i].X.shape[1]),dtype=np.int)
                    first = False
                for j in threads[i].X[:]:
                    X[nppoint] = j
                    nppoint += 1
        except:
            for i in range(thread):
                with threads[i].lock:
                    threads[i].stop = True
            raise
        sys.stdout.write("\rFinished {} / {}\n".format(size,size))
        sys.stdout.flush()


        nThread = 0
        threads = {}
        splitlist = {}
        n = 0
        try:
            while n < size:
                x = rd.randint(0,self.imageWidth - 1)
                y = rd.randint(0,self.imageHeight - 1)
                for k in range(x,min(x + rd.randint(10,30),self.imageWidth - 1)):
                    if n >= 2*size:
                        break;
                    for l in range(y,min(y + rd.randint(10,30),self.imageHeight - 1)):
                        if n >= 2*size:
                            break;
                        if not tuple((k,l)) in self.dataCoordonate:
                            if n >= (size) - ((thread-nThread-1)*(size/thread)):
                                threads[nThread] = Asker(self.imageGroup,splitlist,conn[nThread])
                                threads[nThread].start()
                                nThread += 1
                                splitlist = {}
                            splitlist[tuple((k,l))] = True
                            n += 1
            if nThread < thread:
                threads[nThread] = Asker(self.imageGroup,splitlist,conn[nThread])
                threads[nThread].start()
                nThread += 1
            #Getting the point from the asker

            for i in range(thread):

                while threads[i].is_alive():
                    count = 0
                    for j in range(thread):
                        with threads[j].lock:
                            sys.stdout.write("\rCount {}: {}             ".format(j,threads[j].count))
                            sys.stdout.flush()
                            count += threads[j].count
                        time.sleep(1)
                    sys.stdout.write("\rWaiting {} / {}".format(count,size))
                    sys.stdout.flush()
                    time.sleep(1)

                for j in threads[i].X[:]:
                    X[nppoint] = j
                    nppoint += 1
        except:
            for i in range(nThread):
                with threads[i].lock:
                    threads[i].stop = True
            raise
        sys.stdout.write("\rFinished {} / {}\n".format(size,size))
        sys.stdout.flush()
        return X,Y

    def resetSample(self):
        self.indexX = 0
        self.indexY = 0

    def sample(self, rectangle=False):
        """
        Parameters
        ----------
        rectangle: True: use the request rectangle in the Cytomine Server

        Returns
        -------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The input data.
        y : array-like, shape (n_samples,)
            The target values.
        """
        if self.indexY == -1:
            return None

        newindexX = self.indexX + self.square_size
        newindexY = self.indexY
        if newindexX >= self.imageWidth:
            rectangeWidth = self.imageWidth - self.indexX
            #we pass to the next line in the next call (if any)
            newindexX = 0
            newindexY = self.indexY + self.square_size
            if newindexY >= self.imageHeight:
                newindexY = -1

        else:
            rectangeWidth = self.square_size

        if self.indexY + self.square_size >= self.imageHeight:
            rectangeHeight = self.imageHeight - self.indexY
        else:
            rectangeHeight = self.square_size
        X = []
        Y = np.zeros(rectangeHeight*rectangeWidth)
        for i in range(self.indexY,self.indexY+rectangeHeight):
            for j in range(self.indexX,self.indexX+rectangeWidth):
                Y[i*rectangeHeight+j] = 1 if tuple((i,j)) in self.dataCoordonate else 0

        if rectangle:
            #make the rectangle request here
            pass
        else:
            #make pxl requests here
            pass
        #update the current indexes
        self.indexX = newindexX
        self.indexY  = newindexY
        return X, Y

class Asker(Thread):
    def __init__(self, imageGroup, dataCoordonate,conn,rectangle=False):
        Thread.__init__(self)
        self.imageGroup = imageGroup
        self.dataCoordonate = dataCoordonate
        self.conn = conn
        self.rectangle = rectangle
        self.lock = RLock()
        self.count = 0
        self.stop = False

    def run(self):
        i = 0
        first = True
        for point in self.dataCoordonate:
            with self.lock:
                if self.stop:
                    return

            if first:
                spectre = self.conn.get_pixel_spectre(self.imageGroup,point[0],point[1])
                if hasattr(spectre, "spectra"):
                    self.X = np.zeros((len(self.dataCoordonate),len(spectre.spectra)),dtype=np.int)
                    self.X[i] = spectre.spectra
                    first = False
                else:
                    print(point)

            else:
                spectre = self.conn.get_pixel_spectre(self.imageGroup,point[0],point[1])
                if hasattr(spectre, "spectra"):
                    self.X[i] = spectre.spectra
                else:
                    print(point)
            i += 1

            with self.lock:
                self.count = i

def rectangleToPoint(listOfRectangle):
    list = {}
    for rect in listOfRectangle:
        for i in range(rect[0],rect[0]+rect[2]):
            for j in range(rect[1],rect[1]+rect[3]):
                list[tuple((i,j))] = True
    return list
