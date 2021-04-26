import pickle
import numpy as np
from os.path import join
from os import listdir
import matplotlib.pyplot as plt
from tqdm import tqdm
import struct as st

class DataLoader:
    def __init__(self,root_dir):
        self.root_dir = root_dir

    def loaddata(self):
        filename = {'images':'train-images.idx3-ubyte',
                    'labels':'train-labels.idx1-ubyte'}

        labels_array =np.array([])

        data_types = {
            0x08: ('ubyte','B',1),
            0x09: ('byte','b',1),
            0x0B: ('>i2','h',2),
            0x0C: ('>i4','i',4),
            0x0D: ('>f4','f',4),
            0x0E: ('>f8','d',8)
        }

        for name in filename.keys():
            if name == 'images':
                imagesfile = open(join(self.root_dir, filename[name]),'rb')
            if name == 'labels':
                labelsfile = open(join(self.root_dir, filename[name]),'rb')
        
        imagesfile.seek(0)

        magic = st.unpack('>4B', imagesfile.read(4))

        if(magic[0] and magic[1])or (magic[2] not in data_types):
            raise ValueError("File Format not correct")

        nDim = magic[3]
        imagesfile.seek(4)
        nImg = st.unpack('>I', imagesfile.read(4))[0]
        nR = st.unpack('>I', imagesfile.read(4))[0]
        nC = st.unpack('>I', imagesfile.read(4))[0]
        nBytes = nImg*nR*nC

        labelsfile.seek(8)

        images_array = 255 - np.asarray(st.unpack('>'+'B'*nBytes,imagesfile.read(nBytes))).reshape((nImg,nR,nC))
        labels_array = np.asarray(st.unpack('>'+'B'*nImg,labelsfile.read(nImg))).reshape((nImg,1))
        labels_array = [l[0] for l in labels_array]
        return images_array.reshape(60000, 28*28), labels_array, None

    def reshape_to_plot(self,data):
        return data.reshape(data.shape[0],28,28).astype("uint8")

    def reshape_to_show(self,data):
        return data.reshape(data.shape[0],20,20).astype("uint8")

    def plot_imgs(self,total_data,n , random =False):
        data = np.array([d for d in total_data])
        data = self.reshape_to_plot(data)
        x1 = min(n//2,5)

        if x1 ==0:
            x1 =1
        y1 = (n//x1)

        x = min(x1,y1)
        y = min(x1,y1)
        fig,ax = plt.subplots(x, y, figsize=(5,5))

        i=0

        for j in range(x):
            for k in range(y):
                if random:
                    i = np.random.choice(range(len(data)))
                ax[j][k].set_axis_off()
                ax[j][k].imshow(data[i:i+1][0], cmap='gray_r')
                i+=1
        plt.show()

    def plot_img(self,data):
        assert data.shape == (28*28)
        data = data.reshape(1,28,28).astype('uint8')
        fig, ax = plt.subplots(figsize=(5,5))
        ax.imshow(data[0])
        plt.show()

