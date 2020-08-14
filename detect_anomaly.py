import numpy as np
import pandas as pd
from scipy.misc import imread,imresize,imsave
from sklearn.cluster import KMeans
from collections import Counter
from sklearn.decomposition import PCA

class Anomaly:
    first_image_path=None
    second_image_path=None
    block_size = 5
    return_diff=False
    def __init__(self,first_image_path,second_image_path,block_size=5,return_diff=False ):
        if(first_image_path==None):
            print("\n Error: one or more image path missing")
        elif(second_image_path==None):
            print("\n Error: one or more image path missing")
        self.first_image_path = first_image_path
        self.second_image_path = second_image_path
        self.block_size = block_size
        self.return_diff = return_diff
    def difference_image(self,image1_path, image2_path):
        first_read_image = imread(image1_path)
        second_read_image = imread(image2_path)
        shape = (np.asarray(first_read_image.shape) / self.block_size).astype(int) * self.block_size
        first_image = imresize(first_read_image, shape).astype(np.int16)
        second_image = imresize(second_read_image,shape).astype(np.int16)
        return abs(first_image-second_image),shape
    def image_to_vectors(self,block_size,diff_image,shape):
        row_pointer=0
        column_pointer=0
        feature_count = 0
        vectors =  np.zeros((int(shape[0] * shape[1] / (block_size*block_size)),(block_size*block_size)))
        while(feature_count<vectors.shape[0]):
            while(row_pointer < shape[0]):
                while(column_pointer < shape[1]):
                    vectors[feature_count,:]=(diff_image[row_pointer:row_pointer+block_size,column_pointer:column_pointer+block_size].ravel())
                    column_pointer = column_pointer + block_size
                row_pointer = row_pointer + block_size
                column_pointer = 0
            feature_count +=1
        return vectors
    def normalisation(self,vectors):
        mean = np.mean(vectors, axis = 0)
        return vectors - mean, mean
    def generate_eigen_vector_space(self,vectors):
        pca = PCA()
        pca.fit(vectors)
        eigen_vector_space = pca.components_
        return eigen_vector_space
    def generate_vector_space(self,block_size,shape,diff_image):
        h = (block_size - 1) // 2
        row_pointer=h
        column_pointer=h
        vectors = []
        while(row_pointer < shape[0] - h):
            while(column_pointer < shape[1] - h):
                vectors.append(diff_image[row_pointer-h:row_pointer+(block_size-h),column_pointer-h:column_pointer+(block_size-h)].flatten())
                column_pointer = column_pointer + 1
            row_pointer = row_pointer + 1
            column_pointer = h
        return vectors
    def generate_feature_vector_space(self,eigen_space, vector_space,mean):
        projection = np.dot(np.asarray(vector_space), eigen_space)
        return projection - mean
    def identify_change(self,features, vector_space):
        kmeans = KMeans(2, verbose = 0)
        kmeans.fit(features)
        clusters = kmeans.predict(features)
        cnt  = Counter(clusters)
        return clusters,cnt
    def generate_map(self,clusters,cnt,shape):
        maps = np.reshape(clusters,(shape[0]-(self.block_size-1),shape[1]-(self.block_size-1)))
        index = min(cnt, key = cnt.get)
        maps[maps == index] = 255
        maps[maps != 255] = 0
        '''for i in range(maps.shape[0]):
            for j in range(maps.shape[1]):
                if maps[i][j]== cnt:
                    maps[i][j]==255
                else:
                    maps[i][j]==0'''
        return maps.astype(np.uint8)
    def detect(self):
        diff_image,shape = self.difference_image(self.first_image_path,self.second_image_path)
        #print(diff_image)
        vector_space = self.image_to_vectors(self.block_size,diff_image,shape)
        #print(len(vector_space))
        normalised , mean = self.normalisation(vector_space)
        eigen_space = self.generate_eigen_vector_space(normalised)
        new_vector_space = self.generate_vector_space(self.block_size,shape,diff_image)
        feature_vector_space = self.generate_feature_vector_space(eigen_space,new_vector_space,mean)
        cluster, count = self.identify_change( feature_vector_space , new_vector_space)
        maps = self.generate_map(cluster,count,shape)
        returns=[]
        returns.append(maps)
        if(self.return_diff=True):
            returns.append(diff_image)
        else:
            returns.append(None)
        return returns