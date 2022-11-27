from tkinter import *  
from math import *
import numpy as np
import pandas as pd

# """Extracting data from the object.txt file"""

# df = pd.read_csv("object.txt", sep = " ", header=None)
# data = df.to_numpy()
# verticesDict = {}
# edges = []

# verticesNum, facesNum = data[0][0].split(',')

# A = np.empty([int(verticesNum), 3])
# gggg = []
# arrVertices = data[1:int(verticesNum)+1, :]
# for i in range(int(verticesNum)):
#     id, x, y, z = arrVertices[i][0].split(',')
    
#     verticesDict[int(id)-1] = (float(x), float(y), float(z))
#     A[i] = (float(x), float(y), float(z))

# arrFaces = data[-int(facesNum):]
# for i in range(len(arrFaces)):

#     v1, v2, v3 = arrFaces[i][0].split(',')
#     gggg.append([int(v1)-1, int(v2)-1, int(v3)-1])
#     if (int(v1)-1, int(v2)-1) not in edges:
#         edges.append((int(v1)-1, int(v2)-1))
#     if (int(v1)-1, int(v3)-1) not in edges:
#         edges.append((int(v1)-1, int(v3)-1))
#     if (int(v2)-1, int(v3)-1) not in edges:
#         edges.append((int(v2)-1, int(v3)-1))

# print(edges)

class MatrixHelpers():

    """
    Class for defining matrtix helper functions
    """
    
    def transpose_matrix(self, matrix):
        """
        Parameters
        -------------------------
        matrix: numpy array
            the matrix for which transpose is needed
        """
        return np.transpose(matrix)

    def translate_point(self, x, y, dx, dy):
        """
        Parameters
        -------------------------
        x: any real number (int/float/double)
        y: any real number (int/float/double)
        dx: any real number (int/float/double)
        dy: any real number (int/float/double)
        """
        return x+dx, y+dy

    def matrix_multiply(self, matrix_a, matrix_b):
        """
        Parameters
        --------------------------
        matrix_a: numpy array of size (n, p)
        matrix_b: numpy array of size (p, m)
        """
        return np.dot(matrix_a, matrix_b)

    def rotate_along_x(self, x, shape):
        """
        Parameters
        --------------------------
        x: any real number (int/float/double)
            angle of rotation along x axis
        shape: numpy array
            input matrix which has to be rotated
        """
        return self.matrix_multiply([[1, 0, 0],
                                    [0, cos(x), -sin(x)], 
                                    [0, sin(x), cos(x)]], shape)

    def rotate_along_y(self, y, shape):
        """
        Parameters
        --------------------------
        y: any real number (int/float/double)
            angle of rotation along y axis
        shape: numpy array
            input matrix which has to be rotated
        """
        return self.matrix_multiply([[cos(y), 0, sin(y)], 
                                    [0, 1, 0], 
                                    [-sin(y), 0, cos(y)]], shape)
    
    def rotate_along_z(self, z, shape):
        """
        Parameters
        --------------------------
        z: any real number (int/float/double)
            angle of rotation along z axis
        shape: numpy array
            input matrix which has to be rotated
        """
        return self.matrix_multiply([[cos(z), sin(z), 0],
                                    [-sin(z), cos(z), 0], 
                                    [0, 0, 1]], shape)

class ExtractData():
    # def __init__(self, data) -> None:
    #     # self.data = data
    #     self.A, self.edges, self.facesList = self.extract(data)

    def extract(self, data):
        verticesNum, facesNum = data[0][0].split(',')
        self.edges = []
        self.facesList = []
        self.A = np.empty([int(verticesNum), 3])
        self.vertices = data[1:int(verticesNum)+1, :]
        self.faces = data[-int(facesNum):]

        for i in range(int(verticesNum)):
            id, x, y, z = self.vertices[i][0].split(',')
            self.A[i] = (float(x), float(y), float(z))

        for i in range(len(self.faces)):

            v1, v2, v3 = self.faces[i][0].split(',')
            self.facesList.append([int(v1)-1, int(v2)-1, int(v3)-1])
            if (int(v1)-1, int(v2)-1) not in self.edges:
                self.edges.append((int(v1)-1, int(v2)-1))
            if (int(v1)-1, int(v3)-1) not in self.edges:
                self.edges.append((int(v1)-1, int(v3)-1))
            if (int(v2)-1, int(v3)-1) not in self.edges:
                self.edges.append((int(v2)-1, int(v3)-1))
        
        return self.A, self.edges, self.facesList


class Shape(MatrixHelpers):

    """
    Class for drawing the shape
    """

    previous_x = 0
    previous_y = 0
    def __init__(self, root, width, height, A, edges) -> None:
        self.root = root
        self.init_data(A)
        self.create_canvas(width, height)
        self.draw_shape(edges)
        self.bind_mouse_buttons(edges)
        # self.continually_rotate()
        self.epsilon = lambda d: d * 0.01

    def init_data(self, A):
        """
        method for initializing the data (A matrix here)
        """
        self.shape = self.transpose_matrix(A)

    def create_canvas(self, width, height):
        """
        method for creating the canvas
        """
        self.canvas = Canvas(self.root, width=width, height=height, background='white')
        self.canvas.pack(fill=BOTH, expand=YES)

    def draw_shape(self, edges):
        """
        method for drawing the shape using create_lines 
        """
        w = self.canvas.winfo_width()/2
        h = self.canvas.winfo_height()/2
        self.canvas.delete(ALL)

        scale = h/2
        for i in range(len(edges)):
            
            self.canvas.create_line(self.translate_point(scale*self.shape[0][edges[i][0]], scale*self.shape[1][edges[i][0]], w, h), 
            self.translate_point(scale*self.shape[0][edges[i][1]], scale*self.shape[1][edges[i][1]], w, h), fill = 'blue')

            self.canvas.create_oval(self.translate_point(scale*self.shape[0][edges[i][0]], scale*self.shape[1][edges[i][0]], w, h), 
            self.translate_point(scale*self.shape[0][edges[i][0]], scale*self.shape[1][edges[i][0]], w, h), outline='blue', width=10)

            self.canvas.create_oval(self.translate_point(scale*self.shape[0][edges[i][1]], scale*self.shape[1][edges[i][1]], w, h), 
            self.translate_point(scale*self.shape[0][edges[i][1]], scale*self.shape[1][edges[i][1]], w, h), outline='blue', width=10)

    # def continually_rotate(self):
    #     self.shape = self.rotate_along_x(0.01, self.shape)
    #     self.shape = self.rotate_along_y(0.01, self.shape)
    #     self.shape = self.rotate_along_z(0.01, self.shape)
    #     self.draw_shape()
    #     self.root.after(15, self.continually_rotate)

    # def bind_mouse_buttons(self):
    #     self.canvas.bind("<Button-1>", self.mouseClick)
    #     self.canvas.bind("<B1-Motion>", self.mouseMotion)

    def bind_mouse_buttons(self, edges):
        self.canvas.bind("<Button-1>", self.mouseClick)
        self.canvas.bind("<B1-Motion>", lambda event, arg = edges: self.mouseMotion(event, arg))

    def mouseClick(self, event):
        self.previous_x = event.x
        self.previous_y = event.y


    def mouseMotion(self, event, edges):
        dy = self.previous_y - event.y 
        dx = self.previous_x - event.x 
        self.shape = self.rotate_along_x(self.epsilon(-dx), self.shape)
        self.shape = self.rotate_along_y(self.epsilon(dy), self.shape)
        self.draw_shape(edges)
        self.mouseClick(event)


def main():

    root = Tk()
    height = root.winfo_screenheight()
    width = root.winfo_screenwidth()
    fileName = "object.txt"
    


    df = pd.read_csv(fileName, sep = " ", header=None)
    data = df.to_numpy()
    A, edges, facesList = ExtractData.extract(data)
    print(A)
    # Shape(root, width, height, A, edges)
    # root.title("Neocis Software Assessment")
    # root.mainloop()


if __name__ == '__main__':
  main()