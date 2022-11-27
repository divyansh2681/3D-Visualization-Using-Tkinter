from tkinter import *  
from math import *
import numpy as np
import pandas as pd

"""Extracting data from the object.txt file"""

df = pd.read_csv("object.txt", sep = " ", header=None)
data = df.to_numpy()
verticesDict = {}
edges = []

verticesNum, facesNum = data[0][0].split(',')

A = np.empty([int(verticesNum), 3])
gggg = []
arrVertices = data[1:int(verticesNum)+1, :]
for i in range(int(verticesNum)):
    id, x, y, z = arrVertices[i][0].split(',')
    
    verticesDict[int(id)-1] = (float(x), float(y), float(z))
    A[i] = (float(x), float(y), float(z))

arrFaces = data[-int(facesNum):]
for i in range(len(arrFaces)):

    v1, v2, v3 = arrFaces[i][0].split(',')
    gggg.append([int(v1)-1, int(v2)-1, int(v3)-1])
    if (int(v1)-1, int(v2)-1) not in edges:
        edges.append((int(v1)-1, int(v2)-1))
    if (int(v1)-1, int(v3)-1) not in edges:
        edges.append((int(v1)-1, int(v3)-1))
    if (int(v2)-1, int(v3)-1) not in edges:
        edges.append((int(v2)-1, int(v3)-1))

print(edges)

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


class Shape(MatrixHelpers):

    """
    Class for drawing the shape
    """

    previous_x = 0
    previous_y = 0
    def __init__(self, root, width, height) -> None:
        self.root = root
        self.init_data()
        self.create_canvas(width, height)
        self.draw_shape()
        self.bind_mouse_buttons()
        # self.continually_rotate()
        self.epsilon = lambda d: d * 0.01

    def init_data(self):
        """
        method for initializing the data (A matrix here)
        """
        self.shape = self.transpose_matrix(A)

    def create_canvas(self, width, height):
        """
        method for creating the canvas
        """
        # self.canvas = Canvas(self.root, width = 800, height = 800, background='white')
        self.canvas = Canvas(self.root, width=width, height=height, background='white')
        self.canvas.pack(fill=BOTH, expand=YES)
        # self.canvas.pack()

    def draw_shape(self):
        """
        method for drawing the shape using create_lines 
        """
        w = self.canvas.winfo_width()/2
        h = self.canvas.winfo_height()/2
        self.canvas.delete(ALL)

        scale = h/2
        # scale_h = h/5
        for i in range(len(edges)):
            
            self.canvas.create_line(self.translate_point(scale*self.shape[0][edges[i][0]], scale*self.shape[1][edges[i][0]], w, h), 
            self.translate_point(scale*self.shape[0][edges[i][1]], scale*self.shape[1][edges[i][1]], w, h), fill = 'blue')

            self.canvas.create_oval(self.translate_point(scale*self.shape[0][edges[i][0]], scale*self.shape[1][edges[i][0]], w, h), 
            self.translate_point(scale*self.shape[0][edges[i][0]], scale*self.shape[1][edges[i][0]], w, h), outline='blue', width=10)

            self.canvas.create_oval(self.translate_point(scale*self.shape[0][edges[i][1]], scale*self.shape[1][edges[i][1]], w, h), 
            self.translate_point(scale*self.shape[0][edges[i][1]], scale*self.shape[1][edges[i][1]], w, h), outline='blue', width=10)

    def continually_rotate(self):
        self.shape = self.rotate_along_x(0.01, self.shape)
        self.shape = self.rotate_along_y(0.01, self.shape)
        self.shape = self.rotate_along_z(0.01, self.shape)
        self.draw_shape()
        self.root.after(15, self.continually_rotate)

    def bind_mouse_buttons(self):
        self.canvas.bind("<Button-1>", self.mouseClick)
        self.canvas.bind("<B1-Motion>", self.mouseMotion)

    def mouseClick(self, event):
        self.previous_x = event.x
        self.previous_y = event.y


    def mouseMotion(self, event):
        dy = self.previous_y - event.y 
        dx = self.previous_x - event.x 
        self.shape = self.rotate_along_x(self.epsilon(-dx), self.shape)
        self.shape = self.rotate_along_y(self.epsilon(dy), self.shape)
        self.draw_shape()
        self.mouseClick(event)
        # print(self.shape)

def main():
  root = Tk()
  height = root.winfo_screenheight()
 
# getting screen's width in pixels
  width = root.winfo_screenwidth()
  Shape(root, width, height)
  root.title("Neocis Software Assessment")
  root.mainloop()


if __name__ == '__main__':
  main()