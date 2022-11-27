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

arrVertices = data[1:int(verticesNum)+1, :]
for i in range(int(verticesNum)):
    id, x, y, z = arrVertices[i][0].split(',')
    verticesDict[int(id)-1] = (float(x), float(y), float(z))
    A[i] = (float(x), float(y), float(z))

gggg = []

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
    


print(gggg)

"""Drawing the shape"""

class MatrixHelpers():
    def transpose_matrix(self, matrix):
        return np.transpose(matrix)

    def translate_point(self, x, y, dx, dy):
        return x+dx, y+dy

    def matrix_multiply(self, matrix_a, matrix_b):
        return np.dot(matrix_a, matrix_b)

    def rotate_along_x(self, x, shape):
        return self.matrix_multiply([[1, 0, 0],
                                    [0, cos(x), -sin(x)], 
                                    [0, sin(x), cos(x)]], shape)

    def rotate_along_y(self, y, shape):
        return self.matrix_multiply([[cos(y), 0, sin(y)], 
                                    [0, 1, 0], 
                                    [-sin(y), 0, cos(y)]], shape)
    
    def rotate_along_z(self, z, shape):
        return self.matrix_multiply([[cos(z), sin(z), 0],
                                    [-sin(z), cos(z), 0], 
                                    [0, 0, 1]], shape)


class Shape(MatrixHelpers):

    previous_x = 0
    previous_y = 0
    def __init__(self, root) -> None:
        self.root = root
        self.init_data()
        self.create_canvas()
        self.draw_shape()
        self.bind_mouse_buttons()
        # self.continually_rotate()
        self.epsilon = lambda d: d * 0.01

    def init_data(self):
        self.shape = self.transpose_matrix(A)

    def create_canvas(self):
        self.canvas = Canvas(self.root, width = 800, height = 800, background='white')
        self.canvas.pack(fill=BOTH, expand=YES)

    def _compute_colors(self, start, end, limit, factor):
        (r1,g1,b1) = self.canvas.winfo_rgb(start)
        (r2,g2,b2) = self.canvas.winfo_rgb(end)
        r_ = int((r1 + (r2-r1) * (limit/factor)))
        g_ = int((g1+(g2-g1) * (limit/factor)))
        b_ = int((b1 + (b2-b1) * (limit/factor)))

        # colors = []
        # for i in range(limit):
        #     nr = int(r1 + (r_ratio * i))
        #     ng = int(g1 + (g_ratio * i))
        #     nb = int(b1 + (b_ratio * i))
        color = "#%4.4x%4.4x%4.4x" % (r_,g_,b_)
            # colors.append(color)
        return color

    def draw_shape(self):
        w = self.canvas.winfo_width()/2
        h = self.canvas.winfo_height()/2
        self.canvas.delete(ALL)
                
        for i in range(len(gggg)):
            

            x_1 = self.shape[0][gggg[i][0]]
            y_1 = self.shape[1][gggg[i][0]]
            z_1 = self.shape[2][gggg[i][0]]

            a = np.array([x_1, y_1, z_1])

            x_2 = self.shape[0][gggg[i][1]]
            y_2 = self.shape[1][gggg[i][1]]
            z_2 = self.shape[2][gggg[i][1]]

            b = np.array([x_2, y_2, z_2])

            x_3 = self.shape[0][gggg[i][2]]
            y_3 = self.shape[1][gggg[i][2]]
            z_3 = self.shape[2][gggg[i][2]]

            c = np.array([x_3, y_3, z_3])

            z_unit = np.array([0, 0, 1])

            vect = np.cross((a-c),(a-b))
            factor = 1000

            val = np.linalg.norm(np.cross(vect/np.linalg.norm(vect),z_unit))*factor
            # print(val)
            # val = np.linalg.norm(np.dot(vect,z_unit))*100

            clr = self._compute_colors("#00005F", "#0000FF", val, factor)

            # self.canvas.create_polygon(self.translate_point(100*self.shape[0][gggg[i][0]], 100*self.shape[1][gggg[i][0]], w, h), 
            # self.translate_point(100*self.shape[0][gggg[i][1]], 100*self.shape[1][gggg[i][1]], w, h), 
            # self.translate_point(100*self.shape[0][gggg[i][2]], 100*self.shape[1][gggg[i][2]], w, h), fill=clr)
            self.canvas.create_polygon(self.translate_point(100*x_1, 100*y_1, w, h), 
            self.translate_point(100*x_2, 100*y_2, w, h), 
            self.translate_point(100*x_3, 100*y_3, w, h), fill=clr)

            self.canvas.create_oval(self.translate_point(100*self.shape[0][edges[i][0]], 100*self.shape[1][edges[i][0]], w, h), 
            self.translate_point(100*self.shape[0][edges[i][0]], 100*self.shape[1][edges[i][0]], w, h), outline='blue', width=10)

            self.canvas.create_oval(self.translate_point(100*self.shape[0][edges[i][1]], 100*self.shape[1][edges[i][1]], w, h), 
            self.translate_point(100*self.shape[0][edges[i][1]], 100*self.shape[1][edges[i][1]], w, h), outline='blue', width=10)


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
        dx = self.previous_y - event.y 
        dy = self.previous_x - event.x 
        self.shape = self.rotate_along_x(self.epsilon(-dx), self.shape)
        self.shape = self.rotate_along_y(self.epsilon(dy), self.shape)
        self.draw_shape()
        self.mouseClick(event)

def main():
  root = Tk()
  Shape(root)
  root.mainloop()


if __name__ == '__main__':
  main()