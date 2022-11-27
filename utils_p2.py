from tkinter import *  
from math import *
import numpy as np

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
        -------------------------
        returns the transposed matrix
        """
        return np.transpose(matrix)

    def translate_point(self, x, y, dx, dy):
        """
        Parameters
        -------------------------
        x: any real number (int/float/double)
            x coordinate of the point that is to be translated
        y: any real number (int/float/double)
            y coordinate of the point that is to be translated
        dx: any real number (int/float/double)
            distance with which x has to be translated
        dy: any real number (int/float/double)
            distance with which y has to be translated
        -------------------------
        returns the translated points
        """
        return x+dx, y+dy

    def matrix_multiply(self, matrix_a, matrix_b):
        """
        Parameters
        --------------------------
        matrix_a: numpy array of size (n, p)
        matrix_b: numpy array of size (p, m)
        --------------------------
        returns the product of the given matrices
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
        --------------------------
        returns the rotated matrix
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
        --------------------------
        returns the rotated matrix
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
        --------------------------
        returns the rotated matrix
        """
        return self.matrix_multiply([[cos(z), sin(z), 0],
                                    [-sin(z), cos(z), 0], 
                                    [0, 0, 1]], shape)

class ExtractData():
    """
    Class for extracting the data
    """
    def extract(data):
        """
        brief: manipulating the input data (numpy here) for extracting the vertices, edges and faces
        --------------------------
        returns - A: matrixc containing all the vertices in the format shown below)
                    [x1 x2 x3 x4 x5 x6
                    y1 y2 y3 y4 y5 y6
                    z1 z2 z3 z4 z5 z6]
                  edges: list containing all the edges of the shape in the format shown below
                    [(v1, v2), (v1, v3), .....]
                    v is the vertex number, so this list has tuples containing vertices for an edge
                  facesList: list containing all the edges of the shape in the format shown below
                    [(v1, v2, v3), (v1, v3, v5), .....]
                    v is the vertex number, so this list has tuples containing vertices for a face
        """
        verticesNum, facesNum = data[0][0].split(',')
        edges = []
        facesList = []
        A = np.empty([int(verticesNum), 3])
        vertices = data[1:int(verticesNum)+1, :]
        faces = data[-int(facesNum):]

        for i in range(int(verticesNum)):
            id, x, y, z = vertices[i][0].split(',')
            A[i] = (float(x), float(y), float(z))

        for i in range(len(faces)):

            v1, v2, v3 = faces[i][0].split(',')
            facesList.append([int(v1)-1, int(v2)-1, int(v3)-1])
            if (int(v1)-1, int(v2)-1) not in edges:
                edges.append((int(v1)-1, int(v2)-1))
            if (int(v1)-1, int(v3)-1) not in edges:
                edges.append((int(v1)-1, int(v3)-1))
            if (int(v2)-1, int(v3)-1) not in edges:
                edges.append((int(v2)-1, int(v3)-1))
        
        return A, edges, facesList

class Shape(MatrixHelpers):
    """
    Class for drawing the shape
    """
    previous_x = 0
    previous_y = 0
    def __init__(self, root, width, height, A, edges, facesList) -> None:
        self.root = root
        self.init_data(A)
        self.create_canvas(width, height)
        self.draw_shape(edges, facesList)
        self.bind_mouse_buttons(edges, facesList)
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
        self.canvas.pack()

    def compute_colors(self, start, end, limit, factor):
        """
        method for returning a color within a given range 
        --------------------------
        Parameters
        --------------------------
        start: initial value of the color in the range 
        end: final value of the color in the range
        limit: multiplier determining the color
        factor: number of parts in which the range will be divided 

        when (limit/factor) is 0, initial color will the returned
        when (limit/factor) is 1, final color will the returned
        """
        (r1,g1,b1) = self.canvas.winfo_rgb(start)
        (r2,g2,b2) = self.canvas.winfo_rgb(end)
        r_ = int((r1 + (r2-r1) * (limit/factor)))
        g_ = int((g1 + (g2-g1) * (limit/factor)))
        b_ = int((b1 + (b2-b1) * (limit/factor)))
        color = "#%4.4x%4.4x%4.4x" % (r_,g_,b_)
        return color

    def draw_shape(self, edges, facesList):
        """
        method for drawing the shape using create_polygon and drawing vertices using create_oval
        this method also colors the face depending on the angle of the face with the z axis
        --------------------------
        Parameters
        --------------------------
        edges: list containing all the edges of the shape in the format shown below
        [(v1, v2), (v1, v3), .....]
        v is the vertex number, so this list has tuples containing vertices for an edge
        facesList: list containing all the edges of the shape in the format shown below
        [(v1, v2, v3), (v1, v3, v5), .....]
        v is the vertex number, so this list has tuples containing vertices for a face 
        """
        w = self.canvas.winfo_width()/2
        h = self.canvas.winfo_height()/2
        self.canvas.delete(ALL)
        scale = h/2
                
        for i in range(len(facesList)):
            x_1 = self.shape[0][facesList[i][0]]
            y_1 = self.shape[1][facesList[i][0]]
            z_1 = self.shape[2][facesList[i][0]]

            a = np.array([x_1, y_1, z_1])

            x_2 = self.shape[0][facesList[i][1]]
            y_2 = self.shape[1][facesList[i][1]]
            z_2 = self.shape[2][facesList[i][1]]

            b = np.array([x_2, y_2, z_2])

            x_3 = self.shape[0][facesList[i][2]]
            y_3 = self.shape[1][facesList[i][2]]
            z_3 = self.shape[2][facesList[i][2]]

            c = np.array([x_3, y_3, z_3])

            z_unit = np.array([0, 0, 1])
            vect = np.cross((a-c),(a-b))
            factor = 10
            val = np.linalg.norm(np.cross(vect/np.linalg.norm(vect),z_unit))*factor
            clr = self.compute_colors("#00005F", "#0000FF", val, factor)

            self.canvas.create_polygon(self.translate_point(scale*x_1, scale*y_1, w, h), 
            self.translate_point(scale*x_2, scale*y_2, w, h), 
            self.translate_point(scale*x_3, scale*y_3, w, h), fill=clr)

            self.canvas.create_oval(self.translate_point(scale*self.shape[0][edges[i][0]], scale*self.shape[1][edges[i][0]], w, h), 
            self.translate_point(scale*self.shape[0][edges[i][0]], scale*self.shape[1][edges[i][0]], w, h), outline='blue', width=10)

            self.canvas.create_oval(self.translate_point(scale*self.shape[0][edges[i][1]], scale*self.shape[1][edges[i][1]], w, h), 
            self.translate_point(scale*self.shape[0][edges[i][1]], scale*self.shape[1][edges[i][1]], w, h), outline='blue', width=10)

    def bind_mouse_buttons(self, edges, facesList):
        """
        method for binding mouse events to mouse event functions (defined below)
        --------------------------
        Parameters
        --------------------------
        edges: list containing all the edges of the shape in the format shown below
        [(v1, v2), (v1, v3), .....]
        v is the vertex number, so this list has tuples containing vertices for an edge
        facesList: list containing all the edges of the shape in the format shown below
        [(v1, v2, v3), (v1, v3, v5), .....]
        v is the vertex number, so this list has tuples containing vertices for a face 
        """
        self.canvas.bind("<Button-1>", self.mouseClick)
        self.canvas.bind("<B1-Motion>", lambda event, arg1 = edges, arg2 = facesList: self.mouseMotion(event, arg1, arg2))

    def mouseClick(self, event):
        """
        mouse event function for getting the coordinates of the current event (mouse click here)
        --------------------------
        Parameters
        --------------------------
        event: any event, mouse click here
        """
        self.previous_x = event.x
        self.previous_y = event.y

    def mouseMotion(self, event, edges, facesList):
        """
        mouse event function for creating a rotated matrix and the new shape according to the rotated matrix
        --------------------------
        Parameters
        --------------------------
        event: any event, mouse motion here
        edges: list containing all the edges of the shape in the format shown below
        [(v1, v2), (v1, v3), .....]
        v is the vertex number, so this list has tuples containing vertices for an edge
        facesList: list containing all the edges of the shape in the format shown below
        [(v1, v2, v3), (v1, v3, v5), .....]
        v is the vertex number, so this list has tuples containing vertices for a face 
        """
        dx = self.previous_y - event.y 
        dy = self.previous_x - event.x 
        self.shape = self.rotate_along_x(self.epsilon(-dx), self.shape)
        self.shape = self.rotate_along_y(self.epsilon(dy), self.shape)
        self.draw_shape(edges, facesList)
        self.mouseClick(event)