from tkinter import *  
import pandas as pd
from utils_p2 import *

def main():
    """
    main function where file is read and passed to the ExtractData class and then object for Shape class is created
    """
    root = Tk()
    height = root.winfo_screenheight()
    width = root.winfo_screenwidth()
    fileName = "object.txt"
    dataframe = pd.read_csv(fileName, sep = " ", header=None)
    data = dataframe.to_numpy()
    A, edges, facesList = ExtractData.extract(data)
    Shape(root, width, height, A, edges, facesList)
    root.title("Neocis Software Assessment - Part 2")
    root.mainloop()


if __name__ == '__main__':
  main()