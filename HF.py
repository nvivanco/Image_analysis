"""""
The following is a set of helper functions (HF) that assist in streamlining code
elsewhere
"""""

import tkinter as tk
from tkinter import filedialog


"""
the following function is used to open up a gui window that allows 
the user to select a directory and returns the path of the directory selected 

"""


def select_directory():
    root = tk.Tk()
    root.withdraw()
    folder_path = filedialog.askdirectory(title = "Select experiment")
    if folder_path:
        print("path_selected: " + folder_path)
        return folder_path
    else:
        print("No folder was selected")
        return None
 