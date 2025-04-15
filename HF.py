import tkinter as tk
from tkinter import filedialog


def select_directory():
    root = tk.Tk()
    root.withdraw()
    folder_path = filedialog.askdirectory(title = "Select experiment")

    if folder_path:
        print("path_selected")
        print(folder_path)
        return folder_path
    else:
        print("No folder was selected")
        return None