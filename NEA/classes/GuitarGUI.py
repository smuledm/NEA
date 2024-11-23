import tkinter as tk
from Constants import *



def create_tablature():
    # Create the main window
    root = tk.Tk()
    root.title("Guitar Tab GUI")

    # Define font and dimensions
    tab_font = ("Courier", 16)  # Monospaced font for alignment
    tab_width = 2200  # Canvas width
    tab_height = 1080  # Canvas height

    # Create a canvas to display the tablature
    canvas = tk.Canvas(root, width=tab_width, height=tab_height, bg="white")
    canvas.pack()

    # Draw the six horizontal lines for the strings
    string_spacing = 154  # Vertical spacing between strings
    start_y = 154  # Starting y-coordinate for the first string
    for i in range(6):  # Six strings
        canvas.create_line(50, start_y + i * string_spacing, tab_width - 50, start_y + i * string_spacing, width=2)

    # Add labels for the strings (e, B, G, D, A, E)
    string_labels = ["e", "B", "G", "D", "A", "E"]
    for i, label in enumerate(string_labels):
        canvas.create_text(5, start_y + i * string_spacing, text=label, font=("Arial", 14), anchor="w")

    # Run the main loop
    root.mainloop()


# Run the tablature GUI
create_tablature()

        