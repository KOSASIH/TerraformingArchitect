import tkinter as tk
from tkinter import ttk

def select_celestial_body():
    selected_body = body_combobox.get()
    print(f"Selected celestial body: {selected_body}")

root = tk.Tk()
root.title("Celestial Body Selection Tool")

# Create labels and comboboxes for each criteria
size_label = ttk.Label(root, text="Size:")
size_label.grid(row=0, column=0, padx=10, pady=5, sticky=tk.W)
size_combobox = ttk.Combobox(root, values=["Small", "Medium", "Large"])
size_combobox.grid(row=0, column=1, padx=10, pady=5, sticky=tk.W)

composition_label = ttk.Label(root, text="Composition:")
composition_label.grid(row=1, column=0, padx=10, pady=5, sticky=tk.W)
composition_combobox = ttk.Combobox(root, values=["Rocky", "Gaseous", "Mixed"])
composition_combobox.grid(row=1, column=1, padx=10, pady=5, sticky=tk.W)

distance_label = ttk.Label(root, text="Distance from Sun:")
distance_label.grid(row=2, column=0, padx=10, pady=5, sticky=tk.W)
distance_combobox = ttk.Combobox(root, values=["Close", "Moderate", "Far"])
distance_combobox.grid(row=2, column=1, padx=10, pady=5, sticky=tk.W)

# Create a button to select the celestial body
select_button = ttk.Button(root, text="Select", command=select_celestial_body)
select_button.grid(row=3, column=0, columnspan=2, padx=10, pady=10)

root.mainloop()
