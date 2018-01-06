# -*- coding: utf-8 -*-

"""
Copyright (c) 2017, University of Southern Denmark
All rights reserved.

This code is licensed under BSD 2-clause license.
See LICENSE file in the project root for license terms.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from Tkinter import *
from tkFileDialog import askopenfilename
import tkMessageBox

class App(Frame):
    def __init__(self, master=None):
        Frame.__init__(self, master)
        self.grid()
        self.create_widgets()

    def create_widgets(self):
        # Top menu
        menu = Menu(self)
        self.master.config(menu=menu)

        # File menu
        filemenu = Menu(menu)
        menu.add_cascade(label="File", menu=filemenu)
        filemenu.add_command(label="Load configuration", command=self.load_config)
        filemenu.add_command(label="Save configuration", command=self.save_config)
        filemenu.add_separator()
        filemenu.add_command(label="Exit", command=self.master.quit)

        # Run menu
        runmenu = Menu(menu)
        menu.add_cascade(label="Run", menu=runmenu)
        runmenu.add_command(label="Run estimation", command=self.run_estimation)

        # Help menu
        helpmenu = Menu(menu)
        menu.add_cascade(label="Help", menu=helpmenu)
        helpmenu.add_command(label="Help", command=self.help)
        helpmenu.add_command(label="About", command=self.about)

    def load_config(self):
        print("Nothing here yet")

    def save_config(self):
        print("Nothing here yet")

    def help(self):
        help_window = Tk()
        help_window.title("Help")
        msg = \
"""
Report a problem:
https://github.com/sdu-cfei/modest-py/issues

Contact the author:
krza@mmmi.sdu.dk

Contribute:
https://github.com/sdu-cfei/modest-py
"""
        label = Label(help_window, text=msg)
        label.pack(ipadx=75, ipady=25)

    def about(self):
        msg = \
"""
ModestPy is an FMI-compliant system identification framework.

Copyright (c) 2017, University of Southern Denmark
All rights reserved.

License: BSD 2-clause
"""
        tkMessageBox.showinfo("About", msg)

    def run_estimation(self):
        print("Nothing here yet")


app = App()
app.master.minsize(800, 500)
app.master.title("ModestPy GUI")
app.master.mainloop()
