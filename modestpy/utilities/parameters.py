"""
Copyright (c) 2017, University of Southern Denmark
All rights reserved.

This code is licensed under BSD 2-clause license.
See LICENSE file in the project root for license terms.
"""

import pandas as pd


class Parameters:
    """
    Enables to:

    * read csv with parameters
    * assign new values
    * save csv with updated parameters

    """
    def __init__(self, f=None):
        self.pars = pd.DataFrame()
        self.f = None
        if f is not None:
            self.read(f)

    def read(self, f):
        self.pars = pd.read_csv(f)
        self.f = f

    def assign(self, **kwargs):
        """
        Assigns new parameters from **kwargs. Does not automatically save the file.

        :param kwargs: name = value
        :return: None
        """
        for key in kwargs:
            self.pars[key] = [kwargs[key]]

    def update_and_save(self, new_par):
        """
        Updates csv with parameters with new parameters from new_par DataFrame. Automatically saves the file.

        :param new_par: DataFrame
        :return: None
        """
        for key in new_par:
            self.pars[key] = new_par[key]
        self.save()

    def save_template(self, dictionary, f):
        self.f = f
        # Convert scalars to unit vectors
        for key in dictionary:
            dictionary[key] = [dictionary[key]]
        # Generate and save DataFrame
        df = pd.DataFrame.from_dict(dictionary)
        df.to_csv(f, index=False)
        # Update self.pars
        self.pars = pd.read_csv(f)

    def show(self):
        print self.pars

    def save(self, f=None):
        if f:
            self.pars.to_csv(f, index=False)
        else:
            self.pars.to_csv(self.f, index=False)
