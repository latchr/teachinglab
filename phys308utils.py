import matplotlib.pyplot as plt
import pint
import numpy as np
import plotly
import plotly.plotly as py
import plotly.graph_objs as go
from scipy.optimize import curve_fit

plotly.offline.init_notebook_mode(connected=True)

units=pint.UnitRegistry()

def make_figure():
    """Make a matplotlib plot object to be populated with data series

    Returns
    -------
    tuple
        matplotlib figure and axes objects.
    """
    fig, ax = plt.subplots()

    return fig, ax

def show_figure(plot):
    """Wrapper for displaying a matplotlib figure

    Parameters
    ----------
    plot: tuple
        Contains (fig, ax) objects from Matplotlib

    Returns
    -------
    fig
        Matplotlib figure.
    """
    return plot[0]

def load_csv(filepath):
    """Load data from CSV file to a dataset object for plotting and manipulation.

    It is important for the CSV file to have a top row with text headers for each column.
    The CSV file must be comma-separated.

    Parameters
    ----------
    filepath : string
        Path to the desired CSV file

    Returns
    -------
    dataset
        dataset object filled with the data from file.
    """

    headers = np.loadtxt(filepath, dtype='string', delimiter=',')[0]

    data = dataset(len(headers))

    data.set_labels(headers.tolist())

    file_data = np.loadtxt(filepath, skiprows=1, delimiter=',')

    for row in file_data:
        data.add_row(row.tolist())

    return data

class dataset():
    """Laboratory dataset with columns of data.
    """

    def __init__(self, dimension):
        self.data = []
        self.dimension = dimension
        self.labels = ['']*dimension

    def add_row(self, row):
        """Add row to dataset

        Parameters
        ----------
        row : list
            New row to append to end of dataset.
        """
        if len(row) != self.dimension:
            print('Cannot add a row of length {} to a dataset with {} columns'.format(len(row), self.dimension))
        else:
            self.data.append(row)

    def add_col(self, col, label=None):
        """Add column to dataset

        Parameters
        ----------
        col : list
            New column to add to dataset. It must have the same length as the other columns.

        label : string , optional
            Label for the new column. Defaults to None
        """

        if label is None:
            label = ' '
        self.labels.append(label)

        if len(col) != len(self.data):
            print('Cannot add a column of length {} to a dataset with {} rows'.format(len(col), len(self.data)))

        else:
            for i, row in enumerate(self.data):
                row = row.append(col[i])

    def del_row(self, rownumber):
        """ Delete a row from the dataset.

        Parameters
        ----------
        rownumber : int
            The index number of the row to delete
        """
        self.data.pop(rownumber)

    def del_col(self, colnumber):
        """Delete a column from the dataset

        Parameters
        ----------
        colnumber : int
            The index number of the column to delete
        """
        for row in self.data:
            row = row.pop(colnumber)

        self.labels.pop(colnumber)


    def set_labels(self, labels):
        """Set the labels of the columns in dataset

        Parameters
        ----------
        labels : list
            List of strings containing the column labels.
        """
        if len(labels) != self.dimension:
            print("Cannot label {} columns with the provided {} labels".format(self.dimension), len(labels))
        else:
            self.labels = labels

    def get_col(self, col):
        """Get the data in a column of the dataset.

        Parameter
        ---------
        col : string or int
            column label or column index.

        Returns
        -------
        list
            The chosen column from dataset
        """
        if type(col) is str:

            if col not in self.labels:
                print('No data columns with label {}, cannot get column.'.format(col))

                return np.array([0])

            else:
                col_idx = self.labels.index(col)

        else:
            col_idx = col

        # Get column data
        column = [row[col_idx] for row in self.data]

        return np.array(column)

    def get_row(self, rownum):
        """Get the data in a row of the dataset

        Parameters
        ----------
        rownum : int
            Index number of the row

        Returns
        -------
        list
            The row from dataset
        """
        if rownum >= len(self.data):
            print('Dataset is not that long.')
            return np.array([0])

        else:
            return np.array(self.data[rownum])

    def plot(self, horizontal_column, vertical_column, xerr_column=None, yerr_column=None, title=None, figure=None, line=False):
        """Prepare a plot of selected data in dataset.

        Parameters
        ----------
        horizontal_column : int
            index of column to use as horizontal-axis values (independent variable)

        vertical_column : int
            index of column to use as vertical-axis values (dependent variable)

        xerr_column : int , optional
            index of column to use as horizontal-axis errors

        yerr_column : int , optional
            index of column to use as vertical-axis errors

        title : string , optional
            Title for the plot

        figure : tuple , optional
            (fig, ax) object if the plot is to be drawn on an existing figure.

        line : bool , optional
            Draw a line or draw datapoints as markers only. Default to markers only.
        """

        data = np.array(self.data)

        if figure is None:
            fig, ax = plt.subplots()
        else:
            fig, ax = figure

        # Sort data so lines draw correctly
        data = data[data[:, horizontal_column].argsort()]

        if xerr_column is None and yerr_column is None:
            if line:
                ax.plot(data.T[horizontal_column], data.T[vertical_column])
            else:
                ax.plot(data.T[horizontal_column], data.T[vertical_column], '.', linestyle='')

        else:
            if xerr_column is None:
                xerr_data = None
            else:
                xerr_data = data.T[xerr_column]

            if yerr_column is None:
                yerr_data = None
            else:
                yerr_data = data.T[yerr_column]
            ax.errorbar(data.T[horizontal_column],
                        data.T[vertical_column],
                        yerr_data,
                        xerr=xerr_data,
                        fmt='o',
                        linestyle=''
                       )
            if line:
                ax.plot(data.T[horizontal_column], data.T[vertical_column])

        if self.labels[horizontal_column] == '':
            horizontal_label = 'Independent variable (arb units)'
        else:
            horizontal_label = self.labels[horizontal_column]

        ax.set_xlabel(horizontal_label)

        if self.labels[vertical_column] == '':
            vertical_label = 'Independent variable (arb units)'
        else:
            vertical_label = self.labels[vertical_column]

        ax.set_ylabel(vertical_label)

        if title is not None:
            ax.set_title(title)

        if figure is None:
            plt.show()

    def show_table(self):
        """Make a nice readible tabular overview of the dataset.
        """
        table_string = ''

        # Find out the maximum number of digits to display the row count
        num_of_rows = len(self.data)
        rowchars = len(str(num_of_rows))

        ####
        # Labels in first row

        # Pad to leave space for the rowcount
        table_string += ' ' * (rowchars + 2)   # double space between rownum and table

        longest_label = max(self.labels, key=len)

        colwidth = len(longest_label)

        # Leave room for 4-sig-fig scientific notation
        if colwidth < 10:
            colwidth = 10

        # Fill each column label in the string
        for label in self.labels:
            table_string += '| {} '.format(label)
            # Pad to keep display nicely formatted
            table_string += ' '* (colwidth - len(label))

        table_string += '|\n'

        for i, row in enumerate(self.data):
            # Print a row index at start of line
            row_idx_string = '{} '.format(i)
            table_string += row_idx_string + ' ' * (rowchars - len(row_idx_string) + 2)  # double space between rownum and table

            for entry in row:
                entry_txt = '| {:.3E} '.format(float(entry))  # convert to float because cocalc uses sage.rings.real_mpfr.RealLiteral
                table_string += entry_txt

                # Pad
                table_string += ' ' * (colwidth - len(entry_txt) + 3)

            table_string += '|\n'

        print(table_string)

    ## text manipulation method,
    def make_fit(self, func, x_col, y_col):
        """Make a simple fit y(x) to some data.

        This is a convenience wrapper for scipy.optimize

        Parameters
        ----------
        func : function
            Desired fit function

        x_col : int
            Index of column in dataset to use as "x" values

        y_col : int
            Index of column in dataset to use as "y" values
        """

        x_data = self.get_col(x_col)
        y_data = self.get_col(y_col)

        popt, pcov = curve_fit(func, x_data, y_data)

        y_fit = func(x_data, *popt)

        return y_fit, popt
    
    def save(self, filepath):
        """Save the dataset to a csv text file for backup
        
        Parameters
        ----------
        filename : string
            File path (including name) to save data
        """
        savedata = np.array(self.data)
        
        header = ','.join(self.labels)
        
        np.savetxt(filepath, savedata, header=header, delimiter=',', comments='')
