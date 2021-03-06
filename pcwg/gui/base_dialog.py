# -*- coding: utf-8 -*-
"""
Created on Wed Aug 10 01:28:28 2016

@author: Stuart
"""
import tk_simple_dialog
import Tkinter as tk
import tkFileDialog
import tkMessageBox

import datetime
import dateutil
import os.path

from date_pick import Calendar
from ..configuration.preferences_configuration import Preferences

import validation

from ..exceptions.handling import ExceptionHandler
from ..core.status import Status

columnSeparator = "|"
filterSeparator = "#"

datePickerFormat = "%Y-%m-%d %H:%M"# "%d-%m-%Y %H:%M"
datePickerFormatDisplay = "[dd-mm-yyyy hh:mm]"

def convertDateToText(date):
    return date.strftime(datePickerFormat)

def getDateFromText(text):
    if len(text) > 0:
        return datetime.datetime.strptime(text, datePickerFormat)
    else:
        return None

def getDateFromEntry(entry):
    return getDateFromText(entry.get())

def getBoolFromText(text):
    if text == "True":
        return True
    elif text == "False":
        return False
    else:
        raise Exception("Cannot convert Text to Boolean: %s" % text)

def intSafe(text, valueIfBlank = 0):
    try:
        return int(text)
    except:
        return valueIfBlank
        
def floatSafe(text, valueIfBlank = 0.):
    try:
        return float(text)
    except:
        return valueIfBlank

class VariableEntry:

    def __init__(self, variable, entry, tip):
        self.variable = variable
        self.entry = entry
        self.pickButton = None
        self.tip = tip
    
    def clearTip(self):
        self.setTip("")
    
    def setTipNotRequired(self):
        self.setTip("Not Required")
    
    def setTip(self, text):
        if self.tip != None:
            self.tip['text'] = text
    
    def get(self):
        return self.variable.get()
    
    def set(self, value):
        return self.variable.set(value)
    
    def configure(self, state):
        self.entry.configure(state = state)
        if self.pickButton != None:
            self.pickButton.configure(state = state)
    
    def bindPickButton(self, pickButton):
        self.pickButton = pickButton
  

class ListBoxEntry(VariableEntry):
    
    def __init__(self, listbox, scrollbar, tip):
        self.scrollbar = scrollbar
        self.listbox = listbox
        self.tip = tip
                            
    def error(self):
        raise Exception("Not possible with listbox object")    
        
    def get(self):
        self.error()
        
    def set(self, value):
        self.error()
        
    def configure(self, state):
        self.error()
        
    def bindPickButton(self, pickButton):
        self.error()
              
class SetFileSaveAsCommand:

    def __init__(self, master, variable):
        self.master = master
        self.variable = variable

    def __call__(self):
        fileName = tkFileDialog.asksaveasfilename(parent=self.master,defaultextension=".xml")
        if len(fileName) > 0: self.variable.set(fileName)
        
class ClearEntry:

    def __init__(self, entry):
        self.entry = entry
    
    def __call__(self):
        self.entry.set("")
                
class BaseDialog(tk_simple_dialog.Dialog):

    def __init__(self, master):

        self.titleColumn = 0
        self.labelColumn = 1
        self.inputColumn = 2
        self.buttonColumn = 3
        self.secondButtonColumn = 4
        self.tipColumn = 5
        self.messageColumn = 6
        
        self.validations = []

        self.row = 0
        self.listboxEntries = {}
        
        self.master = master

        tk_simple_dialog.Dialog.__init__(self, master)
        
    def prepareColumns(self, master):

        master.columnconfigure(self.titleColumn, pad=10, weight = 0)
        master.columnconfigure(self.labelColumn, pad=10, weight = 0)
        master.columnconfigure(self.inputColumn, pad=10, weight = 1)
        master.columnconfigure(self.buttonColumn, pad=10, weight = 0)
        master.columnconfigure(self.secondButtonColumn, pad=10, weight = 0)
        master.columnconfigure(self.tipColumn, pad=10, weight = 0)
        master.columnconfigure(self.messageColumn, pad=10, weight = 0)

    def addDatePickerEntry(self, master, title, validation, value, width = None):

        if value != None:
            if type(value) == str:
                textValue = value
            else:
                textValue = value.strftime(datePickerFormat)
        else:
            textValue = None
                
        entry = self.addEntry(master, title + " " + datePickerFormatDisplay, validation, textValue, width = width)
        entry.entry.config(state=tk.DISABLED)
        
        pickButton = tk.Button(master, text="...", command = DatePicker(self, entry, datePickerFormat), width=3, height=1)
        pickButton.grid(row=(self.row-1), sticky=tk.N, column=self.inputColumn, padx = 160)
                
        entry.bindPickButton(pickButton)

        return entry
                
    def addPickerEntry(self, master, title, validation, value, width = None):

        entry = self.addEntry(master, title, validation, value, width = width)
        pickButton = tk.Button(master, text=".", command = ColumnPicker(self, entry), width=5, height=1)
        pickButton.grid(row=(self.row-1), sticky=tk.E+tk.N, column=self.buttonColumn)

        entry.bindPickButton(pickButton)

        return entry
        
    def addOption(self, master, title, options, value):

        label = tk.Label(master, text=title)
        label.grid(row=self.row, sticky=tk.W, column=self.labelColumn)

        variable = tk.StringVar(master, value)

        option = apply(tk.OptionMenu, (master, variable) + tuple(options))
        option.grid(row=self.row, column=self.inputColumn, sticky=tk.W)

        self.row += 1

        return variable
                
    def addListBox(self, master, title, height = 3):
            
        scrollbar =  tk.Scrollbar(master, orient=tk.VERTICAL)
        tipLabel = tk.Label(master, text="")
        tipLabel.grid(row = self.row, sticky=tk.W, column=self.tipColumn)                
        lb = tk.Listbox(master, yscrollcommand=scrollbar, selectmode=tk.EXTENDED, height=height)  
        
        self.listboxEntries[title] = ListBoxEntry(lb,scrollbar,tipLabel)
        self.row += 1

        self.listboxEntries[title].scrollbar.configure(command=self.listboxEntries[title].listbox.yview)
        self.listboxEntries[title].scrollbar.grid(row=self.row, sticky=tk.W+tk.N+tk.S, column=self.titleColumn)
        return self.listboxEntries[title]

    def addCheckBox(self, master, title, value):

        label = tk.Label(master, text=title)
        label.grid(row=self.row, sticky=tk.W, column=self.labelColumn)
        variable = tk.IntVar(master, value)

        checkButton = tk.Checkbutton(master, variable=variable)
        checkButton.grid(row=self.row, column=self.inputColumn, sticky=tk.W)

        self.row += 1

        return variable

    def addTitleRow(self, master, title):

        tk.Label(master, text=title).grid(row=self.row, sticky=tk.W, column=self.titleColumn, columnspan = 2)

        #add dummy label to stop form shrinking when validation messages hidden
        tk.Label(master, text = " " * 70).grid(row=self.row, sticky=tk.W, column=self.messageColumn)

        self.row += 1

    def addEntry(self, master, title, validation, value, width = None, read_only = False):

        variable = tk.StringVar(master, value)

        label = tk.Label(master, text=title)
        label.grid(row = self.row, sticky=tk.W, column=self.labelColumn)

        tipLabel = tk.Label(master, text="")
        tipLabel.grid(row = self.row, sticky=tk.W, column=self.tipColumn)

        if validation != None:
            validation.messageLabel.grid(row = self.row, sticky=tk.W, column=self.messageColumn)
            validation.title = title
            self.validations.append(validation)
            validationCommand = validation.CMD
        else:
            validationCommand = None

        entry = tk.Entry(master, textvariable=variable, validate = 'key', validatecommand = validationCommand, width = width)

        if read_only:
            entry.config(state=tk.DISABLED)

        entry.grid(row=self.row, column=self.inputColumn, sticky=tk.W)

        if validation != None:
            validation.link(entry)

        self.row += 1

        return VariableEntry(variable, entry, tipLabel)

    def addFileSaveAsEntry(self, master, title, validation, value, width = 60):

        variable = self.addEntry(master, title, validation, value, width)

        button = tk.Button(master, text="...", command = SetFileSaveAsCommand(master, variable), height=1)
        button.grid(row=(self.row - 1), sticky=tk.E+tk.W, column=self.buttonColumn)

        return variable

    def addFileOpenEntry(self, master, title, validation, value, basePathVariable = None, width = 60):

        variable = self.addEntry(master, title, validation, value, width)

        button = tk.Button(master, text="...", command = SetFileOpenCommand(master, variable, basePathVariable), height=1)
        button.grid(row=(self.row - 1), sticky=tk.E+tk.W, column=self.buttonColumn)

        return variable

    def validate(self):

        valid = True
        message = ""

        for item in self.validations:
                
            if not item.valid:
                if not isinstance(item, validation.ValidateDatasets):
                    message += "%s (%s)\r" % (item.title, item.messageLabel['text'])
                else:
                    message += "Datasets error. \r"
                valid = False
                
        if not valid:

            tkMessageBox.showwarning(
                    "Validation errors",
                    "Illegal values, please review error messages and try again:\r%s" % message
                    )
                    
            return 0

        else:

            return 1

class SetFileOpenCommand:

    def __init__(self, master, variable, basePathVariable = None):
        self.master = master
        self.variable = variable
        self.basePathVariable = basePathVariable

    def __call__(self):

        if self.basePathVariable != None:
            initial_folder = os.path.dirname(self.basePathVariable.get())
        else:
            initial_folder = None
            
        fileName = tkFileDialog.askopenfilename(parent=self.master, initialdir=initial_folder)
        
        if len(fileName) > 0:
            self.variable.set(fileName)

class ColumnPicker:

    def __init__(self, parentDialog, entry):

        self.parentDialog = parentDialog
        self.entry = entry

    def __call__(self):
        self.parentDialog.ShowColumnPicker(self.parentDialog, self.pick, self.entry.get())

    def pick(self, column):
            
        if len(column) > 0:
            self.entry.set(column)

class DateFormatPickerDialog(BaseDialog):

    def __init__(self, master, callback, availableFormats, selectedFormat):

        self.callback = callback
        self.availableFormats = availableFormats
        self.selectedFormat = selectedFormat
        
        BaseDialog.__init__(self, master)
                    
    def body(self, master):

        self.prepareColumns(master)     
                
        self.dateFormat = self.addOption(master, "Select Date Format:", self.availableFormats, self.selectedFormat)

    def apply(self):
                    
        self.callback(self.dateFormat.get())
                
class DateFormatPicker:

    def __init__(self, parentDialog, entry, availableFormats):

        self.parentDialog = parentDialog
        self.entry = entry
        self.availableFormats = availableFormats

    def __call__(self):
                    
        try:                                
            DateFormatPickerDialog(self.parentDialog, self.pick, self.availableFormats, self.entry.get())
        except Exception as e:
            ExceptionHandler.add(e, "ERROR picking dateFormat")

    def pick(self, column):
            
        if len(column) > 0:
            self.entry.set(column)
                        
                        
class ColumnSeparatorDialog(BaseDialog):

    def __init__(self, master, callback, availableSeparators, selectedSeparator):

        self.callback = callback
        self.availableSeparators = availableSeparators
        self.selectedSeparator = selectedSeparator
        
        BaseDialog.__init__(self, master)
                    
    def body(self, master):

        self.prepareColumns(master)     
                
        self.separator = self.addOption(master, "Select Column Separator:", self.availableSeparators, self.selectedSeparator)

    def apply(self):
                    
        self.callback(self.separator.get())
                
class ColumnSeparatorPicker:

    def __init__(self, parentDialog, entry, availableSeparators):

        self.parentDialog = parentDialog
        self.entry = entry
        self.availableSeparators = availableSeparators

    def __call__(self):
                    
        try:                                
            ColumnSeparatorDialog(self.parentDialog, self.pick, self.availableSeparators, self.entry.get())
        except Exception as e:
            ExceptionHandler.add(e, "ERROR picking separator")

    def pick(self, column):
            
        if len(column) > 0:
            self.entry.set(column)

class ColumnPickerDialog(BaseDialog):

    def __init__(self, master, callback, availableColumns, column):

        self.callback = callback
        self.availableColumns = availableColumns
        self.column = column
        
        BaseDialog.__init__(self, master)
                    
    def body(self, master):

        self.prepareColumns(master)     

        if len(self.availableColumns) > 0:
            self.column = self.addOption(master, "Select Column:", self.availableColumns, self.column)
                
    def apply(self):
                    
        self.callback(self.column.get())

class DatePicker:

    def __init__(self, parentDialog, entry, dateFormat):

        self.parentDialog = parentDialog
        self.entry = entry
        self.dateFormat = dateFormat
    
    def __call__(self):

        if len(self.entry.get()) > 0:
            date = datetime.datetime.strptime(self.entry.get(), self.dateFormat)
        else:
            date = None

        calendar = Calendar(self.parentDialog, selected_date=date, date_format = self.dateFormat)

        if calendar.is_ok:
            if not calendar.selected_date is None:
                self.entry.set(calendar.selected_date.strftime(self.dateFormat))                  
            else:
                self.entry.set("")

class BaseConfigurationDialog(BaseDialog):

    def __init__(self, master, callback, config, index = None):

        self.index = index
        self.callback = callback
        
        self.isSaved = False
        self.isNew = config.isNew

        self.config = config
                
        if not self.isNew:
                self.originalPath = config.path
        else:
                self.originalPath = None
   
        BaseDialog.__init__(self, master)
                
    def body(self, master):

        self.prepareColumns(master)         

        if self.config.isNew:
                path = None
        else:
                path = self.config.path
                
        self.addFilePath(master, path)

        self.addFormElements(master, path)

    def addFilePath(self, master, path):
        self.addTitleRow(master, "General Settings:")    
        self.filePath = self.addFileSaveAsEntry(master, "Configuration XML File Path:", validation.ValidateDatasetFilePath(master), path)

    def getInitialFileName(self):
        return "Config"
    
    def getInitialFolder(self):
        preferences = Preferences.get()
        return preferences.analysis_last_opened_dir()
    
    def validate_file_path(self):

        if len(self.filePath.get()) < 1:
            path = tkFileDialog.asksaveasfilename(parent=self.master,defaultextension=".xml", initialfile="%s.xml" % self.getInitialFileName(), title="Save New Config", initialdir=self.getInitialFolder())
            self.filePath.set(path)
            
        if len(self.filePath.get()) < 1:
            
            tkMessageBox.showwarning(
                    "File path not specified",
                    "A file save path has not been specified, please try again or hit cancel to exit without saving.")
                
            return 0
            
        if self.originalPath != None and self.filePath.get() != self.originalPath and os.path.isfile(self.filePath.get()):                        
            result = tkMessageBox.askokcancel(
            "File Overwrite Confirmation",
            "Specified file path already exists, do you wish to overwrite?")
            if not result: return 0

        return 1

    def validate(self):

        if BaseDialog.validate(self) == 0: return

        return self.validate_file_path()
    
    def save_config(self):
        self.config.save()

    def execute_callback(self):        
        if self.callback != None:
            if self.index == None:
                self.callback(self.config.path)
            else:
                self.callback(self.config.path, self.index)

    def set_file_path(self):
        self.config.path = self.filePath.get()

    def apply(self):
            
        try:
            
            self.set_file_path()

            self.setConfigValues()                
            
            self.save_config()
            
            self.isSaved = True

            if self.isNew:
                Status.add("Config created")
            else:
                Status.add("Config updated")

            self.execute_callback()

        except ExceptionHandler.ExceptionType as e:
            ExceptionHandler.add(e, "Cannot create/update config")
