from time import localtime, strftime
import os
import time
import shutil
import numpy as np
import csv

def log_with_time(logmsg):
    logt = time.time()
    print("[{:.4f}] {}".format(logt, logmsg) )
    
def create_folder(directory):
  try:
      if not os.path.exists(directory):
          os.makedirs(directory)
  except OSError:
      print ('Error: Creating directory. ' +  directory)

def delete_and_create_folder(directory):
  try:
      if not os.path.exists(directory):
        os.makedirs(directory)
      else:
        print("path exists : delete all dataset")
        shutil.rmtree(directory)
        os.makedirs(directory)      
      print("directory made @ " + directory)
  except OSError:
      print ('Error: Creating directory. ' +  directory)
      
def savedata(inputdata, filename):
    if type(inputdata) is float or type(inputdata) is int:
        filename = f"{filename}.txt"
        save_value(inputdata, filename)
    elif type(inputdata) is list:
        if type(inputdata[0]) is list:
            filename = f"{filename}.txt"
            for row in inputdata:
                save_list_in_row(row, filename)
        elif type(inputdata[0]) is dict:
            filename = f"{filename}.csv"            
            save_dict(inputdata, filename)
        else:
            for row in inputdata:
                save_value(row, filename)

def save_value(val_input, filename):
    with open(filename, 'a') as f:
      f.write(str(val_input))
      f.write('\n')
      
def save_list_in_row(val_input, filename):
    with open(filename, 'a') as f:
        n = len(val_input)
        i = 0
        for i in range(n):
            f.write(str(val_input[i]))
            if i == n-1 :
                f.write('\n')
            else:
                f.write(', ')
                
def save_dict(listofdict, filename):
    data0 = listofdict[0]
    fieldnames = data0.keys()
    with open(filename, 'a') as f:        
        writer = csv.DictWriter(f, fieldnames = fieldnames)
        writer.writeheader()
        for row in listofdict:
            writer.writerow(row)
            
def save_dictoflist(dictoflist, filename):
    fieldnames = dictoflist.keys()
    with open(filename, 'a') as f:        
        writer = csv.DictWriter(f, fieldnames = fieldnames)
        writer.writeheader()
        n = len(dictoflist[list(fieldnames)[0]])
        for i in range(n):
            row = dict()
            for k,v in dictoflist.items():
                row[k] = v.pop(0)
            writer.writerow(row)