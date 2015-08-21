#!/home/bginsburg/anaconda/bin/python

"""Extract accuracy and loss from log files
   usage: file [file ...]
   Written by Lior Shani.
"""

import sys
import re
import os
import math
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

#--------------------------------------------------------------------
def extract_accuracy(filename):
  f=open(filename,'r')
  openFile = f.read()
  iteration = re.findall(r'Iteration (\d*), Testing net \(#0\)', openFile)
  #print iteration
  accuracy = re.findall(r'Test net output #0: accuracy = (\d*.\d*)', openFile) 
  #print accuracy
  out = []
  for i in range(len(iteration)):
    out.append((iteration[i],accuracy[i]))
  return out 

#--------------------------------------------------------------------
def extract_loss(filename):
  f=open(filename,'r')
  openFile = f.read()
  iteration = re.findall(r'Iteration (\d*), Testing net ', openFile)
  loss = re.findall(r'Test net output #1: loss = (\d*.\d*)', openFile) 
  out = []
  for i in range(len(loss)):
    out.append((iteration[i],loss[i]))
  return out 

def extract_iteration(filename):
  f=open(filename,'r')
  print filename
  openFile = f.read()
  iteration = re.findall(r"Iteration (\d*), Testing net \(#0\)", openFile)
  return iteration


#----------------------------------------------------------------------
def main():
  # command-line parsing
  args = sys.argv[1:]
  if not args:
    print 'usage: [file ...]'
    sys.exit(1)

  #build legend names from file names 
  filenames = []
  legendnames = []
  for name in args:
    if name.endswith('.log'):
      filenames.append(os.path.split(os.path.splitext(name)[0])[1])
      legendnames.append(filenames[-1])
  
  #plot accuracy  
  colors = iter(cm.rainbow(np.linspace(0, 1, len(args))))
  plt.figure(1)
  for filename in args:
    accuracy = extract_accuracy(filename)
    plt.scatter(*zip(*accuracy),color=next(colors))
    print filename + ": Final accuracy is: " + accuracy[len(accuracy)-1][1]
  plt.title('Accuracy')
  plt.xlabel('Iteration')
  plt.xlim(0,30000)
  plt.ylabel('Accuracy')
  plt.ylim(0,1)
  plt.legend(legendnames,loc='lower right')
  # plt.show() 
  

  # -----------
  #plot loss   
	
  colors = iter(cm.rainbow(np.linspace(0, 1, len(args))))
  plt.figure(2)
  for filename in args:
    loss = extract_loss(filename)
    [iteration,loss] =  zip(*loss)
    loss_new = []
    for i in range(len(loss)):
      loss_new.append(np.log(float(loss[i])))
    plt.scatter(iteration,loss_new,color=next(colors))
  plt.title('Log Loss')
  plt.xlabel('Iteration')
  plt.ylabel('Log Loss')
  plt.legend(legendnames,loc='upper right')
  plt.show() 
  
  
if __name__ == '__main__':
  main()
