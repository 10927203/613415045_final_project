import csv
import queue
from sqlite3 import converters
from matplotlib.pyplot import fill
import pandas as pd
import os
from PIL import Image, ImageDraw
import numpy as np
from collections import deque
import json
import imghdr

#　df = pd.read_csv("video_0332.csv")

df = pd.read_csv("./output/jaad_16_16_16/jaad_test_16_16_16.csv")

# print(df.head())
# ID,bounding_box,future_bounding_box,scenefolderpath,filename,crossing_obs,crossing_true,label

ID = df['ID']
bounding_box = df['bounding_box']
future_bounding_box = df['future_bounding_box']
scenefolderpath = df['scenefolderpath']
filename = df['filename']
crossing_obs = df['crossing_obs']
crossing_true = df['crossing_true']
label = df['label']

crossing_ID = []
filenamelist = []
crossing_obslist = []
crossing_truelist = []
bounding_boxlist = []
future_bounding_boxlist = []


for i in range(len(ID)):
#    print(i)
#     #----filename
    nwe = str(filename[i])
    nwe2 = nwe.replace("\'", "\"")
    alist = json.loads(nwe2)
    filenamelist.append(alist)
    
#     #----crossing_ID
    crossing_ID.append(json.loads(str(ID[i]))) 
#     #----crossing_obs
    crossing_obslist.append(json.loads(crossing_obs[i]))
#     #----crossing_true
    crossing_truelist.append(json.loads(crossing_true[i]))
#     #----bounding_box
    bounding_boxlist.append(json.loads(bounding_box[i]))
#     #----future_bounding_box
    future_bounding_boxlist.append(json.loads(future_bounding_box[i]))

#run = "video_0313"

"""
count = 0
for j in range(len(ID)):
    
    name = "video_" + scenefolderpath[j][38] + scenefolderpath[j][39] + scenefolderpath[j][40] + scenefolderpath[j][41] 
    #print( name )
    #if run != name : break 
    #print(crossing_truelist[j])
    #print( "bounding_boxlist: ", bounding_boxlist[j] )
    #print( "future_bounding_boxlist: ", future_bounding_boxlist[j] )
    for i in range(len(filenamelist[j])):
        scene = Image.open(os.path.join('D:/pedestrian/JAAD/images/') + name + '/' + filenamelist[j][i])
        img = ImageDraw.Draw(scene)
        if (crossing_truelist[j][i] == 1):
            sizeb = 10
        else:
            sizeb = 5

        x = bounding_boxlist[j][i][0] # x
        y = bounding_boxlist[j][i][1] # y
        w = bounding_boxlist[j][i][2] # w
        h = bounding_boxlist[j][i][3] # h

        x1 = int(x - w/2)   #x1 
        x2 = int(x + w/2)   #y1
        y1 = int(y - h/2)   #x2
        y2 = int(y + h/2)   #y2
        # print( x, "  ", y, end = '\n\n' )
        # points = (x_low_left, x_high_right), (x_high_right, y_high_right), (y_high_right, y_low_left), (y_low_left, x_low_left), (x_low_left, x_high_right)
        # points = (x_low_left, x_high_right), (y_low_left, x_high_right), (y_low_left, y_high_right), (x_low_left, y_high_right), (x_low_left, x_high_right)
        # points = (x_low_left, y_low_left), (y_high_right, y_low_left), (y_high_right, y_high_right), (x_low_left, y_high_right), (x_low_left, y_low_left)
        points = (x1, y2), (x2, y2), (x2, y1), (x1, y1), (x1, y2)
        img.line(points, fill="red", width=sizeb)
        
        fx = future_bounding_boxlist[j][i][0] # x
        fy = future_bounding_boxlist[j][i][1] # y
        fw = future_bounding_boxlist[j][i][2] # w
        fh = future_bounding_boxlist[j][i][3] # h

        fx1 = int(fx - fw/2)   #x1 
        fx2 = int(fx + fw/2)   #y1
        fy1 = int(fy - fh/2)   #x2
        fy2 = int(fy + fh/2)   #y2
        
        fpoints = (fx1, fy2), (fx2, fy2), (fx2, fy1), (fx1, fy1), (fx1, fy2)
        img.line(fpoints, fill="blue", width=sizeb)
        
        path = "D:/pedestrian/bounding-box-prediction/output/"
        if not os.path.isdir( path + name ): 
            os.mkdir( path + name )
            count = 0
        scene.save(os.path.join('output/' ) + name + "/" + str(count) + ".png" )
        count = count + 1
        
"""

print( crossing_ID )


for j in range(len(ID)):
    print( j )
    name = "video_" + scenefolderpath[j][38] + scenefolderpath[j][39] + scenefolderpath[j][40] + scenefolderpath[j][41]
    if ( crossing_ID[j] == 1.0 ) :     
        for i in range(len(filenamelist[j])):
            scene = Image.open(os.path.join('D:/pedestrian/JAAD/images/') + name + '/' + filenamelist[j][i])
        
            img = ImageDraw.Draw(scene)
            if (crossing_truelist[j][i] == 1): sizeb = 10
            else: sizeb = 5

            x = bounding_boxlist[j][i][0] # x
            y = bounding_boxlist[j][i][1] # y
            w = bounding_boxlist[j][i][2] # w
            h = bounding_boxlist[j][i][3] # h

            x1 = int(x - w/2)   #x1 
            x2 = int(x + w/2)   #y1
            y1 = int(y - h/2)   #x2
            y2 = int(y + h/2)   #y2
      
            points = (x1, y2), (x2, y2), (x2, y1), (x1, y1), (x1, y2)
            img.line(points, fill="red", width=sizeb)
        
            fx = future_bounding_boxlist[j][i][0] # x
            fy = future_bounding_boxlist[j][i][1] # y
            fw = future_bounding_boxlist[j][i][2] # w
            fh = future_bounding_boxlist[j][i][3] # h

            fx1 = int(fx - fw/2)   #x1 
            fx2 = int(fx + fw/2)   #y1
            fy1 = int(fy - fh/2)   #x2
            fy2 = int(fy + fh/2)   #y2
        
            fpoints = (fx1, fy2), (fx2, fy2), (fx2, fy1), (fx1, fy1), (fx1, fy2)
            img.line(fpoints, fill="blue", width=sizeb)
        
            path = "D:/pedestrian/bounding-box-prediction/output/"
            if not os.path.isdir( path + name ): 
                os.mkdir( path + name )
            scene.save(os.path.join('output/' ) + name + "/" + filenamelist[j][i] )
            
    else :     
        for i in range(len(filenamelist[j])):
            
            dir_path = 'D:/pedestrian/bounding-box-prediction/output/' + name + '/'
            img_path = os.path.join(dir_path, filenamelist[j][i])
            print( img_path )
            if os.path.exists(img_path) == False :
              scene = Image.open(os.path.join('D:/pedestrian/JAAD/images/') + name + '/' + filenamelist[j][i])
            
            else : scene = Image.open(os.path.join('D:/pedestrian/bounding-box-prediction/output/') + name + '/' + filenamelist[j][i])
        
            img = ImageDraw.Draw(scene)
            if (crossing_truelist[j][i] == 1): sizeb = 10
            else: sizeb = 5

            x = bounding_boxlist[j][i][0] # x
            y = bounding_boxlist[j][i][1] # y
            w = bounding_boxlist[j][i][2] # w
            h = bounding_boxlist[j][i][3] # h

            x1 = int(x - w/2)   #x1 
            x2 = int(x + w/2)   #y1
            y1 = int(y - h/2)   #x2
            y2 = int(y + h/2)   #y2
      
            points = (x1, y2), (x2, y2), (x2, y1), (x1, y1), (x1, y2)
            img.line(points, fill="red", width=sizeb)
        
            fx = future_bounding_boxlist[j][i][0] # x
            fy = future_bounding_boxlist[j][i][1] # y
            fw = future_bounding_boxlist[j][i][2] # w
            fh = future_bounding_boxlist[j][i][3] # h

            fx1 = int(fx - fw/2)   #x1 
            fx2 = int(fx + fw/2)   #y1
            fy1 = int(fy - fh/2)   #x2
            fy2 = int(fy + fh/2)   #y2
        
            fpoints = (fx1, fy2), (fx2, fy2), (fx2, fy1), (fx1, fy1), (fx1, fy2)
            img.line(fpoints, fill="blue", width=sizeb)
        
            path = "D:/pedestrian/bounding-box-prediction/output/"
            scene.save(os.path.join('output/' ) + name + "/" + filenamelist[j][i] )
