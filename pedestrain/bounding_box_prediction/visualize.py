"""
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

df = pd.read_csv("./output/jaad_16_16_16/jaad_test_16_16_16.csv")

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

output_path = "D:/pedestrian/bounding-box-prediction/output/"


for j in range(len(ID)):
    name = "video_" + scenefolderpath[j][38] + scenefolderpath[j][39] + scenefolderpath[j][40] + scenefolderpath[j][41]
    
    #----filename
    nwe = str(filename[j])
    nwe2 = nwe.replace("\'", "\"")
    alist = json.loads(nwe2)
    filenamelist.append(alist)
    
#     #----crossing_ID
    crossing_ID = json.loads(str(ID[j]))
#     #----crossing_obs
    crossing_obslist = json.loads(crossing_obs[j])
#     #----crossing_true
    crossing_truelist = json.loads(crossing_true[j])
#     #----bounding_box
    bounding_boxlist = json.loads(bounding_box[j])
#     #----future_bounding_box
    future_bounding_boxlist = json.loads(future_bounding_box[j])

    if ( crossing_ID == 1.0 ) :     
        for i in range(len(filenamelist[j])):
            scene = Image.open(os.path.join('D:/pedestrian/JAAD/images/') + name + '/' + filenamelist[j][i])
        
            img = ImageDraw.Draw(scene)

            x = bounding_boxlist[i][0] # x
            y = bounding_boxlist[i][1] # y
            w = bounding_boxlist[i][2] # w
            h = bounding_boxlist[i][3] # h

            x1 = int(x - w/2)   #x1 
            x2 = int(x + w/2)   #y1
            y1 = int(y - h/2)   #x2
            y2 = int(y + h/2)   #y2
      
            points = (x1, y2), (x2, y2), (x2, y1), (x1, y1), (x1, y2)
            img.line(points, fill="red", width=5)
        
            fx = future_bounding_boxlist[i][0] # x
            fy = future_bounding_boxlist[i][1] # y
            fw = future_bounding_boxlist[i][2] # w
            fh = future_bounding_boxlist[i][3] # h

            fx1 = int(fx - fw/2)   #x1 
            fx2 = int(fx + fw/2)   #y1
            fy1 = int(fy - fh/2)   #x2
            fy2 = int(fy + fh/2)   #y2
        
            fpoints = (fx1, fy2), (fx2, fy2), (fx2, fy1), (fx1, fy1), (fx1, fy2)
            img.line(fpoints, fill="blue", width=5)
        
            if not os.path.isdir( output_path + name ): os.mkdir( output_path + name )
            scene.save(os.path.join('output/' ) + name + "/" + filenamelist[j][i] )
            
    else :     
        for i in range(len(filenamelist[j])):         
            # dir_path = output_path + name + '/'
            img_path = os.path.join(output_path + name + '/', filenamelist[j][i])
            
            if os.path.exists(img_path) == False :
              scene = Image.open(os.path.join('D:/pedestrian/JAAD/images/') + name + '/' + filenamelist[j][i])
            
            else : scene = Image.open(os.path.join('D:/pedestrian/bounding-box-prediction/output/') + name + '/' + filenamelist[j][i])
        
            img = ImageDraw.Draw(scene)

            x = bounding_boxlist[i][0] # x
            y = bounding_boxlist[i][1] # y
            w = bounding_boxlist[i][2] # w
            h = bounding_boxlist[i][3] # h

            x1 = int(x - w/2)   #x1 
            x2 = int(x + w/2)   #y1
            y1 = int(y - h/2)   #x2
            y2 = int(y + h/2)   #y2
      
            points = (x1, y2), (x2, y2), (x2, y1), (x1, y1), (x1, y2)
            img.line(points, fill="red", width=5)
        
            fx = future_bounding_boxlist[i][0] # x
            fy = future_bounding_boxlist[i][1] # y
            fw = future_bounding_boxlist[i][2] # w
            fh = future_bounding_boxlist[i][3] # h

            fx1 = int(fx - fw/2)   #x1 
            fx2 = int(fx + fw/2)   #y1
            fy1 = int(fy - fh/2)   #x2
            fy2 = int(fy + fh/2)   #y2
        
            fpoints = (fx1, fy2), (fx2, fy2), (fx2, fy1), (fx1, fy1), (fx1, fy2)
            img.line(fpoints, fill="blue", width=5)
            if not os.path.isdir( output_path + name ): os.mkdir( output_path + name )
            scene.save(os.path.join('output/' ) + name + "/" + filenamelist[j][i] )
"""

import csv
import queue
from sqlite3 import converters
from matplotlib.pyplot import fill
import pandas as pd
import os
#from PIL import Image, ImageDraw
import cv2
import numpy as np
from collections import deque
import json
import imghdr

df = pd.read_csv("./output/jaad_1_1_1/testing.csv")

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

output_path = "D:/pedestrian/bounding_box_prediction/output/"


for j in range(len(ID)):
    name = "video_" + scenefolderpath[j][38] + scenefolderpath[j][39] + scenefolderpath[j][40] + scenefolderpath[j][41]
    
    #----filename
    nwe = str(filename[j])
    nwe2 = nwe.replace("\'", "\"")
    alist = json.loads(nwe2)
    filenamelist.append(alist)
    
#     #----crossing_ID
    crossing_ID = json.loads(str(ID[j]))
#     #----crossing_obs
#    crossing_obslist = json.loads(crossing_obs[j])
#     #----crossing_true
#    crossing_truelist = json.loads(crossing_true[j])
#     #----bounding_box
    bounding_boxlist = json.loads(bounding_box[j])
#     #----future_bounding_box
    future_bounding_boxlist = json.loads(future_bounding_box[j])

    if ( crossing_ID == 1.0 ) :
        for i in range(len(filenamelist[j])):
            scene = cv2.imread(os.path.join('D:/pedestrian/JAAD/images/') + name + '/' + filenamelist[j][i])
            
            x = int(bounding_boxlist[i][0]) # x
            y = int(bounding_boxlist[i][1]) # y
            w = int(bounding_boxlist[i][2]) # w
            h = int(bounding_boxlist[i][3]) # h
      
            # （img为原图，左上角坐标，右下角坐标，线的颜色，线宽）
            cv2.rectangle( scene, (int(x-w/2),int(y+h/2)), (int(x+w/2),int(y-h/2)), (0,0,255), 5)
        
            fx = int(future_bounding_boxlist[i][0]) # x
            fy = int(future_bounding_boxlist[i][1]) # y
            fw = int(future_bounding_boxlist[i][2]) # w
            fh = int(future_bounding_boxlist[i][3]) # h
        
            cv2.rectangle( scene, (int(fx-fw/2),int(fy+fh/2)), (int(fx+fw/2),int(fy-fh/2)), (255,0,0), 5)
        
            if not os.path.isdir( output_path + name ): os.mkdir( output_path + name )
            cv2.imwrite(os.path.join('output/' ) + name + "/" + filenamelist[j][i], scene )
    
    else :     
        for i in range(len(filenamelist[j])):         
            img_path = os.path.join(output_path + name + '/', filenamelist[j][i])
            
            if os.path.exists(img_path) == False :
              scene = cv2.imread(os.path.join('D:/pedestrian/JAAD/images/') + name + '/' + filenamelist[j][i])
            
            else : scene = cv2.imread(os.path.join('D:/pedestrian/bounding_box_prediction/output/') + name + '/' + filenamelist[j][i])

            x = int(bounding_boxlist[i][0]) # x
            y = int(bounding_boxlist[i][1]) # y
            w = int(bounding_boxlist[i][2]) # w
            h = int(bounding_boxlist[i][3]) # h
      
            # （img为原图，左上角坐标，右下角坐标，线的颜色，线宽）
            cv2.rectangle( scene, (int(x-w/2),int(y+h/2)), (int(x+w/2),int(y-h/2)), (0,0,255), 5)
        
            fx = int(future_bounding_boxlist[i][0]) # x
            fy = int(future_bounding_boxlist[i][1]) # y
            fw = int(future_bounding_boxlist[i][2]) # w
            fh = int(future_bounding_boxlist[i][3]) # h
        
            cv2.rectangle( scene, (int(fx-fw/2),int(fy+fh/2)), (int(fx+fw/2),int(fy-fh/2)), (255,0,0), 5)
            
            if not os.path.isdir( output_path + name ): os.mkdir( output_path + name )
            cv2.imwrite(os.path.join('output/' ) + name + "/" + filenamelist[j][i], scene )


"""
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

df = pd.read_csv("./output/jaad_16_16_16/jaad_test_16_16_16.csv")

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

output_path = "D:/pedestrian/bounding-box-prediction/output/"


for j in range(len(ID)):
    name = "video_" + scenefolderpath[j][38] + scenefolderpath[j][39] + scenefolderpath[j][40] + scenefolderpath[j][41]
    
    #----filename
    nwe = str(filename[j])
    nwe2 = nwe.replace("\'", "\"")
    alist = json.loads(nwe2)
    filenamelist.append(alist)
    
#     #----crossing_ID
    crossing_ID = json.loads(str(ID[j]))
#     #----crossing_obs
#    crossing_obslist = json.loads(crossing_obs[j])
#     #----crossing_true
#    crossing_truelist = json.loads(crossing_true[j])
#     #----bounding_box
    bounding_boxlist = json.loads(bounding_box[j])
#     #----future_bounding_box
    future_bounding_boxlist = json.loads(future_bounding_box[j])

    if ( crossing_ID == 1.0 ) :     
        for i in range(len(filenamelist[j])):
            scene = Image.open(os.path.join('D:/pedestrian/JAAD/images/') + name + '/' + filenamelist[j][i])
        
            img = ImageDraw.Draw(scene)

            x = bounding_boxlist[i][0] # x
            y = bounding_boxlist[i][1] # y
            w = bounding_boxlist[i][2] # w
            h = bounding_boxlist[i][3] # h

            x1 = int(x - w/2)   #x1 
            x2 = int(x + w/2)   #y1
            y1 = int(y - h/2)   #x2
            y2 = int(y + h/2)   #y2
      
            points = (x1, y2), (x2, y2), (x2, y1), (x1, y1), (x1, y2)
            img.line(points, fill="red", width=5)
        
            fx = future_bounding_boxlist[i][0] # x
            fy = future_bounding_boxlist[i][1] # y
            fw = future_bounding_boxlist[i][2] # w
            fh = future_bounding_boxlist[i][3] # h

            fx1 = int(fx - fw/2)   #x1 
            fx2 = int(fx + fw/2)   #y1
            fy1 = int(fy - fh/2)   #x2
            fy2 = int(fy + fh/2)   #y2
        
            fpoints = (fx1, fy2), (fx2, fy2), (fx2, fy1), (fx1, fy1), (fx1, fy2)
            img.line(fpoints, fill="blue", width=5)
        
            if not os.path.isdir( output_path + name ): os.mkdir( output_path + name )
            scene.save(os.path.join('output/' ) + name + "/" + filenamelist[j][i] )
            
    else :     
        for i in range(len(filenamelist[j])):         
            # dir_path = output_path + name + '/'
            img_path = os.path.join(output_path + name + '/', filenamelist[j][i])
            
            if os.path.exists(img_path) == False :
              scene = Image.open(os.path.join('D:/pedestrian/JAAD/images/') + name + '/' + filenamelist[j][i])
            
            else : scene = Image.open(os.path.join('D:/pedestrian/bounding-box-prediction/output/') + name + '/' + filenamelist[j][i])
        
            img = ImageDraw.Draw(scene)

            x = bounding_boxlist[i][0] # x
            y = bounding_boxlist[i][1] # y
            w = bounding_boxlist[i][2] # w
            h = bounding_boxlist[i][3] # h

            x1 = int(x - w/2)   #x1 
            x2 = int(x + w/2)   #y1
            y1 = int(y - h/2)   #x2
            y2 = int(y + h/2)   #y2
      
            points = (x1, y2), (x2, y2), (x2, y1), (x1, y1), (x1, y2)
            img.line(points, fill="red", width=5)
        
            fx = future_bounding_boxlist[i][0] # x
            fy = future_bounding_boxlist[i][1] # y
            fw = future_bounding_boxlist[i][2] # w
            fh = future_bounding_boxlist[i][3] # h

            fx1 = int(fx - fw/2)   #x1 
            fx2 = int(fx + fw/2)   #y1
            fy1 = int(fy - fh/2)   #x2
            fy2 = int(fy + fh/2)   #y2
        
            fpoints = (fx1, fy2), (fx2, fy2), (fx2, fy1), (fx1, fy1), (fx1, fy2)
            img.line(fpoints, fill="blue", width=5)
            if not os.path.isdir( output_path + name ): os.mkdir( output_path + name )
            scene.save(os.path.join('output/' ) + name + "/" + filenamelist[j][i] )
"""