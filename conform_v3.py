from pedestrian.StrongSORT.track_v7 import yolov7_track_main
from pedestrian.bounding_box_prediction.test import pedestrian_main
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip

import cv2
import os
import glob
import csv
import time
import time
import pandas as pd
from PIL import Image
from shapely.geometry import Point, Polygon

leftImg8bit = './data/leftImg8bit/'
video_dir = './data/mix.mp4'

class_list_for_pedestrian = []
class_list_for_non_pedestrian = []
class_list_for_warning_object = []
road_info = [] 
obstacle_info = []
road_cnt = []
t = 0 # time

pedestrian_warning_distance = 10
obstacle_warning_distance = 10

basic_frame = 1
frame_cut = 1

class RoadObject :
    def __init__(self, object_info ) :# object_distance
        self.object_class = object_info[5]
        self.object_id = object_info[4]
        self.object_bounding_box = object_info[0:4]
        #---------------distance--------------------------
        # self.object_distance = object_distance
        #---------------pedestrian------------------------
        self.object_future_bounding_box = [0,0,0,0]
        
    def add_pedestrian_future_bounding_box(self, x, y, w, h ):
        self.object_future_bounding_box = [x,y,w,h]
        
        
def remove_dir_image( path ) :
    for f in os.listdir(path):
        os.remove(os.path.join(path, f))
   
def get_images_from_video( video_dir ) :
    vc = cv2.VideoCapture(video_dir)
    fps = vc.get(cv2.CAP_PROP_FPS)
    frame_count = int(vc.get(cv2.CAP_PROP_FRAME_COUNT))
    #print('frame_count', frame_count)

    for idx in range(0, frame_count, basic_frame):
        vc.set(1, idx)
        ret, frame = vc.read()

        if frame is not None:
            file_name = '{}{:05d}.png'.format(leftImg8bit,idx)
            cv2.imwrite(file_name, cv2.resize(frame, (2048, 1024), interpolation=cv2.INTER_AREA))

        #print("\rprocess: {}/{}".format(idx+1 , frame_count), end = '')
    vc.release()
    return frame_count

# 输入目录路径，输出最新文件完整路径
def find_new_file(dir):
    file_lists = os.listdir(dir)
    file_lists.sort(key=lambda fn: os.path.getmtime(dir + fn))
    file = os.path.join(dir, file_lists[-1])
    #print(file)
    return file

def read_txt( bbox_list, input_path ) :
    txt_file = open( input_path, 'r' )
    
    lines = txt_file.readlines()
    for i in range( len(lines) ) :
        line = lines[i].split()
        list_tmp = []
        list_tmp.append( int(line[3]) + int( int(line[5]) / 2 ) ) # x 中心
        list_tmp.append( int(line[4]) + int( int(line[6]) / 2 ) ) # y 中心
        list_tmp.append( int(line[5]) ) # w
        list_tmp.append( int(line[6]) ) # h
        list_tmp.append( int(line[2]) ) # id
        list_tmp.append( int(line[1]) ) # class
        
        bbox_list.append( list_tmp )
        
    return bbox_list

def create_csv( file_name, input_dir, path ):
    csvName_testing = './pedestrian/JAAD/processed_annotations/test/testing.csv'
    csvFile = open(csvName_testing, 'a+', newline = '' )
    writer = csv.writer(csvFile) # 建立 CSV 檔寫入器
    have_pedestrian = False
    
    f = open( str( path / 'labels' / str(file_name + ".txt") ), 'r'  )
    for line in f.readlines():
      line_data = line.split()
      if line_data[1] == '0' :

        writer.writerow([ str(int(file_name)), line_data[2] , line_data[3] , line_data[4] , 
                           line_data[5], line_data[6], input_dir , file_name + '.png'])
                           
    f.close
    csvFile.close()
    
    file = open( './pedestrian/JAAD/processed_annotations/test/testing.csv' , 'r' )
    reader = csv.reader(file)
    data_list = list(reader)
    file.close()
 
    if len(data_list) >= 3 and int(file_name) > 1 : have_pedestrian = True
    return have_pedestrian     

def draw_pedestrian( img_name, bounding_box, future_bounding_box ) :
    if len(img_name) < 5 :
        for t in range(5-len(img_name)): img_name = "0" + img_name 
    
    yolov7_track_path = find_new_file("./runs/track/")
    img = cv2.imread( yolov7_track_path + '/' + img_name + ".png" ) 

    ############## bounding_box ################

    center_x = int(bounding_box[0])
    center_y = int(bounding_box[1])
    bounding_box_w = int(bounding_box[2])
    bounding_box_h = int(bounding_box[3])
    
    cv2.rectangle( img, (center_x-bounding_box_w//2,center_y-bounding_box_h//2), 
                        (center_x+bounding_box_w//2,center_y+bounding_box_h//2), 
                        (0,0,255), 4)
    
    ############## future_bounding_box ################

    future_center_x = int(future_bounding_box[0])
    future_center_y = int(future_bounding_box[1])
    bounding_box_w = future_bounding_box[2]
    bounding_box_h = future_bounding_box[3]
    
    cv2.rectangle( img, (int(future_center_x-bounding_box_w//2),int(future_center_y-bounding_box_h//2)), 
                        (int(future_center_x+bounding_box_w//2),int(future_center_y+bounding_box_h//2)), 
                        (255,0,0), 4)
    
    cv2.imwrite(yolov7_track_path + '/' + img_name + ".png", img)
  
def convert(dir,width,height):
    file_list = os.listdir(dir)
    #print(file_list)
    for filename in file_list:
        path = ''
        path = dir+filename
        im = Image.open(path)
        out = im.resize((width,height),Image.ANTIALIAS)
        
        out.save(path)
               
def pedestrian_warning(index) :
    for i in range(len(class_list_for_pedestrian)) :
        bx_point = ( class_list_for_pedestrian[i].object_bounding_box[0],
                     class_list_for_pedestrian[i].object_bounding_box[1] + class_list_for_pedestrian[i].object_bounding_box[3] // 2 )
        fbx_point = ( class_list_for_pedestrian[i].object_future_bounding_box[0],
                      class_list_for_pedestrian[i].object_future_bounding_box[1] + class_list_for_pedestrian[i].object_future_bounding_box[3] // 2 )
        
        
        draw_pedestrian( str(index * basic_frame), 
                         class_list_for_pedestrian[i].object_bounding_box, 
                         class_list_for_pedestrian[i].object_future_bounding_box)             
        
        
        # if is_it_within_the_road_area( bx_point, index ) and is_it_within_the_road_area( fbx_point, index ):
            
        #     M = cv2.moments(road_cnt[0])
        #     # 計算中心點 x 座標
        #     center_x = int(M["m10"] / M["m00"]) 
        #     center_y = int(M["m01"] / M["m00"]) 
           
        #     # 越來越靠近1024(中心) => x 漸漸增加(遞增) => 由左至右的跨越馬路!!!
        #     if fbx_point[0] > bx_point[0] and 0 < ( center_x - fbx_point[0] ) < ( center_x - bx_point[0] ) :
        #         if class_list_for_pedestrian[i].object_distance <= pedestrian_warning_distance :
        #             #####################################
        #             warning_object = Warning_Object( class_list_for_pedestrian[i].object_bounding_box, 
        #                                              class_list_for_pedestrian[i].object_distance,
        #                                              'pedestrian crossing the road!' )
        #             class_list_for_warning_object.append( warning_object )
        #             #####################################
        #             add_waring_text( str(index * basic_frame), 
        #                              class_list_for_pedestrian[i].object_bounding_box, 
        #                              class_list_for_pedestrian[i].object_future_bounding_box,
        #                              "pedestrian!!!", center_x )
        #             #####################################
        
        #     # 越來越靠近1024(中心) => x 漸漸減少(遞減) => 由右至左的跨越馬路!!!
        #     elif fbx_point[0] < bx_point[0] and 0 < ( fbx_point[0] - center_x ) < ( bx_point[0] - center_x ) :
        #         if class_list_for_pedestrian[i].object_distance <= pedestrian_warning_distance :
        #             #####################################
        #             warning_object = Warning_Object( class_list_for_pedestrian[i].object_bounding_box, 
        #                                              class_list_for_pedestrian[i].object_distance,
        #                                              "pedestrian crossing the road!" )
        #             class_list_for_warning_object.append( warning_object )
        #             #####################################
        #             add_waring_text( str(index * basic_frame), 
        #                              class_list_for_pedestrian[i].object_bounding_box, 
        #                              class_list_for_pedestrian[i].object_future_bounding_box,
        #                              "pedestrian!!!", center_x )
        #             #####################################

        #     elif abs( fbx_point[0] - center_x ) < 100 and abs( bx_point[0] - center_x ) < 100 and \
        #          fbx_point[1] > center_y and bx_point[1] > center_y :
                 
        #         if class_list_for_pedestrian[i].object_distance <= pedestrian_warning_distance :
        #             #####################################
        #             warning_object = Warning_Object( class_list_for_pedestrian[i].object_bounding_box, 
        #                                              class_list_for_pedestrian[i].object_distance,
        #                                              "pedestrian stand in the middle!" )
        #             class_list_for_warning_object.append( warning_object )
        #             #####################################
        #             add_waring_text( str(index * basic_frame), 
        #                              class_list_for_pedestrian[i].object_bounding_box, 
        #                              class_list_for_pedestrian[i].object_future_bounding_box,
        #                              "pedestrian!!!", center_x )
        #             #####################################

    
def delete_csv_row( path, frame ) :
    df = pd.read_csv(path)
    df = df.drop( df[df.frame.apply( lambda x : x < frame -basic_frame ) ].index )
    df.to_csv(path, index=False)

def main():
    image_num = 0
    road_totalTime = 0
    yolov7_strongsort_totalTime = 0
    pedestrian_totalTime = 0
    monodepth2_totalTime = 0
    drawRoadLine_totalTime = 0
    algo_totalTime = 0

    global t 
    end = False

    while end == False :
        
        global road_info, obstacle_info, road_cnt
        road_info.clear()
        obstacle_info.clear()
        road_cnt.clear()
        
        # remove_dir_image( leftImg8bit )  如果要重切影片要打開
        
        ############################################ "路況分析系統" ############################################
        
        print("start!!!!!!!!!") 
        
        
        # # step1 : # 將道路影片裁剪成圖片
        # image_num = get_images_from_video( video_dir )    # 如果要重切影片要打開
        # convert(leftImg8bit,2048,1024)
        
        
        total_img_list = sorted(glob.glob('./data/leftImg8bit/' + '*.png'))

        for i in range(len(total_img_list)) :
        
            print( '########################################' )
            print()
            print( '第' + str(i) + '張照片' ) 
            
            if i == 0 : 
                # step4 : yolov7 + StrongSORT 模型
                yolov7_strongsort_start = time.time()
                pre, strongsort_list, input_save_dir = yolov7_track_main( total_img_list[i], [None], True, [], "" )
                yolov7_strongsort_end = time.time()
                print("yolov7+strongsort執行時間：%f 秒" % ( yolov7_strongsort_end - yolov7_strongsort_start ) ) 
                yolov7_strongsort_totalTime = yolov7_strongsort_totalTime + ( yolov7_strongsort_end - yolov7_strongsort_start )
                
                csvFile = open('./pedestrian/JAAD/processed_annotations/test/testing.csv', 'w', newline = '' )
                writer = csv.writer(csvFile) # 建立 CSV 檔寫入器
                writer.writerow(['frame', 'ID', 'x', 'y', 'w', 'h', 'scenefolderpath', 'filename'])# 寫入
                csvFile.close()
            
            else :
                # step4 : yolov7 + StrongSORT 模型
                yolov7_strongsort_start = time.time()
                pre, strongsort_list, input_save_dir = yolov7_track_main( total_img_list[i], pre, False, strongsort_list, input_save_dir )
                yolov7_strongsort_end = time.time()
                print("yolov7+strongsort執行時間：%f 秒" % ( yolov7_strongsort_end - yolov7_strongsort_start ) ) # 輸出障礙物時間
                yolov7_strongsort_totalTime = yolov7_strongsort_totalTime + ( yolov7_strongsort_end - yolov7_strongsort_start )
                
                file_name = str(i*basic_frame)
                while len(file_name) < 5 : file_name = "0" + file_name

                if os.path.exists(str( input_save_dir / 'labels' / str(file_name + ".txt") ))  :  # and len(road_cnt[0]) != 0
                    
                    pedestrian_start = time.time()
                    if i >= 3 : delete_csv_row( './pedestrian/JAAD/processed_annotations/test/testing.csv', i*basic_frame )
                    
                    # step5 : 得到行人 bounding box 資訊，並將資訊製作成行人模型的輸入csv
                    have_pedestrian = create_csv( file_name, leftImg8bit, input_save_dir  )
                    
                    bounding_box_info = []
                    future_bounding_box_info = []
                    
                    # step6 : 行人模型 => 產出行人 future bounding box 資訊
                    if have_pedestrian == True and i >= 2 : 
                        bounding_box_info, future_bounding_box_info = pedestrian_main()
                        pedestrian_end = time.time()
                        print("行人動向模型執行時間：%f 秒" % ( pedestrian_end - pedestrian_start )) # 輸出行人時間
                        pedestrian_totalTime = pedestrian_totalTime + ( pedestrian_end - pedestrian_start )



                    
                    ############################################ "道路異常警示系統" ############################################
                    algo_start = time.time()
                    
                    class_list_for_pedestrian.clear()
                    class_list_for_non_pedestrian.clear()
                    class_list_for_warning_object.clear()

                    # depth_info = np.load(disparity + '/' + file_name + '_disp.npy')[0][0]
                    bbox_list = []
                    bbox_list = read_txt( bbox_list, str( input_save_dir / 'labels' / str(file_name + ".txt") ) )
                    # distance_list = calu_main( bbox_list, depth_info )
                    
                    pedestrian_index = 0
                    for k in range(len(bbox_list)) :  
                        #if ( bbox_list[k][5] == 0 ) : print( bbox_list[k][2], bbox_list[k][3] )
                        if ( bbox_list[k][5] != 0 ) :  # "非行人"
                            # roadObject = RoadObject( bbox_list[k], distance_list[k] )
                            roadObject = RoadObject( bbox_list[k] )                
                            class_list_for_non_pedestrian.append(roadObject)
                           
                        elif pedestrian_index < len(future_bounding_box_info) and \
                             int(bounding_box_info[pedestrian_index][2]) == bbox_list[k][2] and \
                             int(bounding_box_info[pedestrian_index][3]) == bbox_list[k][3] : # "行人"
                                 
                            # roadObject = RoadObject( bbox_list[k], distance_list[k] )            
                            roadObject = RoadObject( bbox_list[k] )          
                            roadObject.add_pedestrian_future_bounding_box( future_bounding_box_info[pedestrian_index][0], 
                                                                           future_bounding_box_info[pedestrian_index][1], 
                                                                           future_bounding_box_info[pedestrian_index][2], 
                                                                           future_bounding_box_info[pedestrian_index][3] )      
                            pedestrian_index = pedestrian_index + 1
                            class_list_for_pedestrian.append(roadObject)

                    if len(class_list_for_pedestrian) != 0 : pedestrian_warning(i) # 行人警示
            
            print( '########################################' )
            print()
                    
        
        print( 'total_image:', image_num // basic_frame )
                                           
        break
        
        

if __name__ == "__main__":
    main()
    