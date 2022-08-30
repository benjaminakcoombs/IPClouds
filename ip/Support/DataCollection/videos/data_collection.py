import numpy as np
import pandas as pd
import urllib
import requests
import datetime
import cv2
import os
import shutil
import math


def date_range(start_date, end_date):
    dates = []
    delta = end_date - start_date
    for i in range(delta.days + 1):
        date = start_date+datetime.timedelta(days=i)
        dates.append(date.strftime("%Y-%m-%d"))
    return dates

def timelapseDataDownload(dates, min_hour = 8, max_hour = 17):
    for day in dates:
        for camera in ['', '4']: #4, 5
        #for camera in ['', '2', '4', '5']: #4, 5
            for hour in range(min_hour, max_hour): #22
                if (hour < 10):
                    hour = '0' + str(hour)
                else:
                    hour = str(hour)
                for exposure in ['A']:#, 'B', 'C']: #Add back in when we do this properly with lots more data?
                    url = "http://www.sp.ph.ic.ac.uk/~erg10/safe/timelapse/videos//" + day + "/tl" + camera + "_" + day + "_" + str(hour) + exposure + ".mp4"
                    print(url)
                    camera_name = camera
                    if camera_name == '':
                        camera_name = '1'
                    name = day + "_" + str(hour) + "_" + camera_name + "_" + exposure + ".mp4"
                    dirName = r"/vol/bitbucket/bac21/steak-generation/timelapse/videos"
                    path = os.path.join(dirName, name)
                    r = requests.get(url)
                    print ("****Connected****")
                    f = open(path, 'wb')
                    print ("Downloading.....")
                    for chunk in r.iter_content(chunk_size=255): 
                        if chunk: # filter out keep-alive new chunks
                            f.write(chunk)
                    print ("Done")
                    f.close()
    #for parent, directories, files in os.walk(dirName):
    #    if os.path.getsize(file) < 5 * 1024:
    #        os.remove(file)


dates = date_range(datetime.datetime(2022, 5, 20), datetime.datetime(2022, 5, 20))
                   #(2022, 3, 10), datetime.datetime(2022, 5, 6))
timelapseDataDownload(dates, 14, 15)


def create_mask(img, num):
    path = r"/vol/bitbucket/bac21/steak-generation/ip/Support/DataCollection/videos"
    building_template = str(num) + "buildingTemplate.png"
    building_mask = os.path.join(path, building_template)
    building_mask = cv2.imread(building_mask, cv2.IMREAD_GRAYSCALE)

    canny_thresh = 40
    if int(num) == 4:
        canny_thresh = 25
    img_size = [1280,960]
        
    building_mask = cv2.resize(building_mask, (img_size[0], img_size[1]))
    for i in range(img_size[1]):
        for j in range(img_size[0]):
            if building_mask[i][j] == 0:
                img[i][j] = 0

    edges = cv2.Canny(img,canny_thresh,canny_thresh)
    
    _,thresh = cv2.threshold(edges, 0, 255, cv2.THRESH_BINARY)
    rect=cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilation = cv2.dilate(thresh,rect,iterations = 6)
    erosion = cv2.erode(dilation, rect, iterations=4)

    num = 0
    for pixel in range(len(erosion[0])):
        if erosion[0][pixel] == 0:
            num = pixel
            break
    im_floodfill = erosion.copy()
    cv2.floodFill(im_floodfill, None, (num, 0), 255)
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)
    im_out = erosion | im_floodfill_inv
    for i in range(0, len(img)):
        for j in range(0, len(img[i])):
            if (max(img[i,j]) - min(img[i,j]) <= 40 and sum(img[i,j] > 10)):
                im_out[i,j] = 255
            elif sum(img[i,j]) < 10:
                im_out[i,j] = 0
    return im_out


dirName = r"/vol/bitbucket/bac21/steak-generation/timelapse"
video_path = os.path.join(dirName, 'videos')
list_of_mp4 = []
for parent, directories, files in os.walk(video_path):
    for file in files:
        list_of_mp4.append(file)
list_of_dates_and_exposures = []
for file in list_of_mp4:
    list_of_dates_and_exposures.append(file[:14] + file[-5])

list_of_dates_and_exposures = list(dict.fromkeys(list_of_dates_and_exposures))

list_of_all_frames = []
#new_dir_path = os.path.join(dirName, 'frames')
#os.mkdir(new_dir_path)
for date_exposure in list_of_dates_and_exposures:
    for parent, directories, files in os.walk(video_path):
        for file in files:
            if date_exposure == (file[:14] + file[-5]):
                file_path = os.path.join(video_path, file)
                vidcap = cv2.VideoCapture(file_path)
                success,image = vidcap.read()
                count = 0
                while count < 10: #while success:
                    frame = date_exposure + '_' + str(count)
                    path = os.path.join(dirName, 'frames', frame)
                    image_path = os.path.join(path, "rgb")    
                    mask_path = os.path.join(path, "mask")    
                    pose_path = os.path.join(path, "pose")
                    if frame not in list_of_all_frames:
                        os.mkdir(path)
                        os.mkdir(image_path)
                        os.mkdir(mask_path)
                        os.mkdir(pose_path)
                        print('New directory created')
                        list_of_all_frames.append(frame)
                    image_name = file[14] + '.png'
                    mask_name = file[14] + '.png'
                    image = cv2.resize(image, (1280, 960))
                    mask = create_mask(image, file[14])
                    image = cv2.resize(image, (640, 480))
                    mask = cv2.resize(mask, (640, 480))
                    cv2.imwrite(os.path.join(image_path, image_name), image)     # save frame as JPEG file 
                    cv2.imwrite(os.path.join(mask_path, mask_name), mask)     # save frame as JPEG file 
                    success,image = vidcap.read()
                    print('Read a new frame: ', success)
                    count += 1
                vidcap.release()




#def split_dataset(list_of_all_frames, train_percent, val_percent, test_percent):
#    if train_percent + val_percent + test_percent != 1:
#        print("Percentages need to add to 1")
#        return
#    #### ADD YOUR CODE HERE ####
#    lendata = len(list_of_all_frames)
#    indices = np.array(list(range(lendata)))
#    np.random.shuffle(indices)
#    train_split = int(train_percent * lendata)
#    val_split = int((val_percent+train_percent) * lendata)
#    idxs_train = indices[:train_split]
#    idxs_val = indices[train_split:val_split]
#    idxs_test = indices[val_split:]
#    train_data = []
#    val_data = []
#    test_data = []
#    for i in idxs_train:
#        train_data.append(list_of_all_frames[i])
#    for j in idxs_val:
#        val_data.append(list_of_all_frames[j])
#    for k in idxs_test:
#        test_data.append(list_of_all_frames[k])
#    return train_data, val_data, test_data
##
#
#train_data, val_data, test_data = split_dataset(list_of_all_frames, 0.8, 0.1, 0.1)
#with open(r"/vol/bitbucket/bac21/steak-generation/timelapse/new_train.lst", 'w') as train_lst:
#    for item in train_data:
#        train_lst.write("%s\n" % item)
#    print('Train done')
#with open(r"/vol/bitbucket/bac21/steak-generation/timelapse/new_val.lst", 'w') as val_lst:
#    for item in val_data:
#            # write each item on a new line
##        val_lst.write("%s\n" % item)
#    print('Val done')
#with open(r"/vol/bitbucket/bac21/steak-generation/timelapse/new_test.lst", 'w') as test_lst:
#    for item in test_data:
#            # write each item on a new line
#        test_lst.write("%s\n" % item)
#    print('Test done')