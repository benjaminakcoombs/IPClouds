import numpy as np
import os
import shutil


scale_mat_0 = [[1,0, 0, 0], 
                [0, 3048, 0, 0], 
                [0, 0, 5200, 0],
                [0, 0, 0, 1]]

world_mat_0 = [[ 0.99988914,0.01471834, 0.00224488, -52.1222311], 
    [-0.01477408,  0.99951947, 0.02724975, -14.0858871], 
    [-0.00184273, -0.02727989, 0.99962616, -0.92779816], 
    [0, 0, 0, 1]]

camera_mat_inv_0 =  [[1016.95046740021, 0, 0, 645.037091207198],
                    [0, 1016.95046740021, 0, 501.180200168174],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]]



scale_mat_1 = [[1,0, 0, 0], 
                [0, 3048, 0, 0], 
                [0, 0, 5200, 0],
                [0, 0, 0, 1]]


world_mat_1 = [[1, 0, 0, 0], 
                [0, 1, 0, 0], 
                [0, 0, 1, 0], 
                [0, 0, 0, 1]]

camera_mat_inv_1 =  [[1016.95046740021, 0, 0, 645.037091207198],
                    [0, 1016.95046740021, 0, 501.180200168174],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]]


dirName = r"/vol/bitbucket/bac21/steak-generation/timelapse/timelapse"
for parent, directories, files in os.walk(dirName):
    for directory in directories:
        path = str(os.path.join(dirName, directory))
        print(path)
        np.savez(path, scale_mat_0=scale_mat_0, world_mat_0=world_mat_0, camera_mat_inv_0= camera_mat_inv_0
                        , scale_mat_1=scale_mat_1, world_mat_1=world_mat_1, camera_mat_inv_1= camera_mat_inv_1)
        for parent, directories, files in os.walk(path):
            image_dir = os.path.join(path, "image")
            os.mkdir(image_dir)
#            for file in files:
#                file_path = os.path.join(path, file)
#                new_file_path = os.path.join(image_dir, file)
#                shutil.move(file_path, new_file_path)
#            np.savez(path, scale_mat_0=scale_mat_0, world_mat_0=world_mat_0, camera_mat_inv_0= camera_mat_inv_0
#                            , scale_mat_1=scale_mat_1, world_mat_1=world_mat_1, camera_mat_inv_1= camera_mat_inv_1)

