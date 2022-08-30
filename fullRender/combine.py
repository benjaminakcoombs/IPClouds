import os
import shutil
from PIL import Image


dirName = r"/vol/bitbucket/bac21/steak-generation/fullRender/fullRender"
dirName_temp = r"/vol/bitbucket/bac21/steak-generation/fullRender/fullRender/temp"
dirName_final = r"/vol/bitbucket/bac21/steak-generation/fullRender/fullRender/final"
dirName_left = r"/vol/bitbucket/bac21/steak-generation/fullRender/Left"
dirName_right = r"/vol/bitbucket/bac21/steak-generation/fullRender/Right"
#for parent, directories, files in os.walk(dirName_left):
#    for file in files:
#        file_path_left = os.path.join(dirName_left, file)
#        file_path_right = os.path.join(dirName_right, file)
#        new_folder_path = os.path.join(dirName_temp, str(int(file[4:8])))
#        os.mkdir(new_folder_path)
#        rgb_folder_path = os.path.join(new_folder_path, "rgb")
#        os.mkdir(rgb_folder_path)
#        pose_folder_path = os.path.join(new_folder_path, "pose")
#        os.mkdir(pose_folder_path)
#        new_file_path_left = os.path.join(rgb_folder_path,"4.png")
#        new_file_path_right = os.path.join(rgb_folder_path, "1.png")
#        shutil.move(file_path_left, new_file_path_left)
#        shutil.move(file_path_right, new_file_path_right)
#        file_path_intrinsics = r"/vol/bitbucket/bac21/steak-generation/testRender_test/9/intrinsics.txt"
#        file_path_pose_1 = r"/vol/bitbucket/bac21/steak-generation/testRender_test/9/pose/1.txt"
#        file_path_pose_4 = r"/vol/bitbucket/bac21/steak-generation/testRender_test/9/pose/4.txt"
#        shutil.copyfile(file_path_intrinsics, os.path.join(new_folder_path, "intrinsics.txt"))
#        shutil.copyfile(file_path_pose_1, os.path.join(pose_folder_path, "1.txt"))
#        shutil.copyfile(file_path_pose_4, os.path.join(pose_folder_path, "4.txt"))


#for parent, directories, files in os.walk(dirName_temp):
#    for directory in directories:
#        current_directory_path = os.path.join(dirName_temp, directory)
#        if int(directory) <= 600:
#            new_directory_path = os.path.join(dirName_final, "fullRender_train", directory)
#            shutil.move(current_directory_path, new_directory_path)
#        elif int(directory) > 600 and int(directory) <= 800:
#            new_directory_path = os.path.join(dirName_final, "fullRender_val", directory)
#            shutil.move(current_directory_path, new_directory_path)
#        else:
#            new_directory_path = os.path.join(dirName_final, "fullRender_test", directory)
#            shutil.move(current_directory_path, new_directory_path)
            


dir_name_1txt = r"/vol/bitbucket/bac21/steak-generation/fullRender/1.txt"
dir_name_4txt = r"/vol/bitbucket/bac21/steak-generation/fullRender/4.txt"
dir_name_itxt = r"/vol/bitbucket/bac21/steak-generation/fullRender/intrinsics.txt"

for directory in ['fullRender_train', 'fullRender_test', 'fullRender_val']:
    spec_dir = os.path.join(dirName_final, directory)
    for parent, folders, files in os.walk(spec_dir):
        for folder in folders:
            if folder not in ['rgb', 'pose', 'mask']:
                spec_dir_2_1 = os.path.join(spec_dir, folder, "pose", "1.txt")
                spec_dir_2_4 = os.path.join(spec_dir, folder, "pose", "4.txt")
                spec_dir_2_i = os.path.join(spec_dir, folder, "intrinsics.txt")
                shutil.copyfile(dir_name_1txt, spec_dir_2_1)
                shutil.copyfile(dir_name_4txt, spec_dir_2_4)
                shutil.copyfile(dir_name_itxt, spec_dir_2_i)

#for parent_2, directories_2, files_2 in os.walk(spec_dir):
##    for directory in directories:
 #       spec_dir = os.path.join(dirName_final, directory)
 #       print(directory)
  #      for parent_2, directories_2, files_2 in os.walk(spec_dir):
   #         print(spec_dir)
    #        print(directories_2)
     #       for directory_2 in directories_2:
      #          spec_dir_2_1 = os.path.join(spec_dir, directory_2, "pose", "1.txt")
       #         spec_dir_2_4 = os.path.join(spec_dir, directory_2, "pose", "4.txt")
        #        shutil.copyfile(dir_name_1txt, spec_dir_2_1)
         #       shutil.copyfile(dir_name_4txt, spec_dir_2_4)



#for directory in ['fullRender_test', 'fullRender_train', 'fullRender_val']:
#    spec_dir = os.path.join(dirName_final, directory)
#    for parent, folders, files in os.walk(spec_dir):
#        for folder in folders:
#            if folder not in ['rgb', 'pose']:
#                spec_dir_folder = os.path.join(spec_dir, folder, "rgb")
#                spec_dir_2_1 = os.path.join(spec_dir, folder, "rgb", "1.png")
#                spec_dir_2_4 = os.path.join(spec_dir, folder, "rgb", "4.png")
#                spec_dir_2_1_tif = os.path.join(spec_dir, folder, "rgb", "1.tif")
#                spec_dir_2_4_tif = os.path.join(spec_dir, folder, "rgb", "4.tif")
#                shutil.move(spec_dir_2_1, spec_dir_2_1_tif)
#                shutil.move(spec_dir_2_4, spec_dir_2_4_tif)
#                im_1 = Image.open(spec_dir_2_1_tif)
#                im_4 = Image.open(spec_dir_2_4_tif)
##                print("Generating jpeg for %s" % name)
#                im_1.thumbnail(im_1.size)
#                im_1.save(os.path.join(spec_dir, folder, "rgb", "1.jpg"), "JPEG", quality=100)
#                im_4.thumbnail(im_4.size)
#                im_4.save(os.path.join(spec_dir, folder, "rgb", "4.jpg"), "JPEG", quality=100)


#for directory in ['fullRender_test', 'fullRender_train', 'fullRender_val']:
#    spec_dir = os.path.join(dirName_final, directory)
#    for parent, folders, files in os.walk(spec_dir):
#        for folder in folders:
#            if folder not in ['rgb', 'pose']:
#                spec_dir_2_1_tif = os.path.join(spec_dir, folder, "rgb", "1.tif")
#                spec_dir_2_4_tif = os.path.join(spec_dir, folder, "rgb", "4.tif")
#                os.remove(spec_dir_2_1_tif)
#                os.remove(spec_dir_2_4_tif)
                