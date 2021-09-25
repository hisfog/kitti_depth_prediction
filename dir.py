import os

def process_dir(path, output_file_name):

    dirlist = os.listdir(path)
    
    file = open(output_file_name, "a")
    
    for dir in dirlist:
        if not os.path.isdir(dir) and dir.endswith('.png'):
            file.write(path+'/'+dir+'\n')
    
    file.close()


def generate_train_data(train_dir, groundtruth_dir, train_file_name, groundtruth_file_name):
    dirlist = os.listdir(groundtruth_dir)
    train_file = open(train_file_name, "a")
    groundtruth_file = open(groundtruth_file_name, "a")
    cnt = 0
    for dir in dirlist:
        if not os.path.isdir(dir) and dir.endswith('.png'):
            train_file.write(train_dir+'/'+dir+'\n')
            groundtruth_file.write(groundtruth_dir+'/'+dir+'\n')
            cnt += 1

    print('{} images done in '.format(cnt)+groundtruth_dir)

    train_file.close()
    groundtruth_file.close()



train_file_name = 'train.txt'
groundtruth_file_name = 'groundtruth.txt'
file = open('train.txt', "w").close()
file = open('groundtruth.txt', "w").close()

train_dir = './2011_09_26/2011_09_26_drive_0014_sync/image_02/data'
groundtruth_dir = './train/2011_09_26_drive_0014_sync/proj_depth/groundtruth/image_02'
# train_dir = './data/2011_09_26_drive_0014_sync/image_02/data'
# groundtruth_dir = './data/2011_09_26_drive_0014_sync/proj_depth/groundtruth/image_02'
# process_dir(train_dir, 'train.txt')
# process_dir(groundtruth_dir, 'groundtruth.txt')
generate_train_data(train_dir, groundtruth_dir, train_file_name, groundtruth_file_name)

train_dir = './2011_09_26/2011_09_26_drive_0001_sync/image_02/data'
groundtruth_dir = './train/2011_09_26_drive_0001_sync/proj_depth/groundtruth/image_02'
generate_train_data(train_dir, groundtruth_dir, train_file_name, groundtruth_file_name)

train_dir = './2011_09_26/2011_09_26_drive_0018_sync/image_02/data'
groundtruth_dir = './train/2011_09_26_drive_0018_sync/proj_depth/groundtruth/image_02'
generate_train_data(train_dir, groundtruth_dir, train_file_name, groundtruth_file_name)

train_dir = './2011_09_26/2011_09_26_drive_0009_sync/image_02/data'
groundtruth_dir = './train/2011_09_26_drive_0009_sync/proj_depth/groundtruth/image_02'
generate_train_data(train_dir, groundtruth_dir, train_file_name, groundtruth_file_name)

train_dir = './2011_09_26/2011_09_26_drive_0015_sync/image_02/data'
groundtruth_dir = './train/2011_09_26_drive_0015_sync/proj_depth/groundtruth/image_02'
generate_train_data(train_dir, groundtruth_dir, train_file_name, groundtruth_file_name)

train_dir = './2011_09_26/2011_09_26_drive_0019_sync/image_02/data'
groundtruth_dir = './train/2011_09_26_drive_0019_sync/proj_depth/groundtruth/image_02'
generate_train_data(train_dir, groundtruth_dir, train_file_name, groundtruth_file_name)

train_dir = './2011_09_26/2011_09_26_drive_0011_sync/image_02/data'
groundtruth_dir = './train/2011_09_26_drive_0011_sync/proj_depth/groundtruth/image_02'
generate_train_data(train_dir, groundtruth_dir, train_file_name, groundtruth_file_name)

train_dir = './2011_09_26/2011_09_26_drive_0017_sync/image_02/data'
groundtruth_dir = './train/2011_09_26_drive_0017_sync/proj_depth/groundtruth/image_02'
generate_train_data(train_dir, groundtruth_dir, train_file_name, groundtruth_file_name)