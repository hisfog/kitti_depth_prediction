#random shuffle lines in train_with_gt.txt
import random


def random_shuffle(path):
    with open(path, 'r') as file:
        lines = file.readlines()
        random.shuffle(lines)
    
    with open('./train_with_gt_shuffled.txt', 'w') as file:
        for _line in lines:
            file.write(_line)

if __name__ == '__main__':
    random_shuffle('./train_with_gt.txt')
    print('shuffle done')