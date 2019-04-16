import os

def read_file_to_dic(path):
    result = {}
    with open(path, 'r') as f:
        for line in f:
            items = line.split()
            key, value = items[0], items[1]
            result[key] = value
    return result


def rm_image_from_dic(image_path, dic_path):
    path_to_target = 'bird_dataset/'+dic_path+'/'+image_path
    os.system("rm -rf "+ path_to_target)

if __name__ == '__main__':
    totalcount = 0
    trainrm = 0
    testrm = 0
    train_test_split_dic = read_file_to_dic('data/CUB_200_2011/CUB_200_2011/train_test_split.txt')
    images_dic = read_file_to_dic('data/CUB_200_2011/CUB_200_2011/images.txt')
    for image_id, flag in train_test_split_dic.items():
        totalcount+=1
        if flag == '0':
            rm_image_from_dic(images_dic[image_id], 'train_images')
            trainrm+=1
        else:
            rm_image_from_dic(images_dic[image_id], 'test_images')
            rm_image_from_dic(images_dic[image_id], 'val_images')
            testrm+=1
    print("Totally there are " + str(totalcount) + " pieces of images, and we split them into " + str(totalcount-trainrm)+' in TRAINSET and ' + str(totalcount-testrm) + ' in VALSET and TESTSET')




