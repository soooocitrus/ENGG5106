import os

def read_file_to_dic(path):
    result = {}
    with open(path, 'r') as f:
        for line in f:
            items = line.split()
            key, value = items[0], items[1]
            result[key] = value
    return result

def read_class_label_to_dic(path):
    result = dict()
    with open(path, 'r') as f:
        for line in f:
            items = line.split()
            image_id, class_label = items[0], items[1]
            if class_label in result:
                result[class_label].append(image_id)
            else:
                result[class_label] = [image_id]
    return result

def split_image(train_image_percent, class_label_dic):
    open('train_test_split.txt', 'w').close()
    with open('train_test_split.txt', 'a') as writer:
        for key, values in class_label_dic.items():
            number = len(values)
            train_num = number * train_image_percent
            for image_id in values:
                if train_num > 0:
                    writer.write(image_id + ' ' + '1\n')
                else:
                    writer.write(image_id + ' ' + '0\n')
                train_num -= 1
        writer.close()




def rm_image_from_dic(image_path, dic_path):
    path_to_target = 'bird_dataset/'+dic_path+'/'+image_path
    os.system("rm -rf "+ path_to_target)

if __name__ == '__main__':
    totalcount = 0
    trainrm = 0
    testrm = 0
    class_label_dic = read_class_label_to_dic('data/CUB_200_2011/CUB_200_2011/image_class_labels.txt')
    split_image(0.8, class_label_dic)
    train_test_split_dic = read_file_to_dic('train_test_split.txt')
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




