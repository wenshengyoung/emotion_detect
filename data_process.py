
import shutil

f = open('E:\\BaiduNetdiskDownload\\basic\EmoLabel\\list_patition_label.txt', 'r')
lines = f.readlines()
# print(len(lines))
for line in lines:

    source_path = 'E:\\BaiduNetdiskDownload\\basic\Image\\aligned\\'
    target_path = 'Y:\\RAF_Data\\'
    if str(line.split('_')[0]) == 'train':

        target_path = target_path + 'train'
        if int(line.split(' ')[1]) == 1:
            target_path = target_path + '\\0'
        elif int(line.split(' ')[1]) == 2:
            target_path = target_path + '\\1'
        elif int(line.split(' ')[1]) == 3:
            target_path = target_path + '\\2'
        elif int(line.split(' ')[1]) == 4:
            target_path = target_path + '\\3'
        elif int(line.split(' ')[1]) == 5:
            target_path = target_path + '\\4'
        elif int(line.split(' ')[1]) == 6:
            target_path = target_path + '\\5'
        else:
            target_path = target_path + '\\6'
        print(target_path)
        shutil.copy(source_path + 'train_' + str(line.split('_')[1][:5]) + '_aligned.jpg', target_path)
    # print(type(str(line.split('_'[0]))))
    # print(line.split('.')[1][4])
    # print(line.split('_')[1][:5])
    else:
        target_path = target_path + 'test'
        if int(line.split(' ')[1]) == 1:
            target_path = target_path + '\\0'
        elif int(line.split(' ')[1]) == 2:
            target_path = target_path + '\\1'
        elif int(line.split(' ')[1]) == 3:
            target_path = target_path + '\\2'
        elif int(line.split(' ')[1]) == 4:
            target_path = target_path + '\\3'
        elif int(line.split(' ')[1]) == 5:
            target_path = target_path + '\\4'
        elif int(line.split(' ')[1]) == 6:
            target_path = target_path + '\\5'
        else:
            target_path = target_path + '\\6'
        # print(target_path)
        shutil.copy(source_path + 'test_' + str(line.split('_')[1][:4]) + '_aligned.jpg', target_path)
print('it is over')
