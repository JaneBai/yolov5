import os
import shutil
from os import listdir, getcwd
from os.path import join

def moveFile(fileDir,tarDir):
    pathDir = os.listdir(fileDir)  # 取图片的原始路径
    filenumber = len(pathDir)
    picknumber = int(filenumber * ratio)  # 按照rate比例从文件夹中取一定数量图片
    sample = random.sample(pathDir, picknumber)  # 随机选取picknumber数量的样本图片
    for name in sample:
        shutil.move(os.path.join(fileDir, name), os.path.join(tarDir, name))
    return

if __name__ == '__main__':
    source_imgfolder='D:/4_code/GitHub_Open/yolov5/data/JuBan/images/'#地址是所有图片的保存地点
    source_labelfolder='D:/4_code/GitHub_Open/yolov5/data/JuBan/labels/'#地址是所有图片的保存地点
    dest='data/JuBan/train.txt' #保存train.txt的地址
    dest2='data/JuBan/val.txt'  #保存val.txt的地址
    if os.path.exists(dest):
        os.remove(dest)
    if os.path.exists(dest2):
        os.remove(dest2)
    train_imgfolder=os.path.join(source_imgfolder, 'train')
    val_imgfolder=os.path.join(source_imgfolder, 'val')
    train_labelfolder=os.path.join(source_labelfolder, 'train')
    val_labelfolder=os.path.join(source_labelfolder, 'val')
    train_file=open(dest,'a')                 #打开文件
    val_file=open(dest2,'a')                  #打开文件
    listImgDir = os.listdir(source_imgfolder)
    listLabelDir=os.listdir(source_labelfolder)
    if not os.path.exists(train_imgfolder):  # 如果val下没有子文件夹，就创建
        os.makedirs(train_imgfolder)
    if not os.path.exists(val_imgfolder):  # 如果val下没有子文件夹，就创建
        os.makedirs(val_imgfolder)
    if not os.path.exists(train_labelfolder):  # 如果val下没有子文件夹，就创建
        os.makedirs(train_labelfolder)
    if not os.path.exists(val_labelfolder):  # 如果val下没有子文件夹，就创建
        os.makedirs(val_labelfolder)
    ##判断图像和标签数目是否相等
    imgNumber=len(os.listdir(source_imgfolder))
    lableNumber=len(os.listdir(source_labelfolder))
    print('imgNumber:{},labelNumber:{}'.format(imgNumber,lableNumber))
    ##先按照4：1的比例进行拆分
    file_imgnum = 0
    file_labelnum = 0
    for file_obj in listImgDir:                #访问文件列表中的每一个文件
        file_path=os.path.join(source_imgfolder,file_obj) 
        #file_name 保存文件的名字，file_extend保存文件扩展名 
        file_name,file_extend=os.path.splitext(file_obj)
        if(file_imgnum%4==0):                #前面是你的图片数目，后面是为了保留文件用于训练
            val_file.write(os.path.join(val_imgfolder, file_obj)+'\n')  #用于训练前900个的图片路径保存在train.txt里面，结尾加回车换行
            shutil.copy(file_path, val_imgfolder)
        else :
            train_file.write(os.path.join(train_imgfolder, file_obj)+'\n')    #其余的文件保存在val.txt里面
            shutil.copy(file_path, train_imgfolder)
        file_imgnum+=1
    for file_obj in listLabelDir:
        file_path=os.path.join(source_labelfolder,file_obj) 
        if(file_labelnum%4==0):
            shutil.copy(file_path, val_labelfolder)
        else:
            shutil.copy(file_path, train_labelfolder) 
        file_labelnum+=1
    train_file.close()#关闭文件
    val_file.close()
    #trainval_file.close()