#训练自己的数据
##1)编写myData.ymal,如下：
###标签文件分布： labes/train， labels/val 
REM train: ./data/JuBan/images/train/  # 16551 images
REM val: ./data/JuBan/images/val  # 4952 images

REM # number of classes
REM nc: 2

REM # class names
REM names: [ 'full', 'broken' ]
##2)修改对应模型的yaml文件，将nc改为实际的类别数即可
##3)进行训练
python train.py --data myData.yaml --weights yolov5s.pt --batch-size 4 --device 0
