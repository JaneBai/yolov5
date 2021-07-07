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

##4)推理
python detect.py --source data/JuBan/images/train --weights best.pt --conf 0.25

##5)训练的模型转为onnx
python export.py --weights yolov5s.pt --img 640 --batch 1 --opset-version 10
##记得加 --train
python export.py --train --weights best_juban.pt --img 640 --batch 1 --opset-version 10

##6）用openvino推理。
##步骤：首先启动openvino虚拟环境，然后cd C:\Program Files (x86)\Intel\openvino_2021.4.582\bin下 执行setupvars.bat，启动openvino环境，再运行下面语句
python .\mo.py  --input_model D:\4_code\GitHub_Open\yolov5\yolov5s.onnx --model_name e:/yolov5s -s 255 --reverse_input_channels --output Conv_339,Conv_291,Conv_243
python .\mo.py  --input_model D:\4_code\GitHub_Open\yolov5\best_juban.onnx --model_name e:/best_juban -s 255 --reverse_input_channels --output Conv_289,Conv_269,Conv_249
##7)推理(openvino2021.2以后) 注意：图像名称要加上序号，否则opencv vide读取有异常。yolov5_demo_OV2021
python yolov5_demo_OV2021.3.py -m openvino\yolov5s.xml -i zidane001.jpg
python yolo_openvino_demo.py -m openvino\yolov5s.xml -i zidane001.jpg -at yolov5
python yolo_openvino_demo.py -m openvino\best_juban.xml -i data\JuBan\images\125.png -at yolov5