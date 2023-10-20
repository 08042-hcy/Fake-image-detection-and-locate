# Fake-image-detection-and-locate
Python实现伪造图片的检测和伪造区域的定位，Vue封装在网页端提供用户上传图片的接口。

相关文件说明如下：
1.detect文件夹：包含用于真伪图像二分类检测的训练测试代码，生成的预训练模型保存在new_model_casia.h5中。
2.locate_model文件夹：包含两个不同的图像定位模型，训练代码在mvssnet.py和resfcn.py文件中，生成的预训练模型保存在mvssnet_casia.pt和resfcn_casia.pt中。
3.good_images文件夹：包含前端测试的图片，true里面是非伪造图片，copy-move里面是移动类型篡改图片，splicing里面是拼接类型篡改图片。
4.interface.py：接口文件，用与前后端交互。
5.locate.py：后端测试文件，不需要前端网页，直接在后端检测定位模型的检测结果。

