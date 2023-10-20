# 导入所需工具包
from tensorflow.keras.models import load_model
import cv2
import numpy as np
from PIL import Image, ImageChops

#定义ELA函数
def ELA(img_path):
    """Performs Error Level Analysis over a directory of images"""
    TEMP = 'ela_' + 'temp.jpg'  # 文件名
    SCALE = 10
    original = Image.open(img_path)
    try:
        original.save(TEMP, quality=90)  # 把原始图片另存一下(将图像写入质量较低级别的 JPEG (例如 90))
        temporary = Image.open(TEMP)  # 打开原始图片(以 JPEG 格式读取图像)
        diff = ImageChops.difference(original, temporary)  # 对比解压图片和原始图片(解压后的图像与原始图像之差的绝对值)
    except:
        original.convert('RGB').save(TEMP, quality=90)  # 把解压图片转为RGB模式载另存一下
        temporary = Image.open(TEMP)  # 打开原始图片
        diff = ImageChops.difference(original.convert('RGB'), temporary)  # 对比解压图片和原始图片
    d = diff.load()  # d保存原始图片和临时图片的不同
    WIDTH, HEIGHT = diff.size  # 宽度，高度
    for x in range(WIDTH):
        for y in range(HEIGHT):
            #print(d[x, y])  # 输出每个坐标位置的differ值
            d[x, y] = tuple(k * SCALE for k in d[x, y])  # 返回一个元组（k1*10,k2*10，...），
    return diff

import tkinter
from tkinter import filedialog

'''程序会打开选择文件夹对话框 手动选中要解压的文件'''
root = tkinter.Tk()
root.withdraw()
# 获得选择好的文件
Filepath = filedialog.askopenfilename(title='请选择要转译的文件 并选择打开')
if Filepath == '':
    print('因为点击了取消，并没有拿到文件，请重新选择文件')
else:
    print(f'文件选择成功=> {Filepath} \n')

CLASSES = ['Fake image','Real image']
image = cv2.imread(Filepath)
#image = 'casia_dataset/a/Au_ani_00001.jpg'
#image = 'good_image/true/Au_ani_00001.jpg'
output = cv2.imread(Filepath)
data = np.array(ELA(Filepath).resize((128, 128))).flatten() / 255.0
data = data.reshape(-1,128,128,3)

# 读取模型和标签
print("------读取模型和标签------")
model = load_model('new_model_casia.h5')

# 预测
preds = model.predict(data)
print(preds)

# 得到预测结果以及其对应的标签
i = preds.argmax(axis=1)[0]
print(i)
classes = ["Real Image","Fake Iamge"]
label = classes[i]
# 在图像中把结果画出来
text = "{}: {:.2f}%".format(label, preds[0][i] * 100)
cv2.putText(output, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(0, 0, 255), 2)
# 绘图
cv2.imshow("Image", output)
cv2.waitKey(0)
