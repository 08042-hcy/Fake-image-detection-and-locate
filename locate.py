import torch
import cv2
import numpy as np
from PIL import Image
from locate_model.mvssnet import get_mvss
from locate_model.resfcn import ResFCN
from locate_model.common.tools import inference_single
#from common.utils import calculate_pixel_f1
#from apex import amp
import tkinter
from tkinter import filedialog

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

'''程序会打开选择文件夹对话框 手动选中要解压的文件'''
root = tkinter.Tk()
root.withdraw()
# 获得选择好的文件
Filepath = filedialog.askopenfilename(title='请选择要转译的文件 并选择打开')
if Filepath == '':
    print('因为点击了取消，并没有拿到文件，请重新选择文件')
else:
    print(f'文件选择成功=> {Filepath} \n')

model_type = 'mvssnet'
mvssnet_path = '.\locate_model\mvssnet_casia.pt'
resfcn_path = '\locate_model\resfcn_casia.pt'
resize = 512
th = 0.5

if model_type == 'mvssnet':
    model = get_mvss(backbone='resnet50',
                     pretrained_base=True,
                     nclass=1,
                     sobel=True,
                     constrain=True,
                     n_input=3,
                     )
    checkpoint = torch.load(mvssnet_path, map_location='cpu')
elif model_type == 'fcn':
    model = ResFCN()
    checkpoint = torch.load(resfcn_path, map_location='cpu')

model.load_state_dict(checkpoint, strict=True)
model.to(device)
model.eval()

fake = cv2.imread(Filepath)
fake_size = fake.shape
fake_ = cv2.resize(fake, (resize, resize))
Image.fromarray(fake[:,:,::-1])

with torch.no_grad():
    fake_seg, _ = inference_single(img=fake_, model=model, th=0)
    fake_seg = cv2.resize(fake_seg, (fake_size[1], fake_size[0]))

Image.fromarray((fake_seg).astype(np.uint8)).show()


