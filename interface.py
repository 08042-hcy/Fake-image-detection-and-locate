import torch
import cv2
import numpy as np
import base64
from PIL import Image ,ImageChops
from tensorflow.keras.models import load_model
from locate_model.mvssnet import get_mvss
from locate_model.common.tools import inference_single
from flask import Flask, request, make_response, jsonify


app = Flask(__name__)

classes = ["Real Image","Fake Iamge"]
model_detect = load_model('./detect/new_model_casia.h5') #篡改检测模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_locate_type = 'mvssnet'
mvssnet_path = './locate_model/mvssnet_casia.pt'
resize = 512
th = 0.5

def ELA(img_path):
    """Performs Error Level Analysis over a directory of images"""
    TEMP = './flask_images/ela_temp.jpg'  # 文件名
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
            d[x, y] = tuple(k * SCALE for k in d[x, y])  # 返回一个元组（k1*10,k2*10，...），
    return diff

@app.route('/')
def index():
    return "hello"

@app.route('/detect', methods=['POST'])
def classify():
    img = request.files.get('file')
    path = "./flask_images/detect_receive.jpeg"
    img.save(path)
    data = np.array(ELA(path).resize((128, 128))).flatten() / 255.0
    data = data.reshape(-1, 128, 128, 3)
    preds = model_detect.predict(data)
    i = preds.argmax(axis=1)[0]
    label = classes[i]
    result = "{}: {:.2f}%".format(label, preds[0][i] * 100)
    response = make_response(jsonify(result))
    response.headers["Access-Control-Allow-Origin"] = '*'
    response.headers["Access-Control-Allow-Methods"] = 'POST'
    response.headers["Access-Control-Allow-Headers"] = "x-requested-with,content-type"
    return response

@app.route('/locate', methods=['POST'])
def locate():
    img = request.files.get('file')
    path = "./flask_images/locate_receive.jpeg"
    img.save(path)
    model = get_mvss(backbone='resnet50',pretrained_base=True,nclass=1,sobel=True,constrain=True,n_input=3,)
    checkpoint = torch.load(mvssnet_path, map_location='cpu')
    model.load_state_dict(checkpoint, strict=True)
    model.to(device)
    model.eval()
    fake = cv2.imread(path)
    fake_size = fake.shape
    fake_ = cv2.resize(fake, (resize, resize))
    Image.fromarray(fake[:, :, ::-1])
    with torch.no_grad():
        fake_seg, _ = inference_single(img=fake_, model=model, th=0)
        fake_seg = cv2.resize(fake_seg, (fake_size[1], fake_size[0]))
    image_mask = Image.fromarray(fake_seg.astype(np.uint8))
    image_mask.save("./flask_images/locate.jpeg")
    with open('./flask_images/locate.jpeg', 'rb') as img_f:
        img_stream = img_f.read()
        img_stream = base64.b64encode(img_stream).decode()
    return img_stream


if __name__ == '__main__':
    app.config['JSON_AS_ASCII'] = False
    app.run(debug=True, host='192.168.43.82',port=8040)