import numpy as np
import cv2
import win32ui
import torch
from repvgg import create_RepVGG_A0
from torchvision import transforms


def openimage():
    modelpath = "./utils/opencv_face_detector_uint8.pb"
    weightpath = "./utils/opencv_face_detector.pbtxt"

    emo_labels = ['Surprise', 'Fear', 'Disgust', 'Happy', 'Sad', 'Anger', 'Neutral']

    deploy_model = create_RepVGG_A0(deploy=True)
    # deploy_model.load_state_dict(torch.load('train_model_20.pth'))
    deploy_model.load_state_dict({k.replace('module.', ''): v for k, v in torch.load('./utils/deploy_model_30.pth').items()})

    trans = transforms.Compose([transforms.ToPILImage(),
                                transforms.Resize((224, 224)),
                                transforms.ToTensor()])
    # 置信度参数，高于此数才认为是人脸，可调
    confidence = 0.85
    font = cv2.FONT_HERSHEY_SIMPLEX
    net = cv2.dnn.readNetFromTensorflow(modelpath, weightpath)

    dlg = win32ui.CreateFileDialog(1)
    dlg.SetOFNInitialDir('Y:\\imgprocess')
    dlg.DoModal()

    imgname = dlg.GetPathName()
    if imgname:
        img = cv2.imread(imgname)

        (h, w) = img.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0, (300, 300), (0, 0, 0))
        net.setInput(blob)
        detections = net.forward()
        for i in range(0, detections.shape[2]):
            res_confidence = detections[0, 0, i, 2]
            if res_confidence > confidence:
                # 获得框的位置
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                image = img[startY: endY, startX: endX]

                image = trans(image)
                image = torch.unsqueeze(image, dim=0)
                deploy_model.eval()
                out = deploy_model(image)
                prediction = torch.argmax(out, 1)

                emo = emo_labels[int(prediction)]
                print('Emotion : ', emo)

                cv2.rectangle(img, (startX, startY), (endX, endY), (0, 255, 0), 2)
                cv2.putText(img, '%s' % emo, (startX + 30, startY + 30), font, 1, (144, 238, 144), 2)
        cv2.imshow("image", img)
        cv2.waitKey(0)
    else:
        print("请输入图片")


if __name__ == '__main__':

    openimage()


