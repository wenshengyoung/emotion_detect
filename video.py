import numpy as np
import cv2
import torch
from torchvision import transforms
from repvgg import create_RepVGG_A0


def video():

    modelpath = "./utils/opencv_face_detector_uint8.pb"
    weightpath = "./utils/opencv_face_detector.pbtxt"

    emo_labels = ['Surprise', 'Fear', 'Disgust', 'Happy', 'Sad', 'Anger', 'Neutral']
    deploy_model = create_RepVGG_A0(deploy=True)
    deploy_model.load_state_dict({k.replace('module.', ''): v for k, v in torch.load('./utils/deploy_model_30.pth').items()})

    trans = transforms.Compose([transforms.ToPILImage(),
                                transforms.Resize((224, 224)),
                                transforms.ToTensor()])
    # 置信度参数，高于此数才认为是人脸，可调
    confidence = 0.85
    font = cv2.FONT_HERSHEY_SIMPLEX
    cam = cv2.VideoCapture(0)
    net = cv2.dnn.readNetFromTensorflow(modelpath, weightpath)
    while True:
        ret, img = cam.read()
        img = cv2.flip(img, 1)
        (h, w) = img.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0, (300, 300), (0, 0, 0))
        net.setInput(blob)
        # 预测结果
        detections = net.forward()
        for i in range(0, detections.shape[2]):
            # 获得置信度
            res_confidence = detections[0, 0, i, 2]
            # 过滤掉低置信度的像素
            if res_confidence > confidence:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                if startX < 10 or startY < 10:
                    continue
                image = img[startY: endY, startX: endX]
                image = trans(image)

                image = torch.unsqueeze(image, dim=0)
                deploy_model.eval()
                out = deploy_model(image)
                prediction = torch.max(out, 1)
                value = torch.tensor(prediction).detach().numpy()[0]
                indice = torch.tensor(prediction).detach().numpy()[1]
                value = "{:.2f}".format(value)

                emo = emo_labels[int(indice)]
                print('Emotion : ', emo)
                y = startY + 14
                cv2.rectangle(img, (startX, startY), (endX, endY), (0, 255, 0), 2)
                cv2.putText(img, value, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
                cv2.putText(img, '%s' % emo, (startX - 20, startY - 20), font, 0.65, (0, 255, 0), 2)

        cv2.imshow('expression classification', img)
        k = cv2.waitKey(10) & 0xff
        if k == 27:
            break
    cam.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':

    video()
