import cv2
import torch
import torchvision                                                                 #Gerekli kütüphaneler import edildi.
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F


model_path = "model_balik_tespiti.pt"                                              #Eğitilen modelin yolu.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")              #Eğer cuda için uygun GPU varsa cuda kullanılması yoksa CPU kullanılması sağlandı.
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)     #Model olarak faster_rcnn_resnet50 seçildi.
num_classes = 2                                                                    #Sınıf sayısı olarak 2 verildi (fish ve background)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
if device.type == "cuda":
    model.load_state_dict(torch.load(model_path))
else:                                                                              #Eğitilen modelin yüklenmesi.
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.to(device)
model.eval()


class_labels = ['background', 'fish']                                              #Belirlenen sınıflar.


video_path = "Video Yolu"                                                     #Videodan balık tespiti için kullanılacak videonun yolu.
cap = cv2.VideoCapture(video_path)

frame_width = int(cap.get(3))                                                      #Ekranda açılacak pencerenin daha büyük olması, videoyu daha iyi görebilmek için ayarlar yapıldı.
frame_height = int(cap.get(4))
scale = 2.5

cv2.namedWindow("Balik Tespiti", cv2.WINDOW_NORMAL)                                #Açılacak pencerenin ismi belirlendi
cv2.resizeWindow("Balik Tespiti", int(frame_width * scale), int(frame_height * scale))   #Pencerenin büyüklüğü için belirlenen ayarlar uygulandı.


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_tensor = F.to_tensor(frame).unsqueeze(0).to(device)                        #Frame'lerin tensore dönüştürülmesi.

    
    with torch.no_grad():
        predictions = model(img_tensor)                                            #Model ile tahminlerin yaptırılması.

    
    boxes = predictions[0]['boxes'].cpu().numpy().astype(int)                      
    labels = predictions[0]['labels'].cpu().numpy()                                #Tespit edilen sonuçların (kutular,etiketler,skorların) listeye eklenmesi.
    scores = predictions[0]['scores'].cpu().numpy()

    for box, label, score in zip(boxes, labels, scores):
        if score > 0.5:                                                            #Eğer tespit edilen nesnenin skoru 0.5'den yüksekse ekranda çizdirilmesi.
            x1, y1, x2, y2 = box
            class_name = class_labels[label]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, f"{class_name}: {score:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 0), 2)


    cv2.imshow('Balik Tespiti', frame)                                             #Çıktının gösterilmesi.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
