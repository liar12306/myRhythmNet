import cv2
from src.utils.seetaface import api
init_mask = api.FACE_DETECT|api.LANDMARKER68

def start():
    seetaFace = api.SeetaFace(init_mask)

    vc = cv2.VideoCapture(0)
    while(True):
        rval, frame = vc.read()
        if rval:
            faces = seetaFace.Detect(frame)
            for i in range(faces.size):
                face = faces.data[i].pos
                landmarks = seetaFace.mark68(frame,face)
                cv2.rectangle(frame, (face.x, face.y), (face.x + face.width, face.y + face.height), (255, 0, 0),2)
                for j in range(68):
                    # 画关键点
                    cv2.circle(frame, (int(landmarks[j].x), int(landmarks[j].y)), 1, (255, 0, 0), -1)
        else:
            break
        cv2.imshow('frame',frame)
        if cv2.waitKey(1)&0xFF == ord('q'):
            break
    vc.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    start()