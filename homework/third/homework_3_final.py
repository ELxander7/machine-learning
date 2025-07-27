import threading # для многопоточности (чтобы не блокировать основной поток при проверке лица)
import cv2 # opencv для работы с видео и изображениями
from deepface import DeepFace # для анализа лиц и эмоция
import mediapipe as mp # для детекции рук и пальцев

# инициализация видеозахвата с камеры
win = cv2.VideoCapture(0, cv2.CAP_DSHOW)
win.set(cv2.CAP_PROP_FRAME_WIDTH, 640) # ширина кадра
win.set(cv2.CAP_PROP_FRAME_HEIGHT, 480) # высота кадра

# инициализация необходимых переменных для распознавания лица
counter = 0 # счётчик кадров (чтобы не проверять лицо каждый кадр)
face_match = False # флаг совпадения лица
#ref_img = cv2.imread("my_face.jpeg")
#ref_img = cv2.imread("happy.jpeg")
ref_img = cv2.imread("fear.jpeg")

# инициализация haar cascade для детекции лиц
cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath) # загрузка модели

# инициализация mediapipe для работы с руками
mp_drawing = mp.solutions.drawing_utils # утилита для рисования landmarks
mp_hands = mp.solutions.hands # модуль для работы с руками
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) # создание детектора рук

# данные пользователя
user_data = {
    "name": "Eldar",
    "surname": "Armanov"
}

# функция для проверки совпадения лица с желаемым
def func_face(frame):
    global face_match


    try:
        if DeepFace.verify(frame, ref_img.copy())['verified']: # сравнивает лица
            face_match = True
        else:
            face_match = False
    except ValueError: # если пальцев не обнаружено
        face_match = False

# функция для подсчёта поднятых пальцев
def count_fingers(hand_landmarks):
    tip_ids = [4, 8, 12, 16, 20] # landmark ids кончиков пальцев
    finger_states = [] # состояние пальцев (поднят/опущен)

    for tip_id in tip_ids:
        finger_tip = hand_landmarks.landmark[tip_id] # координаты кончика пальца
        finger_mcp = hand_landmarks.landmark[tip_id - 1] # координаты сустава

        if tip_id == 4: # большой палец (проверка по Х координате)
            finger_states.append(finger_tip.x < finger_mcp.x)
        else: # Остальные пальцы (проверка по Y координате)
            finger_states.append(finger_tip.y < finger_mcp.y)

    return finger_states.count(True) # количество поднятых пальцев

# основной цикл обработки видео
while True:
    ret, frame = win.read() # чтение кадра с камеры

    if ret: # если кадр прочитан успешно

        # зеркально отображает
        frame = cv2.flip(frame, 1)

        # конвертирует в rgb для mediapipe (для работы модели)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # детекция рук с помощью mediapipe
        hand_results = hands.process(rgb_frame)

        # подсчёт пальцев
        finger_count = 0
        if hand_results.multi_hand_landmarks: # если руки обнаружены
            hand_landmarks = hand_results.multi_hand_landmarks[0] # берёт первую руку
            finger_count = count_fingers(hand_landmarks) # считает пальцы

            # отрисовка landmarks рук (точек и соединений)
            for hand_landmarks in hand_results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                )

        # детекция лиц с помощью haar cascade
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1, # параметр масштабирования
            minNeighbors=5, # минимальное количество соседей
            minSize=(30, 30) # минимальный размер лица
        )

        # проверка совпадения лица каждые 30 кадров (чтобы не нагружать систему)
        if counter % 30 == 0:
            try:
                # запуск в отдельном потоке для избежания лагов
                threading.Thread(target=func_face, args=(frame.copy(),)).start()
            except ValueError:
                pass
        counter +=1 # инкремент счётчика кадров

        # обработка информации в зависимости от количества пальцев
        info_text = ""
        if face_match and len(faces) > 0: # если лицо совпало и обнаружено
            (x, y, w, h) = faces[0] # координаты первого лица

            if finger_count == 1:
                info_text = f"Name: {user_data['name']}"
            elif finger_count == 2:
                info_text = f"Surname: {user_data['surname']}"
            elif finger_count == 3:
                face_roi = frame[y:y + h, x:x + w] # область лица
                try:
                    face_roi_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
                    result = DeepFace.analyze(face_roi_rgb, actions=['emotion'], enforce_detection=False)
                    info_text = f"Emotion: {result[0]['dominant_emotion']}"
                except Exception as e:
                    print(f"Error in emotion detection: {e}")
                    info_text = "Emotion: Unknown"

            # отображение информации над лицом
            if info_text:
                cv2.putText(frame, info_text, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        # рисует прямоугольники вокруг обнаруженных лиц
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # отображение результата проверки лица
        if face_match:
            cv2.putText(frame, "MATCH!", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
        else:
            cv2.putText(frame, "NO MATCH!", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

        # отображение количества пальцев
        cv2.putText(frame, f'Fingers: {finger_count}', (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # инструкция пользования
        cv2.putText(frame, '1 finger - Name, 2 - Surname, 3 - Emotion', (50, 400),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # отображение кадра в окне
        cv2.imshow("Video", frame)

    # выход
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

# освобождение ресурсов
win.release() # закрывает видеопоток
cv2.destroyAllWindows() # закрывает все окна opencv
hands.close() # закрывает детектор рук mediapipe