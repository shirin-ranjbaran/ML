import cv2
import numpy as np

# بارگیری مدل پیش آموزش دیده برای تشخیص علف‌های هرز
model = cv2.dnn.readNetFromCaffe("path/to/caffe/deploy.prototxt", "path/to/caffe/model.caffemodel")

# تنظیم تنظیمات ورودی و خروجی مدل
input_size = (300, 300)  # اندازه ورودی مدل
scale_factor = 1.0       # فاکتور مقیاس‌بندی تصویر ورودی

# بارگیری ویدئو
video = cv2.VideoCapture("path/to/video.mp4")

while True:
    ret, frame = video.read()
    if not ret:
        break

    # تغییر اندازه تصویر ورودی
    resized_frame = cv2.resize(frame, input_size)

    # ساخت بلوب از تصویر ورودی
    blob = cv2.dnn.blobFromImage(resized_frame, scale_factor, input_size, (104, 177, 123))

    # اجرای تشخیص علف‌های هرز بر روی بلوب
    model.setInput(blob)
    detections = model.forward()

    # نمایش نتایج تشخیص
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:  # تعیین آستانه اطمینان برای تشخیص
            # محاسبه مستطیل محدود کننده علف هرز
            box = detections[0, 0, i, 3:7] * np.array([input_size[0], input_size[1], input_size[0], input_size[1]])
            (startX, startY, endX, endY) = box.astype("int")

            # نمایش مستطیل و علف هرز تشخیص داده شده در تصویر اصلی
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

    # نمایش ویدئو با تشخیص علف‌های هرز
    cv2.imshow("Weed Detection", frame)
    if cv2.waitKey(1) == ord('q'):
        break

# آزاد کردن منابع
video.release()
cv2.destroyAllWindows()