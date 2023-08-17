import cv2

# تعیین مسیر فایل ویدئو
video_path = "path/to/video.mp4"

# بارگیری مدل پیش‌آموزش‌دیده برای تشخیص علف‌های هرز
model = cv2.dnn.readNetFromCaffe("path/to/deploy.prototxt", "path/to/model.caffemodel")

# تنظیم تنظیمات ورودی و خروجی مدل
input_size = (300, 300)  # اندازه تصویر ورودی برای مدل
scale_factor = 1.0       # فاکتور مقیاس‌بندی تصویر ورودی

# بارگیری ویدئو
video = cv2.VideoCapture(video_path)

while True:
    ret, frame = video.read()
    if not ret:
        break

    # تغییر اندازه تصویر ورودی برای مدل
    resized_frame = cv2.resize(frame, input_size)

    # ایجاد بلوب از تصویر ورودی برای مدل
    blob = cv2.dnn.blobFromImage(resized_frame, scale_factor, input_size, (104, 177, 123))

    # اجرای تشخیص علف‌های هرز بر روی تصویر
    model.setInput(blob)
    detections = model.forward()

    # نمایش نتایج تشخیص در تصویر اصلی
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:  # آستانه اطمینان برای تشخیص
            # محاسبه مستطیل محدودکننده علف هرز
            box = detections[0, 0, i, 3:7] * np.array([input_size[0], input_size[1], input_size[0], input_size[1]])
            (startX, startY, endX, endY) = box.astype("int")

            # رسم مستطیل حول علف هرز در تصویر اصلی
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

    # نمایش ویدئو با علف‌های هرز تشخیص داده شده
    cv2.imshow("Weed Detection", frame)
    if cv2.waitKey(1) == ord('q'):
        break

# آزاد کردن منابع
video.release()
cv2.destroyAllWindows()