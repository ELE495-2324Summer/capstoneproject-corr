import cv2
import time
from math import sqrt
from jetbot import Robot  # JetBot kütüphanesini import et
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import requests
import json

robot = Robot()
robot.stop()
park_started = 0
measured_value = 0
duvara_geldi_mi = 0
flag = 0
baslangic = 0
ilk_rakam_yonelimi = 0
park_started = 0
frame_counter = 0
red_flag = 0
a_little_more = 0

device = torch.device("cpu")
scripted_model = torch.jit.load('70x70_14.07.pt', map_location=device)
scripted_model.eval()

# Görüntü dönüşüm fonksiyonunu tanımlayın
mean = 0.37
std = 0.37

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),  # 3 kanallı gri tonlama
    transforms.Resize((70, 70)),  # 70x70 boyutuna yeniden boyutlandırma
    transforms.ToTensor(),
    transforms.Normalize((mean,), (std,))
])

def transform_image(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 105, 255, cv2.THRESH_BINARY_INV)
    binary_rgb = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)  # 3 kanallı gri tonlama
    pil_image = Image.fromarray(binary_rgb)
    tensor = transform(pil_image).to(device)
    return tensor.unsqueeze(0)

def send_number_to_server(number):
    url = "http://192.168.95.88:5000/send_number"
    payload = {
        "number": number
    }

    try:
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            print("Sayı başarıyla gönderildi.")
        else:
            print("Sayı gönderme hatası:", response.status_code)
    except Exception as e:
         print("Bir hata oluştu:", str(e))
send_number_to_server(42)        
def get_plate_from_server():
    url = "http://192.168.95.88:5000/get_plate"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            plate_str = data.get('plate', '1111')  # Varsayılan olarak '0' kullanılır
            try:
                plate = int(plate_str)  # Değeri tam sayıya dönüştür
                print("Alınan plaka (int):", plate)
            except ValueError:
                print("Plaka değeri bir tam sayı değil:", plate_str)
        else:
            print("Plaka alma hatası:", response.status_code)
    except Exception as e:
        print("Bir hata oluştu:", str(e))
    return plate
def update_park_status(status):
    url = "http://192.168.95.88:5000/park_complete"  # Flask sunucusunun park durumu güncelleme endpoint'i
    payload = {
        "status": status  # Gönderilecek park durumu
    }

    try:
        response = requests.post(url, json=payload)  # POST isteği gönder
        if response.status_code == 200:  # Başarılı yanıt kontrolü
            print("Park durumu başarıyla güncellendi.")
        else:
            print("Park durumu güncelleme hatası:", response.status_code)  # Hata durumunda yanıt kodunu yazdır
    except Exception as e:
        print("Bir hata oluştu:", str(e))  # İstisna durumunda hata mesajını yazdır

# Test için park durumunu güncelle



class PIDController:
    def __init__(self, Kp, Ki, Kd, integral_limit=100):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.prev_error = 0
        self.integral = 0
        self.integral_limit = integral_limit

    def update(self, setpoint, measured_value):
        error = setpoint - measured_value
        self.integral += error
        # Integral windup'ı önlemek için sınır koyma
        if self.integral > self.integral_limit:
            self.integral = self.integral_limit
        elif self.integral < -self.integral_limit:
            self.integral = -self.integral_limit
        derivative = error - self.prev_error
        self.prev_error = error
        # PID denklemi
        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        return output

def move_for_duration(robot, left_speed, right_speed, duration):
    start_time = time.time()
    while time.time() - start_time < duration:
        robot.left(left_speed)
        robot.right(right_speed)
    robot.stop()

def wait_for_duration(duration):
    start_time = time.time()
    while time.time() - start_time < duration:
        pass

def turn_back(robot):
    move_for_duration(robot, 0.28, -0.28, 0.468)
    wait_for_duration(0.5)

def turn_left(robot):
    move_for_duration(robot, 0.28, -0.28, 0.235)
    wait_for_duration(0.5)

def turn_right(robot):
    move_for_duration(robot, -0.28, 0.28, 0.472)
    wait_for_duration(0.5)

def go_forward(robot):
    move_for_duration(robot, 0.091, 0.0907, 2.25)
    wait_for_duration(0.5)

def approach_digit(frame):
    # Rakamın boyutuna göre durma
    digit_size = np.sum(frame[:, :, 0] == 255)
    print("digit_size = ", digit_size)
    if digit_size > 5000:  # Bu değer, deneme yanılma ile ayarlanabilir
        robot.stop()
        
        print("Hedefe ulaşıldı.")
        update_park_status("complete")

        return -1
    return 0
k = 0
def black_to_white(frame, robot):
    # Görüntüyü kırmızı tonlarına duyarlı bir maske ile işleyin
    # HSV renk uzayına dönüştürme
    global k
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Koyu kırmızı renk aralıklarını belirleme
    lower_dark_red1 = np.array([0, 50, 50])
    upper_dark_red1 = np.array([10, 255, 150])
    lower_dark_red2 = np.array([170, 50, 50])
    upper_dark_red2 = np.array([180, 255, 150])

    # Maskeleri oluşturma
    mask1 = cv2.inRange(hsv, lower_dark_red1, upper_dark_red1)
    mask2 = cv2.inRange(hsv, lower_dark_red2, upper_dark_red2)
    mask = mask1 + mask2

    # Görüntüyü kırpma
    height, width = frame.shape[:2]
    left = int(0.2 * width)
    right = int(0.8 * width)
    cropped_image = frame[:, left:right]

    # Maskeyi kırpılmış görüntüye uygulama
    cropped_hsv = hsv[:, left:right]
    mask1_cropped = cv2.inRange(cropped_hsv, lower_dark_red1, upper_dark_red1)
    mask2_cropped = cv2.inRange(cropped_hsv, lower_dark_red2, upper_dark_red2)
    mask_cropped = mask1_cropped + mask2_cropped

    # Sonuç görüntüsünü oluşturma
    result = np.zeros_like(cropped_image)  # Tüm görüntüyü siyah yap
    result[mask_cropped != 0] = [255, 255, 255]  # Kırmızı alanları beyaza çevir
    if(k < 20):
        
        
        cv2.imwrite(f"{k}abc.jpg",result)
        print("bastim")
    k = k+1
    # Beyaz piksellerin ortalama koordinatını hesaplama
    white_pixels = np.column_stack(np.where(result[:, :, 0] == 255))
    if white_pixels.size == 0:
        print("beyaz piksel yok")
        return -1  # Hiç beyaz piksel yoksa -1 döndür

    centroid_x = np.mean(white_pixels[:, 1])
    centroid_y = np.mean(white_pixels[:, 0])
    centroid = (int(centroid_x), int(centroid_y))

    # Kesilen görüntünün merkezini hesaplama
    cropped_center_x = (right - left) // 2
    cropped_center_y = height // 2
    cropped_center = (cropped_center_x, cropped_center_y)

    # Sapmayı hesaplama (merkezden ağırlık noktasına olan uzaklık)
    deviation_x = centroid[0] - cropped_center[0]

    if(abs(deviation_x)<10):
        return deviation_x*3
    elif(abs(deviation_x)<30):
        return 6*deviation_x
    else:
        return 8*deviation_x

def check_white_pixels(processed_image):
    # Beyaz piksel olup olmadığını kontrol et
    if np.sum(processed_image == 255) == 0:
        return 1  # Flag bitini 1 yap
    else:
        return 0  # Flag bitini 0 yap

def siyah_algilayici(image):
    # Siyah ve tonları (gri dahil) beyaza, diğer renkler siyaha dönüşecek
    lower_gray = np.array([0, 0, 0])
    upper_gray = np.array([100, 100, 100])

    # Maskeyi oluştur
    mask = cv2.inRange(image, lower_gray, upper_gray)

    # Maskeyi kullanarak siyah ve gri tonlarını beyaza çevir
    processed_image = np.zeros_like(image)
    processed_image[mask == 255] = [255, 255, 255]
    a = check_white_pixels(processed_image)
    return a

# Initialize PID controller with specific gains
pid = PIDController(Kp=0.15, Ki=0.1, Kd=0.05, integral_limit=100)

# Klasör oluşturma
output_dir = 'abc'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# GStreamer pipeline for the camera
gst_pipeline = (
    "nvarguscamerasrc ! "
    "video/x-raw(memory:NVMM), width=640, height=480, format=NV12, framerate=30/1 ! "
    "nvvidconv flip-method=2 ! "
    "video/x-raw, width=640, height=480, format=BGRx ! "
    "videoconvert ! "
    "video/x-raw, format=BGR ! appsink"
)

# OpenCV VideoCapture object with GStreamer pipeline
camera = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
start_time = time.time()
"""
while time.time() - start_time < 5:
    ret,frame = camera.read()
"""
image_counter = 0

def get_jetbot_position(ret, frame):
    global frame_counter
    global image_counter

    if not ret:
        return 0  # Kameradan görüntü alınamazsa merkezi döndür

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)

    height, width = thresh.shape
    center_of_frame = width / 2

    # Görüntüyü sağdan ve soldan %20 kes

    # Üstten %20'sini karart
    crop_height = int(height * 0.20)
    thresh[:crop_height, :] = 0

    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest_contour)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
        else:
            cx = center_of_frame
    else:
        cx = center_of_frame

    deviation = cx - center_of_frame
    if deviation < -30 and deviation > -30:
        return deviation * 1.5
    elif deviation < 100 and deviation > -100:
        return deviation * 3
    else:
        return deviation * 5

    
def adjust_jetbot_motors_red(control_output):
    global duvara_geldi_mi
    global flag
    base_speed = 0.10  # JetBot'un temel hızı
    
    if ( 1 == 0):
        print("k")
    else:
        if control_output == 0:
            left_motor_speed = base_speed
            right_motor_speed = base_speed
        else:
            isaret = abs(control_output) / control_output
            left_motor_speed = (base_speed - isaret * sqrt(abs(control_output)) / 1000.0)
            right_motor_speed = (base_speed + isaret * sqrt(abs(control_output)) / 1000.0)

        left_motor_speed = max(min(left_motor_speed, 0.5), -0.5)
        right_motor_speed = max(min(right_motor_speed, 0.5), -0.5)

        robot.set_motors(left_motor_speed, right_motor_speed)
        
def adjust_jetbot_motors(control_output, speed):
    global duvara_geldi_mi
    global flag
    base_speed = speed  # JetBot'un temel hızı

    if duvara_geldi_mi == 1:
        # Duvara çarptıysa 180 derece dön
        turn_back(robot)
        flag = 1
    else:
        if control_output == 0:
            right_motor_speed = base_speed
            left_motor_speed = base_speed
        else:
            isaret = abs(control_output) / control_output
            left_motor_speed = (base_speed + isaret * sqrt(abs(control_output)) / 1000.0 + 0.007)
            right_motor_speed = (base_speed - isaret * sqrt(abs(control_output)) / 1000.0)

        left_motor_speed = max(min(left_motor_speed, 0.5), -0.5)
        right_motor_speed = max(min(right_motor_speed, 0.5), -0.5)

        robot.set_motors(left_motor_speed, right_motor_speed)

frame_counter = 0

def follow_line_for_duration(duration, st):
    start_time = time.time()
    frame_counter = 0
    global flag
    global camera
    global duvara_geldi_mi
    global ilk_rakam_yonelimi
    global baslangic
    global measured_value
    while time.time() - start_time < duration:
        if(flag == 1):
            camera = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
            flag = 0
        ret, frame = camera.read()
        setpoint = 0  # Siyah çizginin merkezi
        if(st == "black"):
            measured_value = get_jetbot_position(ret, frame)  # JetBot'un mevcut konumu

            duvara_geldi_mi = siyah_algilayici(frame)
            if(duvara_geldi_mi == 1):
                camera.release()
                baslangic = 1
                
                flag = 1
                adjust_jetbot_motors(measured_value, 0.11)
                
                return 0
            adjust_jetbot_motors(measured_value, 0.11)

            if frame_counter % 50 == 0:
                print(f"Deviation: {measured_value}")

            frame_counter += 1
        
        elif(st == "red"):
            measured_value = black_to_white(frame, robot)

            if measured_value != -1:
                adjust_jetbot_motors(measured_value, 0.1)

            if measured_value == -1:
                break

def follow_line_for_duration_red(duration):
    start_time = time.time()
    global measured_value
    global frame_counter
    global flag
    global camera
    global duvara_geldi_mi
    while time.time() - start_time < duration:
        if flag == 1:
            camera = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
            flag = 0
        ret, frame = camera.read()
        measured_value = black_to_white(frame, robot)
        
        if duvara_geldi_mi == 1:
            camera.release()
            flag = 1
        
        if measured_value != -1:
            adjust_jetbot_motors_red(0.1)
        
        if measured_value == -1:
            break
        
        frame_counter += 1
pid = PIDController(Kp=0.15, Ki=0.1, Kd=0.05, integral_limit=100)

# Örnek kullanım: siyah çizgiyi 10 saniye boyunca takip et
i = 0
def first_tilt(robot):
    global camera
    ret, frame = camera.read()
    tilt_value = calculate_deviation(frame, robot)
    isaret = abs(tilt_value)/tilt_value
    robot.left(0.15)
    robot.right(-0.15)    
    time.sleep(abs(tilt_value)/2900)
    robot.stop()
    print(tilt_value)
    
def park_etme():
    global baslangic
    global ilk_rakam_yonelimi
    global i
    while True:
        b=time.time()


        follow_line_for_duration_red(0.08)

        robot.stop()
        time.sleep(0.1)
        i = i+1
        a=time.time()
        print(b-a)
        if((i > 120 or measured_value==-1) and a_little_more == 0):
            robot.forward(0.13)
            time.sleep(0.4)
            ilk_rakam_yonelimi = 0
            baslangic = 0
            robot.stop()
            update_park_status("complete")
            break
                
a = get_plate_from_server()
def run_inference():
    i = 0
    while True:
        i = i + 1
        ret, frame = camera.read()
        tensor_image = transform_image(frame)
        with torch.no_grad():
            output = scripted_model(tensor_image)
            probs, indices = torch.topk(output, 1)
            probs = torch.exp(probs)
            probs, indices = probs.cpu().numpy()[0], indices.cpu().numpy()[0]
        if i == 20:
            print(indices[0])
            return indices[0]


    
while True:
    #£plate = get_plate_from_server()
    while True:
        plate = get_plate_from_server()
        if (a != plate):
            update_park_status("incomplete")
            if(baslangic == 0):
                i = 0
                while True:
                    i = i + 1
                    follow_line_for_duration(100, "black")
                    robot.stop()
                    wait_for_duration(0.2)
                    if(i == 1000):
                        break
                    if(duvara_geldi_mi == 1):
                        break
            elif((baslangic == 1) and (ilk_rakam_yonelimi == 0)):
                i = 0
                while True:
                    follow_line_for_duration(0.1, "black")
                    robot.stop()
                    if(i == 18):
                        break
                    i = i + 1
                    wait_for_duration(0.2)
                turn_left(robot)
                ilk_rakam_yonelimi += 1

                if(run_inference() == plate):
                    park_etme()
                    a = plate
                    break
                if(measured_value == -1):
                    break
                turn_back(robot)

                if(run_inference() == plate):
                    park_etme()
                    a = plate
                    break
                turn_left(robot)
            elif(baslangic == 1 and ilk_rakam_yonelimi < 20):
                i = 0
                while True:
                    follow_line_for_duration(0.0977, "black")
                    if(i == 18):
                        break
                    i = i + 1
                    robot.stop()
                    wait_for_duration(0.2)
                turn_left(robot)
                ilk_rakam_yonelimi += 1

                if(run_inference() == plate):
                    park_etme()
                    a = plate
                    break
                turn_back(robot)
                if(run_inference() == plate):
                    park_etme()
                    a = plate
                    break
                turn_left(robot)
            elif(ilk_rakam_yonelimi == 5):
                ilk_rakam_yonelimi += 1
                follow_line_for_duration(100, "black")
                baslangic = 0

robot.stop()
camera.release()
