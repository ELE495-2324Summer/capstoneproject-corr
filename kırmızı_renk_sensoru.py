import Jetson.GPIO as GPIO
import time
import signal
import sys

# Pin tanımlamaları (BCM modunda)
S0 = 21  # GPIO pin numaralarınızı girin
S1 = 19
S2 = 35
S3 = 37
OUT = 23
LED = 24  # LED pininizin GPIO numarasını girin

# GPIO modunu ve pinleri ayarlayın
GPIO.setmode(GPIO.BCM)
GPIO.setup(S0, GPIO.OUT)
GPIO.setup(S1, GPIO.OUT)
GPIO.setup(S2, GPIO.OUT)
GPIO.setup(S3, GPIO.OUT)
GPIO.setup(OUT, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.setup(LED, GPIO.OUT)

# LED kontrol fonksiyonu
def control_led(state):
    if state:
        GPIO.output(LED, GPIO.HIGH)
    else:
        GPIO.output(LED, GPIO.LOW)

# Kırmızı renk için frekans ölçümü yapacak fonksiyon
def measure_red_frequency():
    control_led(True)  # LED'i aç
    # Kırmızı renk filtresi ayarı
    GPIO.output(S2, GPIO.LOW)
    GPIO.output(S3, GPIO.LOW)
    GPIO.output(S1, GPIO.HIGH)
    GPIO.output(S0, GPIO.HIGH)
    
    time.sleep(0.05)  # Sensörün ayarlanmasını bekleyin (daha kısa süre)
    # Frekansı okuyun
    start = time.time()
    for i in range(10):  # Daha az ölçüm yapın (örneğin 10 ölçüm)
        GPIO.wait_for_edge(OUT, GPIO.FALLING)
    end = time.time()
    frequency = 10 / (end - start)  # Frekansı hesaplayın
    control_led(False)  # LED'i kapat
    
    return frequency

# Kırmızı renk eşik değeri
RED_THRESHOLD = 2000  # Eşik değerinizi belirleyin

# Programı düzgün şekilde sonlandırmak için sinyal işleyici
def signal_handler(sig, frame):
    print("Program sonlandırıldı.")
    GPIO.cleanup()
    sys.exit(0)

# Sinyal işleyiciyi ayarla
signal.signal(signal.SIGINT, signal_handler)

# Programı çalıştırın
try:
    while True:
        red_frequency = measure_red_frequency()
        if red_frequency > RED_THRESHOLD:
            print("Kırmızı renk algılandı.")
        else:
            print("Kırmızı renk algılanmadı.")
        time.sleep(0.1)  # Her ölçüm arası 0.1 saniye bekle
except Exception as e:
    print(f"Hata: {e}")
finally:
    GPIO.cleanup()  # GPIO pinlerini temizleyin
