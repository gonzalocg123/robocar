import time
from gpiozero import DigitalOutputDevice, PWMOutputDevice
from gpiozero.pins.lgpio import LGPIOFactory

# Use the same factory as the main app
factory = LGPIOFactory()

# Pin definitions (from robocar.py)
LEFT_IN1 = 17
LEFT_IN2 = 27
LEFT_EN = 18

RIGHT_IN3 = 22
RIGHT_IN4 = 23
RIGHT_EN = 25

print("Initializing motors...")
try:
    l_in1 = DigitalOutputDevice(LEFT_IN1, pin_factory=factory)
    l_in2 = DigitalOutputDevice(LEFT_IN2, pin_factory=factory)
    l_en = PWMOutputDevice(LEFT_EN, pin_factory=factory)

    r_in3 = DigitalOutputDevice(RIGHT_IN3, pin_factory=factory)
    r_in4 = DigitalOutputDevice(RIGHT_IN4, pin_factory=factory)
    r_en = PWMOutputDevice(RIGHT_EN, pin_factory=factory)
except Exception as e:
    print(f"Error initializing: {e}")
    exit(1)

def stop_all():
    l_in1.off()
    l_in2.off()
    l_en.value = 0
    r_in3.off()
    r_in4.off()
    r_en.value = 0

stop_all()
time.sleep(1)

print("Testing LEFT Motor FORWARD (IN1=1, IN2=0)")
l_in1.on()
l_in2.off()
l_en.value = 0.5
time.sleep(2)
stop_all()
time.sleep(1)

print("Testing LEFT Motor BACKWARD (IN1=0, IN2=1)")
l_in1.off()
l_in2.on()
l_en.value = 0.5
time.sleep(2)
stop_all()
time.sleep(1)

print("Testing RIGHT Motor FORWARD (IN3=1, IN4=0)")
r_in3.on()
r_in4.off()
r_en.value = 0.5
time.sleep(2)
stop_all()
time.sleep(1)

print("Testing RIGHT Motor BACKWARD (IN3=0, IN4=1)")
r_in3.off()
r_in4.on()
r_en.value = 0.5
time.sleep(2)
stop_all()

print("Test complete.")
