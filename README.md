# DonkeyCar Robocar Raspberry Pi 5 (DonkeyCar-style)

Este proyecto implementa **control remoto**, **streaming de cámara** y **recolección de datos** para un Robocar basado en Raspberry Pi 5.
Además incluye un **modo autónomo** que usa un modelo **TFLite** para predecir `steering` desde la imagen.

> Nota: esto es “DonkeyCar-style” (behavioral cloning). Si queréis usar el framework DonkeyCar oficial, mirad su documentación de instalación para Bookworm.

---

## Hardware
- **Raspberry Pi 5** (4GB)
- **Camera V3**
- **Motor Driver L298N**
- **Chasis** (2 o 4 motores) con **tracción diferencial** (skid-steer)

## Conexiones L298N (GPIO BCM)
- **Motor Izquierdo**: IN1 (GPIO 17), IN2 (GPIO 27), ENA (GPIO 18 - PWM)
- **Motor Derecho**: IN3 (GPIO 22), IN4 (GPIO 23), ENB (GPIO 25 - PWM)

✅ IMPORTANTE:
- Alimenta **motores** con batería/fuente aparte.
- Une **GND batería** con **GND Raspberry** (masa común).

---

## Instalación y arranque (Pi OS Bookworm)
1. Instala dependencias del sistema (una vez):
   ```bash
   sudo apt update
   sudo apt install -y python3-venv python3-picamera2 python3-opencv
   ```

2. Arranca el proyecto:
   ```bash
   chmod +x start.sh
   ./start.sh
   ```

3. Abre en el móvil/PC:
   `http://<IP_DE_TU_PI>:5000`

> GPIO en Pi 5: en Bookworm suele ir mejor con `lgpio` como pin factory (el `start.sh` exporta `GPIOZERO_PIN_FACTORY=lgpio`).

---

## Características
- **Control por joystick** (web móvil) + soporte Gamepad (API del navegador).
- **Streaming de vídeo** (MJPEG) desde la cámara.
- **Recolección de datos**: botón REC guarda en `data/run_YYYYmmdd_HHMMSS/`.
- **Modo autónomo**: carga `models/model.tflite` y conduce con visión.

---

## Datos (estructura)
Cada grabación crea un “run”:
```
data/
  run_20260211_102233/
    images/
      img_....jpg
    log.jsonl
```

---

## IA (workflow recomendado)
1. **Recolecta**: REC ON, conduce varias vueltas, REC OFF.
2. **Entrena en PC**: mira `training/README.md`.
3. **Despliega**: copia `models/model.tflite` a `robocar/models/model.tflite`.
4. **Autónomo**: cambia a AUTO MODE.
