# Robocar RP5

Aplicacion de control remoto y conduccion autonoma para un robocar basado en Raspberry Pi 5.
El proyecto implementa una arquitectura tipo "behavioral cloning" (estilo DonkeyCar): primero se recolectan datos conduciendo en modo manual y despues se entrena un modelo que predice direccion (`steering`) desde imagenes.

## Tabla de contenidos
1. [Resumen](#resumen)
2. [Caracteristicas](#caracteristicas)
3. [Arquitectura](#arquitectura)
4. [Requisitos](#requisitos)
5. [Instalacion en Raspberry Pi 5](#instalacion-en-raspberry-pi-5)
6. [Uso diario](#uso-diario)
7. [Cableado L298N (GPIO BCM)](#cableado-l298n-gpio-bcm)
8. [Configuracion por variables de entorno](#configuracion-por-variables-de-entorno)
9. [API HTTP](#api-http)
10. [Dataset y entrenamiento](#dataset-y-entrenamiento)
11. [Estructura del proyecto](#estructura-del-proyecto)
12. [Seguridad operativa](#seguridad-operativa)
13. [Troubleshooting](#troubleshooting)
14. [Estado del proyecto](#estado-del-proyecto)

## Resumen
- Backend en Flask para control, streaming y grabacion de sesiones.
- UI web movil con joystick virtual (nipplejs) para teleoperacion.
- Grabacion de dataset en `data/run_YYYYmmdd_HHMMSS/`.
- Inferencia en tiempo real con modelo TFLite (`models/model.tflite`).
- Watchdog de seguridad para detener motores si se pierde control.

## Caracteristicas
- Control en tiempo real por navegador (telefono o PC).
- Streaming MJPEG en `/video_feed`.
- Modo `manual` y `autonomous`.
- Grabacion de muestras de entrenamiento con `steering` y `throttle`.
- Parametrizacion completa por variables de entorno (pines, polaridad, resolucion, seguridad).
- Fallback de hardware:
  - Camara por Picamera2 (preferido en Pi) o OpenCV.
  - TFLite runtime o interpreter de TensorFlow.

## Arquitectura
```text
Browser (UI joystick)
   -> POST /control, /set_mode, /toggle_recording
   <- GET  /status, /video_feed

Flask app (robocar.py)
   -> MotorController (gpiozero + L298N)
   -> Camera (Picamera2/OpenCV)
   -> DataRecorder (imagenes + log.jsonl)
   -> Autopilot (TFLite, steering prediction)
```

Flujo operativo:
1. Usuario conduce en `manual`.
2. Se activa `REC` para generar dataset.
3. Se entrena el modelo en PC (`training/train_steering.py`).
4. Se copia `model.tflite` a `models/`.
5. Se cambia a `autonomous`.

## Requisitos

Hardware recomendado:
- Raspberry Pi 5.
- Camara compatible con Picamera2 (ejemplo: Camera Module V3).
- Driver de motores L298N.
- Chasis diferencial (2WD o 4WD).
- Fuente separada para motores.

Software:
- Raspberry Pi OS Bookworm.
- Python 3.
- Paquetes del sistema para camara/OpenCV:
  - `python3-venv`
  - `python3-picamera2`
  - `python3-opencv`

Dependencias Python del proyecto (`requirements.txt`):
- `flask`
- `flask-cors`
- `gpiozero`
- `pillow`

Para entrenamiento en PC:
- `tensorflow`
- `opencv-python`
- `numpy`

## Instalacion en Raspberry Pi 5

1. Instalar dependencias del sistema (una sola vez):
```bash
sudo apt update
sudo apt install -y python3-venv python3-picamera2 python3-opencv
```

2. Entrar al directorio del proyecto y arrancar:
```bash
chmod +x start.sh
./start.sh
```

`start.sh` realiza:
- Export de `GPIOZERO_PIN_FACTORY=lgpio` (si no existe).
- Creacion de `venv` con `--system-site-packages`.
- Upgrade de `pip`.
- Instalacion de dependencias Python.
- Inicio del servidor Flask en `0.0.0.0:5000`.

3. Abrir la interfaz:
```text
http://<IP_DE_LA_PI>:5000
```

## Uso diario

Control manual:
1. Abrir la UI web.
2. Conducir con joystick virtual.
3. Verificar estado `ONLINE` en la cabecera.

Grabacion de datos:
1. Pulsar `REC`.
2. Conducir varias vueltas representativas del circuito.
3. Pulsar `STOP REC`.
4. Validar carpeta nueva en `data/run_YYYYmmdd_HHMMSS/`.

Conduccion autonoma:
1. Copiar `model.tflite` a `models/model.tflite`.
2. Pulsar `AUTO MODE`.
3. Supervisar comportamiento y tener parada manual preparada.

## Cableado L298N (GPIO BCM)

Asignacion por defecto en backend:
- Motor izquierdo:
  - `LEFT_MOTOR_IN1=17`
  - `LEFT_MOTOR_IN2=27`
  - `LEFT_MOTOR_ENA=18` (PWM)
- Motor derecho:
  - `RIGHT_MOTOR_IN3=22`
  - `RIGHT_MOTOR_IN4=23`
  - `RIGHT_MOTOR_ENB=25` (PWM)

Importante:
- Motores con alimentacion separada de la Raspberry Pi.
- Compartir GND entre fuente de motores y Raspberry Pi.
- Probar primero con ruedas elevadas antes de apoyar el vehiculo en suelo.

## Configuracion por variables de entorno

El backend lee estas variables al iniciar:

| Variable | Default | Descripcion |
| --- | --- | --- |
| `LEFT_MOTOR_IN1` | `17` | Pin direccion motor izquierdo A |
| `LEFT_MOTOR_IN2` | `27` | Pin direccion motor izquierdo B |
| `LEFT_MOTOR_ENA` | `18` | Pin PWM motor izquierdo |
| `RIGHT_MOTOR_IN3` | `22` | Pin direccion motor derecho A |
| `RIGHT_MOTOR_IN4` | `23` | Pin direccion motor derecho B |
| `RIGHT_MOTOR_ENB` | `25` | Pin PWM motor derecho |
| `STREAM_W` | `640` | Ancho del streaming MJPEG |
| `STREAM_H` | `480` | Alto del streaming MJPEG |
| `RECORD_W` | `160` | Ancho de imagen para dataset e inferencia |
| `RECORD_H` | `120` | Alto de imagen para dataset e inferencia |
| `RECORD_HZ` | `10.0` | Frecuencia maxima de guardado de muestras |
| `CONTROL_TIMEOUT_S` | `0.60` | Tiempo maximo sin control antes de stop |
| `THROTTLE_DEADZONE` | `0.05` | Zona muerta de aceleracion |
| `STEERING_DEADZONE` | `0.01` | Zona muerta de direccion |
| `THROTTLE_POLARITY` | `-1.0` | Invierte aceleracion si es necesario |
| `STEERING_POLARITY` | `1.0` | Invierte direccion si es necesario |
| `MODEL_PATH` | `models/model.tflite` | Ruta del modelo TFLite |
| `AUTO_THROTTLE` | `0.40` | Aceleracion fija en modo autonomo |
| `AUTO_STEER_SMOOTH` | `0.30` | Suavizado EMA del steering autonomo |
| `DATA_DIR` | `data` | Directorio raiz de runs |
| `SPEED_LIMIT` | `1.0` | Limite global de potencia de motores |

Ejemplo:
```bash
THROTTLE_POLARITY=1.0 SPEED_LIMIT=0.8 AUTO_THROTTLE=0.35 ./start.sh
```

## API HTTP

### `GET /`
Devuelve la interfaz web.

### `GET /video_feed`
Devuelve streaming MJPEG en tiempo real.

### `GET /status`
Devuelve estado actual:
- `steering`
- `throttle`
- `mode`
- `recording`
- `speed_limit`
- `run_id`
- `model_path`

### `POST /control`
Actualiza control manual.

Payload:
```json
{
  "steering": -0.25,
  "throttle": 0.40
}
```

### `POST /set_mode`
Cambia modo de conduccion.

Payload:
```json
{
  "mode": "manual"
}
```

`mode` admite `manual` o `autonomous`.

### `POST /toggle_recording`
Alterna entre grabar y detener grabacion.

## Dataset y entrenamiento

Formato esperado de un run:
```text
data/
  run_YYYYmmdd_HHMMSS/
    images/
      img_YYYYmmdd_HHMMSS_microsec.jpg
    log.jsonl
```

Cada linea en `log.jsonl` contiene:
- `timestamp`
- `image` (ruta relativa)
- `steering` (float)
- `throttle` (float)

Entrenamiento (PC):
```bash
cd training
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install tensorflow opencv-python numpy
python train_steering.py --data_root ../data --epochs 15
```

Opciones utiles:
- `--run_dir <ruta_run>` para entrenar un run especifico.
- `--min_abs_throttle 0.05` para filtrar frames sin movimiento.
- `--img_w` y `--img_h` para ajustar resolucion de entrenamiento.

Salidas:
- `models/model.h5`
- `models/model.tflite`

Despliegue:
1. Copiar `model.tflite` al Raspberry Pi.
2. Ubicarlo en `models/model.tflite` (o ajustar `MODEL_PATH`).
3. Cambiar a modo `autonomous` desde la UI.

## Estructura del proyecto

```text
.
├── robocar.py               # Backend principal (Flask + motores + camara + IA)
├── start.sh                 # Script de arranque para Raspberry Pi
├── test_motors.py           # Prueba basica de giro por motor
├── requirements.txt         # Dependencias Python de ejecucion
├── templates/
│   └── index.html           # UI web de control
├── training/
│   ├── train_steering.py    # Entrenamiento y export a TFLite
│   └── README.md            # Guia de entrenamiento
├── models/                  # Modelos entrenados (output/deploy)
└── data/                    # Runs grabados (ignorado en git)
```

## Seguridad operativa

Antes de pruebas:
1. Elevar ruedas y verificar sentido de giro.
2. Confirmar `THROTTLE_POLARITY` y `STEERING_POLARITY`.
3. Validar que el watchdog detiene motores al perder control.
4. Probar `test_motors.py` en pulsos cortos antes de conducir.

Durante pruebas:
1. Mantener espacio despejado.
2. Evitar superficies con baja adherencia en primeras iteraciones.
3. Supervisar siempre modo autonomo con capacidad de corte inmediato.

## Troubleshooting

`No video` o stream vacio:
- Verificar camara y cable ribbon.
- Confirmar instalacion de `python3-picamera2` y `python3-opencv`.
- Revisar logs: el backend indica si usa Picamera2 o fallback OpenCV.

Motores giran al reves:
- Ajustar `THROTTLE_POLARITY` o `STEERING_POLARITY`.
- Revisar cableado IN1/IN2 e IN3/IN4.

Modo autonomo no responde:
- Verificar `models/model.tflite`.
- Revisar que exista interprete TFLite (`tflite_runtime` o TensorFlow).
- Confirmar que el modelo produzca salida en rango `[-1, 1]`.

Paradas frecuentes en manual:
- El watchdog corta si no recibe `/control` dentro de `CONTROL_TIMEOUT_S`.
- Revisar conectividad WiFi y latencia del dispositivo de control.

## Estado del proyecto

Proyecto funcional para prototipado y experimentacion educativa con robocar.
No incluye licencia explicita en este repositorio; agrega una licencia antes de uso comercial o distribucion externa.
