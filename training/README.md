# Training (PC/Laptop)

> Consejo: Entrena en tu PC (mejor con GPU). En la Raspberry Pi 5 puedes hacer inferencia con `tflite-runtime`.

## 1) Recolecta datos en la Pi
1. Arranca el servidor en la Pi: `./start.sh`
2. Abre `http://IP_DE_LA_PI:5000`
3. Pulsa **REC** y conduce un circuito (varias vueltas).
4. Pulsa **STOP REC**.
5. En la Pi tendrás una carpeta: `data/run_YYYYmmdd_HHMMSS/` con `images/` y `log.jsonl`.

## 2) Copia el dataset al PC
Copia la carpeta `data/` completa o al menos el `run_...` que quieras.

## 3) Entrena
En tu PC:
```bash
cd training
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install tensorflow opencv-python numpy
python train_steering.py --data_root ../data --epochs 15
```

## 4) Despliega el modelo en la Pi
El script crea:
- `models/model.h5`
- `models/model.tflite`

Copia `models/model.tflite` a la Pi, dentro del proyecto:
`robocar/models/model.tflite`

Luego, en la UI, cambia a **AUTO MODE**.

## Notas prácticas
- Si el coche oscila, baja `AUTO_THROTTLE` o sube el suavizado `AUTO_STEER_SMOOTH`.
- Si graba demasiadas fotos, baja `RECORD_HZ` (por ejemplo 8 o 5).
