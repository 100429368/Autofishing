import math
import time
import tkinter as tk
from typing import Optional, Tuple

import cv2
import numpy as np
import pyaudiowpatch as pyaudio
import pyautogui

# Desactivar la pausa de seguridad de PyAutoGUI si se requiere rapidez
pyautogui.FAILSAFE = True

# Umbral de sensibilidad para la captura de audio (0.0 a 1.0)
UMBRAL_AUDIO = 0.02


class RegionSelector:
    """Crea una ventana transparente a pantalla completa para seleccionar un área mediante el ratón."""

    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.attributes("-fullscreen", True)
        self.root.attributes("-alpha", 0.3)  # Ventana semi-transparente
        self.root.config(cursor="cross")

        self.canvas = tk.Canvas(self.root, cursor="cross", bg="grey")
        self.canvas.pack(fill="both", expand=True)

        self.canvas.bind("<ButtonPress-1>", self.on_button_press)
        self.canvas.bind("<B1-Motion>", self.on_move_press)
        self.canvas.bind("<ButtonRelease-1>", self.on_button_release)

        self.start_x: Optional[int] = None
        self.start_y: Optional[int] = None
        self.rect: Optional[int] = None
        self.region: Optional[Tuple[int, int, int, int]] = None

    def on_button_press(self, event: tk.Event) -> None:
        self.start_x = event.x
        self.start_y = event.y
        self.rect = self.canvas.create_rectangle(
            event.x, event.y, event.x, event.y, outline="red", width=2
        )

    def on_move_press(self, event: tk.Event) -> None:
        if self.start_x is None or self.start_y is None or self.rect is None:
            return
        cur_x, cur_y = event.x, event.y
        self.canvas.coords(self.rect, self.start_x, self.start_y, cur_x, cur_y)

    def on_button_release(self, event: tk.Event) -> None:
        if self.start_x is None or self.start_y is None:
            return
        end_x, end_y = event.x, event.y
        x1 = min(self.start_x, end_x)
        y1 = min(self.start_y, end_y)
        width = abs(end_x - self.start_x)
        height = abs(end_y - self.start_y)

        self.region = (x1, y1, width, height)
        self.root.destroy()

    def get_region(self) -> Optional[Tuple[int, int, int, int]]:
        self.root.mainloop()
        return self.region

def detect_target_color_instant(
    region: Tuple[int, int, int, int]
) -> Optional[Tuple[int, int]]:
    """Captura de pantalla instantánea para buscar la pluma/boya roja de pesca en la región seleccionada."""
    left, top, width, height = region

    screenshot = pyautogui.screenshot(region=region)
    frame = np.array(screenshot)

    # Convertir a espacio de color HSV (RGB -> HSV)
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)

    # ---------------------------------------------------------
    # RANGO DE COLOR PARA LA BOYA / PLUMA ROJA (WotLK / WoW)
    # En OpenCV HSV:
    # H va de 0 a 179 (el color rojo/ámbar abarca: 0-20 y 160-180)
    # S va de 75 a 255 (saturación para aislar la pluma roja del agua y terreno)
    # V va de 20 a 255 (permite detectar la pluma tanto en sombra/noche como al sol)
    # ---------------------------------------------------------
    lower_red1 = np.array([0, 75, 20], dtype=np.uint8)
    upper_red1 = np.array([20, 255, 255], dtype=np.uint8)

    lower_red2 = np.array([160, 75, 20], dtype=np.uint8)
    upper_red2 = np.array([180, 255, 255], dtype=np.uint8)

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)

    # Filtro morfológico para eliminar pequeñas motas o ruido puntual y rellenar la forma
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)

    # Buscar contornos de las áreas detectadas
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # Filtrar por área para ignorar ruido pero detectar boyas a cualquier distancia
    valid_contours = [c for c in contours if cv2.contourArea(c) >= 5]

    if valid_contours:
        # Tomar el contorno más grande detectado en la región
        largest_contour = max(valid_contours, key=cv2.contourArea)
        M = cv2.moments(largest_contour)
        if M["m00"] != 0:
            rel_x = int(M["m10"] / M["m00"])
            rel_y = int(M["m01"] / M["m00"])
        else:
            x, y, w, h = cv2.boundingRect(largest_contour)
            rel_x = x + w // 2
            rel_y = y + h // 2

        abs_x = left + rel_x
        abs_y = top + rel_y

        return (abs_x, abs_y)

    return None

def listen_for_sound_threshold(
    audio_stream: pyaudio.Stream,
    threshold: float = UMBRAL_AUDIO,
    timeout: float = 20.0,
) -> bool:
    """Escucha el audio durante un tiempo límite. Devuelve True si supera el umbral."""
    start_time = time.time()
    print(f"Escuchando audio durante {timeout} segundos...")

    while time.time() - start_time < timeout:
        data = audio_stream.read(1024, exception_on_overflow=False)
        samples = np.frombuffer(data, dtype=np.float32)
        rms = np.sqrt(np.mean(samples**2))

        if rms > threshold:
            print(f"¡Sonido detectado! Nivel: {rms:.4f}")
            return True

        time.sleep(0.01)

    print("Tiempo de escucha finalizado sin superar el umbral.")
    return False


def main() -> None:
    # 1. Esperar 2 segundos
    print("[Paso 1] Iniciando script en 2 segundos...")
    time.sleep(2)

    # 2. Pedir al usuario capturar la zona de la pantalla deseada
    print("[Paso 2] Selecciona la zona de la pantalla con el ratón...")
    selector = RegionSelector()
    region = selector.get_region()

    if not region or region[2] == 0 or region[3] == 0:
        print("Selección no válida o cancelada.")
        return

    print(f"Zona seleccionada correctamente: {region}")

    # Configuración de captura de audio Loopback
    p = pyaudio.PyAudio()
    stream = None

    try:
        wasapi_info = p.get_host_api_info_by_type(pyaudio.paWASAPI)
        default_speakers = p.get_device_info_by_index(
            wasapi_info["defaultOutputDevice"]
        )

        loopback = next(
            dev
            for dev in p.get_device_info_generator()
            if dev["isLoopbackDevice"]
            and default_speakers["name"] in dev["name"]
        )

        print(f"Capturando sonido de: {loopback['name']}")

        stream = p.open(
            format=pyaudio.paFloat32,
            channels=loopback["maxInputChannels"],
            rate=int(loopback["defaultSampleRate"]),
            input=True,
            input_device_index=loopback["index"],
        )

        print("\n--- Iniciando bucle principal (Presiona Ctrl+C para detener) ---")

        while True:
            # 3. Esperar 5 segundos
            print("\n[Paso 3] Esperando 5 segundos...")
            time.sleep(5.0)

            # 4. Pulsar "4"
            print("[Paso 4] Pulsando la tecla '4'...")
            pyautogui.press("4")

            # 5. Esperar 1 segundo
            print("[Paso 5] Esperando 1 segundo...")
            time.sleep(1.0)

            # 6. Buscar la zona del objeto
            print("[Paso 6] Buscando zona objetivo (boya de pesca)...")
            target_pos = detect_target_color_instant(region)

            time.sleep(1.0)

            # 7. Mover el ratón a la zona encontrada
            if target_pos:
                print(f"[Paso 7] Moviendo ratón a la zona objetivo: {target_pos}")
                pyautogui.moveTo(target_pos[0], target_pos[1], duration=0.05)
            else:
                print("[Paso 7] No se encontró la zona objetivo en la captura.")

            # 8. Si supera el umbral de audición, click derecho
            print("[Paso 8] Escuchando audio...")
            sound_detected = listen_for_sound_threshold(
                stream, threshold=UMBRAL_AUDIO, timeout=20.0
            )

            if sound_detected:
                print("¡Umbral superado! Haciendo Clic Derecho...")
                if target_pos:
                    pyautogui.moveTo(target_pos[0], target_pos[1], duration=0)
                pyautogui.rightClick()

            # 9. Esperar 0.5 segundos
            print("[Paso 9] Esperando 0.5 segundos...")
            time.sleep(0.5)

            # 10. Click izquierdo
            print("[Paso 10] Haciendo Clic Izquierdo...")
            pyautogui.click()

    except KeyboardInterrupt:
        print("\nEjecución detenida por el usuario.")
    except StopIteration:
        print("\nNo se encontró un dispositivo Loopback compatible.")
    except Exception as e:
        print(f"\nOcurrió un error inesperado: {e}")
    finally:
        if stream is not None:
            stream.stop_stream()
            stream.close()
        p.terminate()


if __name__ == "__main__":
    main()
    