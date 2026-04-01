# WakeWordProject

Projekt automatyzuje trening własnego wake word dla `openWakeWord` w Dockerze
oraz przygotowuje gotowy model `.tflite` do użycia w Home Assistant.

Docelowa fraza treningowa: **"hey lolita"**

## Co robi ten projekt

- buduje kontener treningowy z kompletem zależności,
- uruchamia pipeline `openWakeWord` (generacja próbek, augmentacja, trening),
- zapisuje model `ONNX` do lokalnego katalogu output,
- konwertuje model do `TFLite` przez dedykowany kontener `onnx2tf`,
- pozwala wznowić pracę po przerwaniu, bez tracenia całego postępu.

## Struktura projektu

- `train.py` - główny runner uruchamiany z hosta (jedna komenda).
- `training_configs/hey_lolita.yml` - konfiguracja treningu.
- `Dockerfile` - obraz z narzędziami do treningu i konwersji.
- `scripts/convert_to_tflite.py` - pomocniczy skrypt konwersji (używany w pipeline).
- `outputs_.../` - katalog wyników (tworzony automatycznie).

## Wymagania

- Windows + PowerShell
- Docker Desktop (uruchomiony, `Engine running`)
- Python 3.10+

## Szybki start (jedna komenda)

W katalogu projektu uruchom:

```bash
python train.py
```

Komenda wykona cały proces end-to-end:

1. Pobierze wymagane dane walidacyjne (przy pierwszym uruchomieniu).
2. Zbuduje obraz Docker.
3. Uruchomi trening `openWakeWord`.
4. Wykona konwersję `ONNX -> TFLite`.
5. Zapisze artefakty do katalogu output.

## Gdzie są wyniki

Domyślnie w:

- `outputs_hey_lolita/hey_lolita.onnx`
- `outputs_hey_lolita/hey_lolita.tflite`

## Najważniejsze opcje uruchomienia

- `python train.py --build_only`  
  Tylko buduje obraz, bez treningu.

- `python train.py --train_only`  
  Pomija generowanie/augmentację i uruchamia tylko etap treningu.

- `python train.py --output_dir outputs_hey_lolita_v2`  
  Używa innego katalogu wynikowego.

- `python train.py --shm_size 8g`  
  Ustawia pamięć współdzieloną Dockera (ważne przy błędach typu bus error / 137).

- `python train.py --force_overwrite`  
  Wymusza nadpisanie plików pośrednich (pełny świeży przebieg).

- `python train.py --skip_tflite_conversion`  
  Kończy na modelu ONNX.

## Konfiguracja treningu

Plik: `training_configs/hey_lolita.yml`

Aktualny profil jest ustawiony na stabilność (mniejsza szansa przerwania):

- `n_samples: 12000`
- `n_samples_val: 2000`
- `augmentation_rounds: 1`
- `steps: 16000`
- `batch_n_per_class: 24/24`

Jeśli masz dużo RAM i chcesz lepszą jakość, możesz stopniowo zwiększać `steps`.

## Rozwiązywanie problemów

- **Exit code 137**  
  Zwykle brak pamięci. Zwiększ RAM w Docker Desktop i uruchom z `--shm_size 8g`.

- **Bus error / shared memory**  
  Uruchamiaj z `--shm_size 8g` (lub więcej).

- **Błędy połączenia w Home Assistant (Wyoming)**  
  Sprawdź, czy addon/serwis wake word działa i czy host/port są poprawne.

## Integracja z Home Assistant

1. Wytrenuj model (`python train.py`).
2. Weź plik `hey_lolita.tflite` z katalogu output.
3. Dodaj model jako custom wake word w `wyoming-openwakeword`.
4. Wybierz ten model w ustawieniach Voice Assistant.
5. Zrestartuj usługę wake word/addon.
6. Przetestuj na docelowym mikrofonie i dostrój próg detekcji.

## Uwagi

- `openWakeWord` obecnie najlepiej współpracuje z formatem `TFLite` w HA/Wyoming.
- Pierwsze uruchomienie trwa najdłużej (build obrazu + pobrania).
