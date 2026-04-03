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

## Google Colab (bez Dockera)

- W repozytorium: notebook `colab/WakeWordProject_Colab.ipynb` — wgraj go do Colab (**File → Upload notebook**) albo otwórz z GitHuba.
- Ustaw `REPO_URL` w notebooku (fork z tym projektem) albo wgraj ZIP do `/content/WakeWordProject` i `USE_GIT_CLONE = False`.
- Skrypt `colab/colab_train.py` klonuje `openWakeWord` v0.6.0, stosuje ten sam patch co Dockerfile i uruchamia konwersję TFLite lokalnie (`onnx2tf` z pip).
- Profil `training_configs/hey_lolita_colab.yml` jest lżejszy (mniej próbek/kroków) pod limity czasu i RAM w Colab; pełna jakość: skopiuj parametry z `hey_lolita.yml`.
- Jeśli `colab_train.py` kończy się **exit 2**: zaktualizuj plik `colab/colab_train.py` z repozytorium (poprzednio `yaml.dump` potrafił psuć config dla `openwakeword.train`; jest też wymuszany `PYTHONPATH` na klon w `/content/openwakeword_v060`). Po zmianach w Colab usuń stary klon: `!rm -rf /content/openwakeword_v060` i uruchom ponownie.

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
  Większa pamięć współdzielona Dockera (domyślnie `4g` — bezpieczniej na laptopach z 16 GB RAM).

- `python train.py --reset_workspace`  
  Usuwa wszystkie `outputs*` oraz `assets/` (w tym pobrane cechy walidacyjne) i kończy — czysty start przed kolejnym `python train.py`.

- `python train.py --force_overwrite`  
  Wymusza nadpisanie plików pośrednich (pełny świeży przebieg).

- `python train.py --skip_tflite_conversion`  
  Kończy na modelu ONNX.

## Konfiguracja treningu

Plik: `training_configs/hey_lolita.yml`

Profil **najlepszej jakości** (nadal sensowny na ~16 GB RAM): więcej próbek, 2 rundy augmentacji, dłuższy trening.

- `n_samples: 22000`, `n_samples_val: 3500`
- `tts_batch_size: 18`, `augmentation_batch_size: 6`, `augmentation_rounds: 2`
- `batch_n_per_class: 14/14`, `steps: 64000` (z gotowymi `*_features_*.npy` możesz podnieść `steps` i użyć `--train_only`)

W **Docker Desktop** ustaw limit pamięci kontenerów ok. **10–12 GB** (zostaw margines dla Windows). **RTX 4050**: obecny `Dockerfile` trenuje na **CPU** w kontenerze Linux; GPU wymagałoby osobnego obrazu z CUDA (nie jest w tym projekcie).

`.dockerignore` zawęża kontekst buildu (szybszy `docker build`, mniejszy transfer do demona).

## Rozwiązywanie problemów

- **Exit code 137**  
  Zwykle brak pamięci. Zwiększ RAM w Docker Desktop, zmniejsz batche w YAML lub uruchom z `--shm_size 8g` (jeśli masz zapas RAM poza Dockerem).

- **Bus error / shared memory**  
  Uruchamiaj z `--shm_size 8g` (lub więcej).

- **`ONNX_GEMM` / nierozwiązany custom op w TFLite (Wyoming)**  
  Stary pipeline `onnx2tf` potrafił wstawić operator `ONNX_GEMM`, którego wbudowany interpreter TFLite w dodatku Wyoming nie obsługuje.  
  `train.py` przepuszcza ONNX przez `scripts/rewrite_last_gemm_to_matmul.py` (ostatnia warstwa `Gemm` → `MatMul` + `Add`).

- **Brak reakcji na słowo mimo niskiego progu (np. 0,1)**  
  `pyopen_wakeword` (Wyoming) podaje do modelu bufor o kształcie **`[1, 16, 96]`** (16 okien × 96 wymiarów embeddingu).  
  Surowy eksport `onnx2tf` często deklaruje wejście jako **`[1, 96, 16]`** — ten sam blok pamięci jest wtedy **źle interpretowany**, wyniki są bezsensowne i próg nigdy nie przechodzi.  
  Bieżący `train.py` po `onnx2tf` (ścieżka `tf_converter`) owija SavedModel skryptem `scripts/wrap_saved_model_wake_input.py`, tak aby plik `.tflite` miał wejście **`[1, 16, 96]`** i zgadzał się numerycznie z ONNX.

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
