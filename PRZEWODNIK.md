# 🎬 Przewodnik: Przetwarzanie nowego wideo przez HybrIK → Blender

## Wymagania wstępne

-   ✅ Zainstalowane środowisko conda 'hybrik'
-   ✅ Naprawiony dodatek HybrIK w Blenderze (fix_blender_addon_v2.py został uruchomiony)
-   ✅ Blender 4.4+ z włączonym dodatkiem HybrIK

## Krok 1: Przygotowanie wideo

1. Umieść nowy plik wideo w folderze HybrIK:
    ```
    c:\Users\xsiad\Desktop\VIBE\HybrIK\twoje_video.mp4
    ```

## Krok 2: Aktywacja środowiska i przetwarzanie

1. Otwórz PowerShell/Command Prompt
2. Przejdź do folderu HybrIK:

    ```powershell
    cd "c:\Users\xsiad\Desktop\VIBE\HybrIK"
    ```

3. Aktywuj środowisko conda:

    ```powershell
    conda activate hybrik
    ```

4. Uruchom przetwarzanie (opcja A - automatyczny skrypt):

    ```powershell
    python process_new_video.py twoje_video.mp4
    ```

    LUB opcja B - ręczne uruchomienie:

    ```powershell
    python scripts/demo_video_simple.py --video twoje_video.mp4 --out_dir results_twoje_video
    ```

## Krok 3: Sprawdzenie wyników

Po zakończeniu sprawdź folder wyników:

```
results_twoje_video/
├── res.pk                    # ← Dane 3D do importu w Blenderze
├── res_2d_twoje_video.mp4   # ← Wideo z nałożonymi kośćmi 2D
└── frames/                   # ← Pojedyncze klatki
```

## Krok 4: Import do Blendera

1. **Otwórz Blender 4.4+**

2. **Sprawdź dodatek HybrIK:**

    - Edit → Preferences → Add-ons
    - Wyszukaj "HybrIK"
    - Upewnij się, że jest włączony ✅

3. **Import animacji:**

    - File → Import → HybrIK (.pk)
    - Wybierz plik: `results_twoje_video/res.pk`
    - Kliknij "Import HybrIK"

4. **Sprawdź wyniki:**
    - W scenie powinien pojawić się model 3D z animacją
    - Naciśnij SPACJĘ, aby odtworzyć animację
    - Użyj kółka myszy do przybliżenia/oddalenia

## Krok 5: Dostosowanie w Blenderze

1. **Zmiana materiałów:**

    - Wybierz model w outlinerze
    - Przejdź do Shading workspace
    - Edytuj materiały według potrzeb

2. **Dodanie oświetlenia:**

    - Add → Light → Sun (oświetlenie słoneczne)
    - Add → Light → Area (oświetlenie powierzchniowe)

3. **Konfiguracja kamery:**

    - Wybierz kamerę
    - Ustaw kąt widzenia (Numpad 0)
    - Dostosuj pozycję

4. **Renderowanie:**
    - F12 - render pojedynczej klatki
    - Render → Render Animation - render całej animacji

## Rozwiązywanie problemów

### Problem: "Błąd importu - brak obiektu Camera"

**Rozwiązanie:** Uruchom fix_blender_addon_v2.py ponownie

### Problem: "use_auto_smooth attribute error"

**Rozwiązanie:** Dodatek został naprawiony, ale sprawdź czy fix został zastosowany

### Problem: "Zbyt długie przetwarzanie"

**Rozwiązanie:**

-   Skróć wideo (używaj filmów 10-30 sekund do testów)
-   Zmniejsz rozdzielczość wideo

### Problem: "Błąd CUDA/GPU"

**Rozwiązanie:** HybrIK będzie działać na CPU (wolniej, ale skutecznie)

## Przykładowe czasy przetwarzania

-   Wideo 10 sekund (30 FPS) = ~300 klatek = ~5-15 minut na CPU
-   Wideo 30 sekund = ~15-45 minut na CPU
-   Z GPU: znacznie szybciej (jeśli CUDA działa)

## Struktura plików po przetwarzaniu

```
HybrIK/
├── twoje_video.mp4
├── results_twoje_video/
│   ├── res.pk              # Do importu w Blenderze
│   ├── res_2d_twoje_video.mp4
│   └── frames/
├── process_new_video.py
└── fix_blender_addon_v2.py
```

## Wskazówki optymalizacyjne

1. **Wideo o dobrej jakości** - lepsze wyniki śledzenia
2. **Dobra widoczność postaci** - unikaj zasłaniania
3. **Stabilne oświetlenie** - unikaj migotania
4. **Rozdzielczość 720p-1080p** - wystarczająca do dobrych wyników
