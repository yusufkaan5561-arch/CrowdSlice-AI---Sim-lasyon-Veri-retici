# CrowdSlice AI — Simülasyon Veri Üretici

Açık hava etkinliği (konser / stadyum) için **5G şebeke verisi simülasyonu**. Üç senaryo üretir:

1. **Normal etkinlik akışı**
2. **Yoğunluk artışı** (ara vermeler, kapı açılışı)
3. **Panik / acil durum**

## Çıktılar

- **CSV:** `outputs/crowdslice_simulation_data.csv` — zaman serisi metrikleri ve etiketler
- **Görsel:** `outputs/crowdslice_visualization.png` — 8 panel dashboard

## Gereksinimler

- Python 3.8+
- Bağımlılıklar: `requirements.txt`

## Kurulum ve Çalıştırma

```bash
cd "CrowdSlice AI - Simülasyon Veri Üretici"
pip install -r requirements.txt
python crowdslice_simulation.py
```

Çıktılar proje içindeki `outputs/` klasörüne yazılır.

## Parametreler (script içinde)

| Parametre | Varsayılan | Açıklama |
|-----------|------------|----------|
| `SAMPLE_INTERVAL_SEC` | 10 | Örnekleme aralığı (saniye) |
| `EVENT_DURATION_MIN` | 180 | Etkinlik süresi (dakika) |
| `N_SECTORS` | 4 | Baz istasyonu sektör sayısı (A–D) |
| `MAX_DEVICES` | 25000 | Maksimum cihaz sayısı |

Panik senaryosu varsayılan olarak **130. dakikada** tetiklenir.
