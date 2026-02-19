"""
CrowdSlice AI - Simülasyon Veri Üretici
========================================
Açık hava etkinliği (konser/stadyum) için 5G şebeke verisi simülasyonu.
3 farklı senaryo üretir:
  1. Normal etkinlik akışı
  2. Yoğunluk artışı (ara vermeler, kapı açılışı)
  3. Panik / acil durum
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from datetime import datetime, timedelta
import json
import os

# Çıktı klasörü (script ile aynı dizinde)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'outputs')

np.random.seed(42)

# ─────────────────────────────────────────────
# PARAMETRELER
# ─────────────────────────────────────────────
SAMPLE_INTERVAL_SEC = 10       # Her 10 saniyede bir veri noktası
EVENT_DURATION_MIN  = 180      # 3 saatlik etkinlik
N_SECTORS           = 4        # Baz istasyonu sektör sayısı (A, B, C, D)
MAX_DEVICES         = 25000    # Alanda maksimum cihaz

# ─────────────────────────────────────────────
# ZAMAN EKSENİ
# ─────────────────────────────────────────────
total_samples = (EVENT_DURATION_MIN * 60) // SAMPLE_INTERVAL_SEC
timestamps = [datetime(2025, 6, 15, 18, 0, 0) + timedelta(seconds=i * SAMPLE_INTERVAL_SEC)
              for i in range(total_samples)]
minutes = np.array([i * SAMPLE_INTERVAL_SEC / 60 for i in range(total_samples)])

# ─────────────────────────────────────────────
# YARDIMCI: Yumuşak geçiş (sigmoid)
# ─────────────────────────────────────────────
def sigmoid(x, center, steepness=0.3):
    return 1 / (1 + np.exp(-steepness * (x - center)))

def gaussian_bump(x, center, width, height):
    return height * np.exp(-0.5 * ((x - center) / width) ** 2)

# ─────────────────────────────────────────────
# 1. BAĞLI CİHAZ SAYISI (connected_devices)
# ─────────────────────────────────────────────
# - Etkinlik başlangıcında yavaş dolar
# - Arada konser molaları ile küçük dalgalanmalar
# - Panik anında anlık düşüş (herkes hareket eder, handover olur)
def generate_connected_devices(minutes):
    # Temel dolum eğrisi
    base = MAX_DEVICES * sigmoid(minutes, center=20, steepness=0.15)
    
    # Ara verme çıkışları (t=75, t=110 dk)
    break1 = -gaussian_bump(minutes, 75, 8, 1500)
    break2 = -gaussian_bump(minutes, 110, 8, 1200)
    
    # Kapanış başlamadan önce erken çıkışlar
    early_exit = -MAX_DEVICES * 0.2 * sigmoid(minutes, center=155, steepness=0.08)
    
    # Panik anı (t=130 dk) - ani el değiştirme → bağlantı sayısı dalgalanır
    panic_drop = -gaussian_bump(minutes, 130, 3, 3000)
    
    # Gürültü
    noise = np.random.normal(0, 300, len(minutes))
    
    devices = base + break1 + break2 + early_exit + panic_drop + noise
    return np.clip(devices, 0, MAX_DEVICES).astype(int)

connected_devices = generate_connected_devices(minutes)

# ─────────────────────────────────────────────
# 2. HANDOVER SAYISI / DAKİKA (handover_rate)
# Normal: 20-60 / dak
# Yoğun hareket: 100-200 / dak
# Panik: 400-800 / dak
# ─────────────────────────────────────────────
def generate_handover_rate(minutes, connected_devices):
    # Baseline: cihaz yoğunluğuyla orantılı
    base_rate = 30 + (connected_devices / MAX_DEVICES) * 50
    
    # Ara vermeler → insanlar hareket eder
    break1_movement = gaussian_bump(minutes, 75, 6, 120)
    break2_movement = gaussian_bump(minutes, 110, 6, 100)
    
    # Panik: ani ve çok yüksek handover fırtınası
    panic_storm = gaussian_bump(minutes, 130, 4, 650)
    
    # Kapanış çıkışı
    exit_movement = gaussian_bump(minutes, 160, 10, 180)
    
    noise = np.random.normal(0, 15, len(minutes))
    
    rate = base_rate + break1_movement + break2_movement + panic_storm + exit_movement + noise
    return np.clip(rate, 5, 1000).astype(int)

handover_rate = generate_handover_rate(minutes, connected_devices)

# ─────────────────────────────────────────────
# 3. VERİ TRAFİĞİ (Mbps) - data_traffic_mbps
# Normal: yüksek (sosyal medya, video)
# Panik: düşer (insanlar telefona bakmaz)
# ─────────────────────────────────────────────
def generate_data_traffic(minutes, connected_devices):
    # Temel trafik cihaz sayısıyla orantılı
    base_traffic = (connected_devices / MAX_DEVICES) * 8000  # max 8 Gbps
    
    # Konser başlangıcında "canlı paylaşım" spike'ı
    start_spike = gaussian_bump(minutes, 25, 5, 1500)
    
    # Ara vermelerde artış (paylaşım)
    break_spikes = (gaussian_bump(minutes, 75, 5, 800) +
                    gaussian_bump(minutes, 110, 5, 700))
    
    # Panik anında düşüş — insanlar telefonu kapatır / sesli aramaya geçer
    panic_drop = -gaussian_bump(minutes, 130, 5, 5000)
    
    noise = np.random.normal(0, 200, len(minutes))
    
    traffic = base_traffic + start_spike + break_spikes + panic_drop + noise
    return np.clip(traffic, 50, 12000).round(1)

data_traffic_mbps = generate_data_traffic(minutes, connected_devices)

# ─────────────────────────────────────────────
# 4. SESLİ ARAMA ORANI (voice_call_rate)
# Normal: düşük
# Panik: çok yüksek
# ─────────────────────────────────────────────
def generate_voice_calls(minutes):
    # Düşük taban
    base = 40 + np.random.normal(0, 8, len(minutes))
    
    # Panik: aile arama, yardım çağrısı
    panic_calls = gaussian_bump(minutes, 130, 6, 900)
    
    # Kapanış
    exit_calls = gaussian_bump(minutes, 162, 8, 200)
    
    return np.clip(base + panic_calls + exit_calls, 10, 1200).astype(int)

voice_call_rate = generate_voice_calls(minutes)

# ─────────────────────────────────────────────
# 5. ORTALAMA RSSI (sinyal gücü, dBm)
# Kalabalık arttıkça sinyal kalitesi düşer
# ─────────────────────────────────────────────
def generate_rssi(minutes, connected_devices):
    base_rssi = -65  # dBm, iyi sinyal
    # Yoğunluk arttıkça kötüleşir
    density_effect = -(connected_devices / MAX_DEVICES) * 20
    noise = np.random.normal(0, 2, len(minutes))
    # Panik anında interference artar
    panic_interference = -gaussian_bump(minutes, 130, 5, 8)
    return (base_rssi + density_effect + panic_interference + noise).round(1)

avg_rssi_dbm = generate_rssi(minutes, connected_devices)

# ─────────────────────────────────────────────
# 6. LATENCY (ms)
# ─────────────────────────────────────────────
def generate_latency(minutes, connected_devices):
    base_latency = 15  # ms, 5G normal
    load_effect = (connected_devices / MAX_DEVICES) * 35
    panic_spike = gaussian_bump(minutes, 130, 5, 80)
    noise = np.random.normal(0, 3, len(minutes))
    return np.clip(base_latency + load_effect + panic_spike + noise, 8, 200).round(1)

avg_latency_ms = generate_latency(minutes, connected_devices)

# ─────────────────────────────────────────────
# 7. SEKTÖR BAZLI CİHAZ DAĞILIMI (A,B,C,D)
# Panik anında sektör C'de (sahne önü) yoğunlaşma
# ─────────────────────────────────────────────
def generate_sector_distribution(minutes, total_devices):
    # Normal dağılım: A=%30, B=%25, C=%30, D=%15
    base = np.array([0.30, 0.25, 0.30, 0.15])
    
    sector_data = {}
    for i, (sector, ratio) in enumerate(zip(['A', 'B', 'C', 'D'], base)):
        sector_devices = (total_devices * ratio).astype(int)
        
        # Panik: herkes C sektörünü terk eder (kaçış yönü A ve B)
        if sector == 'C':
            panic_effect = -gaussian_bump(minutes, 132, 4, 0.15) * total_devices
        elif sector == 'A':
            panic_effect = gaussian_bump(minutes, 133, 4, 0.08) * total_devices
        elif sector == 'B':
            panic_effect = gaussian_bump(minutes, 133, 4, 0.06) * total_devices
        else:
            panic_effect = gaussian_bump(minutes, 132, 5, 0.03) * total_devices
        
        noise = np.random.normal(0, 100, len(minutes))
        devices = np.clip(sector_devices + panic_effect + noise, 100, MAX_DEVICES * 0.6).astype(int)
        sector_data[f'sector_{sector}_devices'] = devices
    
    return sector_data

sector_data = generate_sector_distribution(minutes, connected_devices)

# ─────────────────────────────────────────────
# 8. ETİKET: DURUM SINIFI
# 0=Normal, 1=Yoğun, 2=Panik, 3=Kriz
# ─────────────────────────────────────────────
def generate_labels(minutes, handover_rate, voice_call_rate, data_traffic_mbps):
    labels = np.zeros(len(minutes), dtype=int)
    label_names = []
    
    for i in range(len(minutes)):
        hr = handover_rate[i]
        vc = voice_call_rate[i]
        dt = data_traffic_mbps[i]
        
        if hr > 500 and vc > 600:
            labels[i] = 3  # Kriz
        elif hr > 300 or vc > 400:
            labels[i] = 2  # Panik
        elif hr > 120 or (connected_devices[i] > MAX_DEVICES * 0.85):
            labels[i] = 1  # Yoğun
        else:
            labels[i] = 0  # Normal
        
        label_names.append(['Normal', 'Yoğun', 'Panik', 'Kriz'][labels[i]])
    
    return labels, label_names

labels, label_names = generate_labels(minutes, handover_rate, voice_call_rate, data_traffic_mbps)

# ─────────────────────────────────────────────
# ANOMALİ SKORU (0-1, AI'ın çıktısını simüle eder)
# ─────────────────────────────────────────────
def generate_anomaly_score(labels, minutes):
    base_score = labels / 3.0  # 0-1 normalize
    # Biraz erken uyarı (gerçekçilik için model önceden sezer)
    early_warning = gaussian_bump(minutes, 127, 5, 0.3)
    noise = np.random.normal(0, 0.03, len(minutes))
    score = base_score + early_warning + noise
    return np.clip(score, 0, 1).round(3)

anomaly_score = generate_anomaly_score(labels, minutes)

# ─────────────────────────────────────────────
# SLICE DAĞILIMI (AI kararı simülasyonu)
# ─────────────────────────────────────────────
def generate_slice_allocation(anomaly_score):
    entertainment = np.zeros(len(anomaly_score))
    general = np.zeros(len(anomaly_score))
    emergency = np.zeros(len(anomaly_score))
    
    for i, score in enumerate(anomaly_score):
        if score < 0.2:
            entertainment[i], general[i], emergency[i] = 60, 30, 10
        elif score < 0.4:
            entertainment[i], general[i], emergency[i] = 50, 30, 20
        elif score < 0.6:
            entertainment[i], general[i], emergency[i] = 40, 35, 25
        elif score < 0.8:
            entertainment[i], general[i], emergency[i] = 20, 25, 55
        else:
            entertainment[i], general[i], emergency[i] = 5, 15, 80
    
    return entertainment, general, emergency

slice_entertainment, slice_general, slice_emergency = generate_slice_allocation(anomaly_score)

# ─────────────────────────────────────────────
# DATAFRAME OLUŞTUR
# ─────────────────────────────────────────────
df = pd.DataFrame({
    'timestamp': timestamps,
    'minute': minutes.round(1),
    'connected_devices': connected_devices,
    'handover_rate_per_min': handover_rate,
    'data_traffic_mbps': data_traffic_mbps,
    'voice_call_rate': voice_call_rate,
    'avg_rssi_dbm': avg_rssi_dbm,
    'avg_latency_ms': avg_latency_ms,
    **sector_data,
    'anomaly_score': anomaly_score,
    'slice_entertainment_pct': slice_entertainment.astype(int),
    'slice_general_pct': slice_general.astype(int),
    'slice_emergency_pct': slice_emergency.astype(int),
    'status_label': labels,
    'status_name': label_names
})

# ─────────────────────────────────────────────
# CSV KAYDET
# ─────────────────────────────────────────────
os.makedirs(OUTPUT_DIR, exist_ok=True)
csv_path = os.path.join(OUTPUT_DIR, 'crowdslice_simulation_data.csv')
df.to_csv(csv_path, index=False)
print(f"✅ Veri seti oluşturuldu: {len(df)} satır, {len(df.columns)} sütun")
print(f"   Kayıt: {csv_path}")
print(f"\nDurum dağılımı:")
print(df['status_name'].value_counts())
print(f"\nİlk 5 satır:")
print(df[['minute','connected_devices','handover_rate_per_min','voice_call_rate','anomaly_score','status_name']].head())

# ─────────────────────────────────────────────
# VİZÜELLEŞTİRME
# ─────────────────────────────────────────────
fig, axes = plt.subplots(4, 2, figsize=(18, 20))
fig.patch.set_facecolor('#0f0f1a')
for ax in axes.flatten():
    ax.set_facecolor('#1a1a2e')
    ax.tick_params(colors='#cccccc')
    ax.xaxis.label.set_color('#cccccc')
    ax.yaxis.label.set_color('#cccccc')
    ax.title.set_color('#ffffff')
    for spine in ax.spines.values():
        spine.set_edgecolor('#333355')

colors = {'Normal': '#00ff88', 'Yoğun': '#ffaa00', 'Panik': '#ff4444', 'Kriz': '#ff0000'}
status_colors = [colors[s] for s in df['status_name']]

# Renkli arka plan bantları
def add_status_background(ax):
    for i in range(len(df) - 1):
        c = {'Normal': '#00ff8808', 'Yoğun': '#ffaa0015', 
             'Panik': '#ff444425', 'Kriz': '#ff000035'}[df['status_name'].iloc[i]]
        ax.axvspan(df['minute'].iloc[i], df['minute'].iloc[i+1], alpha=1, color=c, linewidth=0)
    ax.axvline(x=130, color='#ff4444', linestyle='--', alpha=0.7, linewidth=1.5, label='Panik anı')

# 1. Bağlı Cihaz
axes[0,0].plot(df['minute'], df['connected_devices'], color='#00d4ff', linewidth=1.5)
add_status_background(axes[0,0])
axes[0,0].set_title('Bağlı Cihaz Sayısı', fontweight='bold', fontsize=11)
axes[0,0].set_ylabel('Cihaz')
axes[0,0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x/1000)}K'))

# 2. Handover Rate
axes[0,1].plot(df['minute'], df['handover_rate_per_min'], color='#ff9500', linewidth=1.5)
add_status_background(axes[0,1])
axes[0,1].set_title('Handover Hızı (/ dakika) — Panik Dedektörü', fontweight='bold', fontsize=11)
axes[0,1].set_ylabel('Handover/dak')
axes[0,1].axhline(y=300, color='#ff4444', linestyle=':', alpha=0.8, linewidth=1, label='Panik eşiği')
axes[0,1].legend(facecolor='#1a1a2e', labelcolor='white', fontsize=8)

# 3. Veri Trafiği
axes[1,0].plot(df['minute'], df['data_traffic_mbps']/1000, color='#a855f7', linewidth=1.5)
add_status_background(axes[1,0])
axes[1,0].set_title('Veri Trafiği (Gbps) — Panik anında düşer', fontweight='bold', fontsize=11)
axes[1,0].set_ylabel('Gbps')

# 4. Sesli Arama
axes[1,1].plot(df['minute'], df['voice_call_rate'], color='#f43f5e', linewidth=1.5)
add_status_background(axes[1,1])
axes[1,1].set_title('Sesli Arama Oranı — Panik anında patlar', fontweight='bold', fontsize=11)
axes[1,1].set_ylabel('Arama/dak')

# 5. RSSI & Latency
ax5 = axes[2,0]
ax5b = ax5.twinx()
ax5.plot(df['minute'], df['avg_rssi_dbm'], color='#22c55e', linewidth=1.5, label='RSSI (dBm)')
ax5b.plot(df['minute'], df['avg_latency_ms'], color='#fb923c', linewidth=1.5, linestyle='--', label='Latency (ms)')
ax5.set_title('Sinyal Kalitesi ve Gecikme', fontweight='bold', fontsize=11, color='white')
ax5.set_ylabel('RSSI (dBm)', color='#22c55e')
ax5b.set_ylabel('Latency (ms)', color='#fb923c')
ax5.set_facecolor('#1a1a2e')
ax5b.tick_params(colors='#cccccc')
add_status_background(ax5)

# 6. Sektör Dağılımı
axes[2,1].stackplot(df['minute'], 
    df['sector_A_devices'], df['sector_B_devices'],
    df['sector_C_devices'], df['sector_D_devices'],
    labels=['Sektör A', 'Sektör B', 'Sektör C (sahne)', 'Sektör D'],
    colors=['#3b82f6', '#8b5cf6', '#ef4444', '#10b981'], alpha=0.8)
axes[2,1].set_title('Sektör Bazlı Cihaz Dağılımı', fontweight='bold', fontsize=11)
axes[2,1].legend(facecolor='#1a1a2e', labelcolor='white', fontsize=8, loc='upper left')
axes[2,1].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x/1000)}K'))
axes[2,1].axvline(x=130, color='#ff4444', linestyle='--', alpha=0.7, linewidth=1.5)

# 7. Anomali Skoru
axes[3,0].fill_between(df['minute'], df['anomaly_score'], 
                        where=df['anomaly_score'] > 0.6,
                        color='#ff4444', alpha=0.4, label='Kritik bölge')
axes[3,0].plot(df['minute'], df['anomaly_score'], color='#ff6b6b', linewidth=2)
axes[3,0].axhline(y=0.6, color='#ffaa00', linestyle='--', alpha=0.8, linewidth=1, label='Alarm eşiği')
axes[3,0].axhline(y=0.8, color='#ff0000', linestyle='--', alpha=0.8, linewidth=1, label='Kriz eşiği')
axes[3,0].set_title('AI Anomali Skoru (0-1)', fontweight='bold', fontsize=11)
axes[3,0].set_ylabel('Skor')
axes[3,0].set_ylim(0, 1.1)
axes[3,0].legend(facecolor='#1a1a2e', labelcolor='white', fontsize=8)
add_status_background(axes[3,0])

# 8. Slice Dağılımı
axes[3,1].stackplot(df['minute'],
    df['slice_entertainment_pct'], df['slice_general_pct'], df['slice_emergency_pct'],
    labels=['Eğlence Slice', 'Genel Slice', 'Acil Durum Slice'],
    colors=['#06b6d4', '#84cc16', '#ef4444'], alpha=0.85)
axes[3,1].set_title('AI Network Slice Kararları (%)', fontweight='bold', fontsize=11)
axes[3,1].set_ylabel('%')
axes[3,1].legend(facecolor='#1a1a2e', labelcolor='white', fontsize=8)
axes[3,1].axvline(x=130, color='#ff4444', linestyle='--', alpha=0.7, linewidth=1.5, label='Panik anı')

# Ortak X ekseni etiketleri
for ax in axes.flatten():
    ax.set_xlabel('Etkinlik süresi (dakika)')
    ax.set_xlim(0, EVENT_DURATION_MIN)

# Legend için renk bandı açıklaması
legend_elements = [
    mpatches.Patch(color='#00ff8820', label='Normal'),
    mpatches.Patch(color='#ffaa0030', label='Yoğun'),
    mpatches.Patch(color='#ff444440', label='Panik'),
    mpatches.Patch(color='#ff000050', label='Kriz'),
]
fig.legend(handles=legend_elements, loc='upper center', ncol=4,
           facecolor='#1a1a2e', labelcolor='white', fontsize=10,
           bbox_to_anchor=(0.5, 0.98))

plt.suptitle('CrowdSlice AI — 5G Konser Etkinliği Simülasyon Verisi\n(T=130 dk: Panik Senaryosu)',
             fontsize=16, fontweight='bold', color='white', y=1.01)

plt.tight_layout()
img_path = os.path.join(OUTPUT_DIR, 'crowdslice_visualization.png')
plt.savefig(img_path, 
            dpi=150, bbox_inches='tight', facecolor='#0f0f1a')
print(f"\n✅ Görselleştirme kaydedildi: {img_path}")

# İstatistik özeti
print("\n" + "="*50)
print("VERİ SETİ ÖZETİ")
print("="*50)
print(f"Toplam veri noktası   : {len(df)}")
print(f"Zaman aralığı         : {df['timestamp'].iloc[0]} → {df['timestamp'].iloc[-1]}")
print(f"Örnekleme sıklığı     : {SAMPLE_INTERVAL_SEC} saniye")
print(f"Sektör sayısı         : {N_SECTORS}")
print(f"\nDurum dağılımı:")
for status, count in df['status_name'].value_counts().items():
    pct = count / len(df) * 100
    print(f"  {status:8s}: {count:4d} örnek ({pct:.1f}%)")
print(f"\nPanik başlangıcı (anomaly_score > 0.6): dakika {df[df['anomaly_score']>0.6]['minute'].min():.0f}")
print(f"Panik zirvesi (max handover)          : dakika {df.loc[df['handover_rate_per_min'].idxmax(), 'minute']:.0f}")
