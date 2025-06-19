# -*- coding: utf-8 -*-
"""
TPS Distance Collector - Program untuk mengumpulkan jarak jalan antar TPS
VERSI DENGAN VERIFIKASI BERGANDA: Menerapkan iterasi multiple untuk memastikan jarak konsisten
"""

import numpy as np
import pandas as pd
import pickle
import os
import requests
import time
import statistics
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import locale

# Set locale untuk format angka yang benar
try:
    # Gunakan locale Indonesia jika tersedia
    locale.setlocale(locale.LC_NUMERIC, 'id_ID.UTF-8')
except:
    try:
        # Fallback ke US locale
        locale.setlocale(locale.LC_NUMERIC, 'en_US.UTF-8')
    except:
        pass  # Jika gagal, gunakan default sistem

# ==========================================================
# AREA INPUT DATA
# ==========================================================
# Anda bisa mengganti data di bawah ini dengan data TPS Anda
# Format: [nama_tps, latitude, longitude, volume]

def input_data():
    """
    Masukkan data TPS Anda di sini atau baca dari file CSV.
    
    Jika menggunakan CSV, pastikan struktur kolom adalah:
    nama,latitude,longitude,volume
    
    Returns:
        pandas.DataFrame: Data TPS dengan kolom nama, latitude, longitude, volume
    """
    # OPSI 1: Masukkan data langsung di sini
    # Format: ["Nama TPS", latitude, longitude, volume]
    data = [
    ["SD Sugiyopranoto", -7.693398300, 110.614024000, 0.250],
    ["Man Sangkalputung", -7.689727100, 110.610051000, 1.950],
    ["SD Alam Selfa", -7.682906800, 110.601103900, 0.200],
    ["Sanden 1", -7.677609400, 110.592838400, 7.170],
    ["TPS 3R Karanglo", -7.691079200, 110.581663700, 6.670],
    ["Balang RW.13", -7.697975930, 110.584441570, 1.595],
    ["SMPN 5 Klaten", -7.725617000, 110.610008300, 1.378],
    ["Panti Semedi", -7.690976300, 110.614134200, 0.870],
    ["Sanden 2", -7.677474500, 110.592932900, 6.670],
    ["CV. Maitama", -7.685687200, 110.621837900, 0.870],
    ["CV. Gendewa Warastra", -7.677782100, 110.665824800, 2.280],
    ["Balang RW.14", -7.699825480, 110.583697160, 1.885],
    ["Jetis", -7.698404270, 110.581107840, 1.305],
    ["Malangjiwan", -7.688277850, 110.565474060, 6.960],
    ["Kaloran RW.08", -7.702138820, 110.593799510, 2.320],
    ["Kaloran RW.06", -7.700681930, 110.592059950, 1.160],
    ["Ketinggen", -7.696826220, 110.587402650, 2.610],
    ["Balang RW. 12", -7.698880100, 110.586303070, 1.882],
    ["SMAN 2  Klaten", -7.720963255, 110.575990369, 2.320],
    ["Desa Trunuh", -7.723280931, 110.575893810, 0.870],
    ["CV. Dharma Prigga Birama", -7.715686497, 110.567757974, 2.096],
    ["TPS Danguran", -7.734229446, 110.583119035, 12.202],
    ["SMPN 7 Klaten", -7.717825124, 110.580695075, 3.190],
    ["TPS Gudang", -7.723669441, 110.581509813, 8.670],
    ["Gebal Birit", -7.754167066, 110.582258443, 12.810],
    ["Perum Puri", -7.732649944005022, 110.59394590633234, 5.800],
    ["bar puri", -7.734256171237, 110.5833722455, 0],
    ["Bentakan", -7.729499965, 110.588552706, 5.220],
    ["Ds Sukorejo", -7.761577293, 110.602813544, 6.780],
    ["Tegalyoso", -7.718574360, 110.591323020, 3.190],
    ["TPS3R Gajah Blorok 1", -7.719837357, 110.595266827, 9.660],
    ["Sumberejo", -7.716807215, 110.585899611, 2.900],
    ["TPS3R Pengkol 1", -7.718926779, 110.586931052, 8.114],
    ["Bendo Buntalan 1", -7.720471682, 110.600961807, 3.480],
    ["TPS3R Gajah Blorok 2", -7.719859739, 110.595409750, 9.370],
    ["Lemah Ireng", -7.722813593, 110.601648637, 2.900],
    ["Sobrah", -7.727053563, 110.600334914, 9.410],
    ["TPS Pengkol", -7.719421552, 110.587016136, 2.320],
    ["TPS3R Pengkol 2", -7.718890628, 110.587076071, 8.622],
    ["Bendo Buntalan 2", -7.720607562, 110.600965291, 3.190],
    ["TPS3R Gajah Blorok 3", -7.719877065, 110.595422286, 9.120],
    ["Pasar Srago 1", -7.711604800, 110.609417300, 12.000],
    ["SD Islam El Yaomy", -7.670729900, 110.689317300, 1.740],
    ["TPS Tegalrejo (Ceper)", -7.669359600, 110.694436800, 11.165],
    ["PT Mondrian", -7.706865400, 110.614921400, 6.525],
    ["Srago Cilik", -7.714382980, 110.616847380, 3.335],
    ["SMP 6 Klaten", -7.696956100, 110.605068400, 2.030],
    ["SMP 1 Klaten", -7.696629300, 110.604746400, 0.870],
    ["Perum Dokter Kencana", -7.695885440, 110.625420740, 0.580],
    ["Perum Kencana", -7.699847600, 110.623299300, 1.160],
    ["Unwidha", -7.692787700, 110.624044800, 5.510],
    ["Pasar Srago 2", -7.711584100, 110.609387300, 12.148],
    ["Metuk Lor 1", -7.705610961, 110.585242000, 12.983],
    ["Pasar Gayamprit 1", -7.70337271, 110.588611100, 12.983],
    ["Sendangan 1", -7.706727000, 110.621080900, 2.030],
    ["Sendangan 2", -7.704989400, 110.621511300, 0.870],
    ["Sendangan 3", -7.706446800, 110.622385200, 0.580],
    ["Sendangan 4", -7.706965300, 110.623474200, 1.160],
    ["Desa Jogosetran", -7.701533000, 110.641699600, 8.343],
    ["TPS3R Nglinggi", -7.702967100, 110.582725400, 1.740],
    ["Metok Lor 2", -7.705652900, 110.585257100, 10.755],
    ["Metok Lor 3", -7.705624300, 110.585224700, 6.525],
    ["Pasar Gayamprit 2", -7.703262280, 110.588751700, 5.385],
    ["TPS 3R Balang Wetan", -7.687780500, 110.626055700, 6.525],
    ["Plembon RW 10", -7.689503700, 110.626563700, 2.449],
    ["Perum Dokter RSI", -7.684948250, 110.636462570, 0.725],
    ["Perum Griya Prima 1", -7.683513600, 110.627246600, 8.480],
    ["TPS Gading Wetan RW 12", -7.682861400, 110.633657700, 2.220],
    ["TPS Gading Wetan RW 14", -7.685168910, 110.633408380, 2.930],
    ["Jomboran", -7.724722800, 110.607385200, 2.600],
    ["Jetis RW 15", -7.686297000, 110.630344000, 6.914],
    ["Ngaran RW 01", -7.683479300, 110.637412400, 3.411],
    ["WM Bu Sum Klaten", -7.685944400, 110.640041800, 0.400],
    ["Dinas Perhubungan Klaten", -7.687116600, 110.636557300, 1.920],
    ["Ketandan", -7.690120000, 110.632459400, 5.800],
    ["Plembon RW 2 (Selatan)", -7.690350400, 110.629007300, 2.200],
    ["Plembon RW 2 (Utara)", -7.689235800, 110.628696100, 1.970],
    ["Perum Griya Prima 2", -7.683448000, 110.627136000, 6.090],
    ["Geritan", -7.680107700, 110.627345800, 2.863],
    ["Mayungan", -7.677614100, 110.623453800, 2.468],
    ["Garasi", -7.707675046626699, 110.61207471510201, 0],
    ["TPA Troketon", -7.677580518930298, 110.7179057656785, 0]
]
    
    # OPSI 2: Baca dari file CSV
    # Uncomment baris di bawah ini dan sesuaikan nama file CSV Anda
    # return pd.read_csv("data_tps.csv")
    
    return pd.DataFrame(data, columns=['nama', 'latitude', 'longitude', 'volume'])

# ==========================================================
# PENGATURAN PROGRAM
# ==========================================================

def get_settings():
    """
    Pengaturan program untuk menghitung jarak jalan antar TPS.
    
    Returns:
        dict: Pengaturan program
    """
    settings = {
        "VEHICLE_TYPE": "truck",           # Gunakan "car" untuk jalan mobil
        "NUM_THREADS": 6,                # Jumlah thread untuk proses paralel (dikurangi untuk menghindari rate limit)
        "DATABASE_DIR": "tps_distances",  # Direktori untuk menyimpan database
        "EXPORT_EXCEL": True,            # Ekspor ke Excel
        "EXPORT_CSV": True,              # Ekspor ke CSV
        "EXPORT_MATRIX": True,           # Ekspor matriks jarak ke file Excel
        "SHOW_PROGRESS": True,           # Tampilkan progress bar
        "RETRY_FAILED": True,            # Coba ulang jika gagal
        "MAX_RETRIES": 4,                # Jumlah maksimal percobaan
        "RETRY_DELAY": 4,                # Waktu tunggu antar percobaan (detik)
        "BATCH_SAVE": 5,                 # Simpan database setiap X pasangan
        "REQUEST_DELAY": 2,            # Delay antar request (detik) untuk menghindari rate limit
        "DECIMAL_PRECISION": 3,          # Presisi angka desimal
        "FLOAT_FORMAT": "{:.3f}",        # Format angka desimal
        "VERIFICATION_ROUNDS": 5,        # Jumlah maksimum iterasi untuk verifikasi jarak
        "VERIFICATION_MIN_ITERATIONS": 3, # Jumlah minimum iterasi sebelum early stopping
        "VERIFICATION_THRESHOLD": 0.1,   # Threshold (km) untuk early stopping
        "VARIANCE_THRESHOLD": 0.5,       # Threshold untuk perbedaan jarak yang signifikan (km)
        "USE_VERIFICATION": True,        # Gunakan verifikasi berganda
        "USE_EARLY_STOPPING": True,      # Gunakan early stopping
        "MANUAL_VALIDATION_FILE": "manual_validations.csv"  # File untuk validasi manual
    }
    return settings

# ==========================================================
# FUNGSI PERHITUNGAN JARAK
# ==========================================================

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Hitung jarak Haversine (jarak udara) antara dua titik.
    
    Args:
        lat1, lon1: Koordinat titik pertama
        lat2, lon2: Koordinat titik kedua
        
    Returns:
        float: Jarak dalam kilometer
    """
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371  # Radius bumi dalam kilometer
    return c * r

class TPSDistanceCollector:
    """
    Kelas untuk mengumpulkan, menyimpan, dan mengolah data jarak antar TPS.
    """
    
    def __init__(self, settings):
        """
        Inisialisasi dengan pengaturan program.
        
        Args:
            settings (dict): Pengaturan program
        """
        self.settings = settings
        self.database_dir = settings["DATABASE_DIR"]
        
        if not os.path.exists(self.database_dir):
            os.makedirs(self.database_dir)
        
        # File database untuk menyimpan jarak antar titik
        self.db_file = os.path.join(self.database_dir, "tps_distances.pkl")
        
        # Load database jika ada
        self.distance_db = {}
        if os.path.exists(self.db_file):
            try:
                with open(self.db_file, 'rb') as f:
                    self.distance_db = pickle.load(f)
                print(f"Database jarak jalan dimuat: {len(self.distance_db)} pasang TPS")
            except Exception as e:
                print(f"Error memuat database: {e}")
                self.distance_db = {}
        
        # Track failures
        self.failures = []
        
        # Manual validations
        self.manual_validations = {}
        self.load_manual_validations()
        
        # Store original TPS data to ensure we always have the names
        self.original_tps_data = None
        
        # Track verifications
        self.verification_stats = {
            "total_verifications": 0,
            "high_variance_count": 0,
            "successful_verifications": 0,
            "failed_verifications": 0
        }
    
    def load_manual_validations(self):
        """
        Load manual validations from CSV file.
        """
        validation_file = os.path.join(self.database_dir, self.settings["MANUAL_VALIDATION_FILE"])
        if os.path.exists(validation_file):
            try:
                df = pd.read_csv(validation_file)
                for _, row in df.iterrows():
                    key = self.get_distance_key(row['tps1_id'], row['tps2_id'])
                    self.manual_validations[key] = row['known_distance']
                print(f"Loaded {len(self.manual_validations)} manual validations")
            except Exception as e:
                print(f"Error loading manual validations: {e}")
    
    def save_manual_validations(self):
        """
        Save manual validations to CSV file.
        """
        if not self.manual_validations:
            return
            
        validation_file = os.path.join(self.database_dir, self.settings["MANUAL_VALIDATION_FILE"])
        try:
            data = []
            for key, distance in self.manual_validations.items():
                tps1_id, tps2_id = key.split('_')
                data.append({
                    'tps1_id': int(tps1_id),
                    'tps2_id': int(tps2_id),
                    'known_distance': distance
                })
            df = pd.DataFrame(data)
            df.to_csv(validation_file, index=False)
            print(f"Saved {len(self.manual_validations)} manual validations")
        except Exception as e:
            print(f"Error saving manual validations: {e}")
    
    def add_manual_validation(self, tps1_id, tps2_id, known_distance):
        """
        Add a manual validation for a TPS pair.
        
        Args:
            tps1_id, tps2_id: IDs of the TPS pair
            known_distance: Known distance in kilometers
        
        Returns:
            bool: True if validation was added, False otherwise
        """
        key = self.get_distance_key(tps1_id, tps2_id)
        self.manual_validations[key] = known_distance
        
        # Update in database if exists
        if key in self.distance_db:
            self.distance_db[key]['distance'] = known_distance
            self.distance_db[key]['source'] = 'manual_validation'
            self.distance_db[key]['verification_note'] = 'Manually validated distance'
            
            # Save database
            self.save_database()
            self.save_manual_validations()
            
            print(f"Manual validation added for {self.get_tps_name(tps1_id)} to {self.get_tps_name(tps2_id)}: {known_distance} km")
            return True
        else:
            print(f"Warning: Cannot find pair in database. Will apply validation when distance is calculated.")
            self.save_manual_validations()
            return False
    
    def get_distance_key(self, tps1_id, tps2_id):
        """
        Membuat kunci unik untuk pasangan TPS.
        
        Args:
            tps1_id, tps2_id: ID TPS
            
        Returns:
            str: Kunci unik untuk pasangan TPS
        """
        # Ensure consistent order
        if int(tps1_id) <= int(tps2_id):
            return f"{tps1_id}_{tps2_id}"
        else:
            return f"{tps2_id}_{tps1_id}"
    
    def get_coord_key(self, lat1, lon1, lat2, lon2):
        """
        Membuat kunci unik untuk pasangan koordinat.
        
        Args:
            lat1, lon1, lat2, lon2: Koordinat dua titik
            
        Returns:
            str: Kunci unik untuk pasangan koordinat
        """
        # Urutkan koordinat agar (A,B) dan (B,A) menghasilkan kunci yang sama
        coords = sorted([(lat1, lon1), (lat2, lon2)])
        return f"{coords[0][0]:.6f}_{coords[0][1]:.6f}_{coords[1][0]:.6f}_{coords[1][1]:.6f}"
    
    def get_verified_road_distance(self, tps1_id, tps2_id, lat1, lon1, lat2, lon2, vehicle_type="truck"):
        """
        Mendapatkan jarak jalan antara dua TPS dengan verifikasi berganda dan penghentian dini.
        
        Args:
            tps1_id, tps2_id: ID TPS
            lat1, lon1, lat2, lon2: Koordinat dua titik
            vehicle_type: Jenis kendaraan ("car" atau "truck")
            
        Returns:
            float: Jarak jalan dalam kilometer
            dict: Metadata verifikasi
        """
        # Check if we have a manual validation
        distance_key = self.get_distance_key(tps1_id, tps2_id)
        if distance_key in self.manual_validations:
            return self.manual_validations[distance_key], {
                "source": "manual_validation",
                "note": "Manually validated distance",
                "iterations": 0,
                "variance": 0,
                "distances": []
            }
        
        # Run verification with early stopping
        self.verification_stats["total_verifications"] += 1
        
        # Collect measurements
        distances = []
        durations = []
        sources = []
        errors = []
        
        # Maximum verification rounds
        max_rounds = self.settings["VERIFICATION_ROUNDS"]
        
        # Threshold for early stopping (smaller value means more precise)
        # Jika perbedaan jarak kurang dari threshold_km, hentikan iterasi
        threshold_km = 0.1  # 100 meter
        
        # Minimum jumlah iterasi sebelum boleh berhenti
        min_iterations = 2
        
        for round_num in range(max_rounds):
            try:
                # Format koordinat untuk OSRM (longitude dulu, lalu latitude)
                osrm_profile = "driving"  # Selalu gunakan "driving" untuk mobil
                
                url = f"http://router.project-osrm.org/route/v1/{osrm_profile}/{lon1},{lat1};{lon2},{lat2}"
                params = {
                    "overview": "false",
                    "alternatives": "false",
                    "steps": "false"
                }
                
                # Tambahkan delay yang lebih lama untuk iterasi lanjutan
                if round_num > 0:
                    # Delay progresif tetapi tidak terlalu lama
                    extra_delay = round_num * 0.5  # Tambahkan 0.5 detik untuk setiap ronde
                    time.sleep(extra_delay)
                
                response = requests.get(url, params=params, timeout=15)  # Timeout
                data = response.json()
                
                if data.get("code") == "Ok" and len(data.get("routes", [])) > 0:
                    # Jarak dalam kilometer
                    distance = data["routes"][0]["distance"] / 1000
                    duration = data["routes"][0]["duration"] / 60  # dalam menit
                    
                    distances.append(distance)
                    durations.append(duration)
                    sources.append("osrm")
                    
                    # Check if we can stop early (after min_iterations)
                    if self.settings["USE_EARLY_STOPPING"] and round_num >= self.settings["VERIFICATION_MIN_ITERATIONS"] - 1:
                        # Get only OSRM distances for consistency check
                        osrm_distances = [d for i, d in enumerate(distances) if sources[i] == "osrm"]
                        
                        # If we have at least min_iterations OSRM measurements
                        if len(osrm_distances) >= self.settings["VERIFICATION_MIN_ITERATIONS"]:
                            # Check consistency of the last min_iterations measurements
                            recent_distances = osrm_distances[-self.settings["VERIFICATION_MIN_ITERATIONS"]:]
                            max_diff = max(recent_distances) - min(recent_distances)
                            
                            # If measurements are consistent, stop early
                            if max_diff <= self.settings["VERIFICATION_THRESHOLD"]:
                                # Uncomment line below for debugging
                                # print(f"Early stopping at iteration {round_num+1} for {self.get_tps_name(tps1_id)} - {self.get_tps_name(tps2_id)}")
                                break
                else:
                    errors.append(f"OSRM response not OK: {data.get('code')}")
                    
                    # Gunakan haversine sebagai fallback untuk iterasi ini
                    hav_distance = haversine_distance(lat1, lon1, lat2, lon2) * 1.3  # Faktor jalan sekitar 1.3
                    distances.append(hav_distance)
                    durations.append(hav_distance * 2)  # Asumsi kecepatan rata-rata 30 km/jam
                    sources.append("haversine_fallback")
                    
            except Exception as e:
                errors.append(str(e))
                
                # Gunakan haversine sebagai fallback untuk iterasi ini
                hav_distance = haversine_distance(lat1, lon1, lat2, lon2) * 1.3  # Faktor jalan sekitar 1.3
                distances.append(hav_distance)
                durations.append(hav_distance * 2)  # Asumsi kecepatan rata-rata 30 km/jam
                sources.append("haversine_fallback")
        
        # Analyze the results
        verification_meta = {
            "iterations": len(distances),
            "distances": distances,
            "durations": durations,
            "sources": sources,
            "errors": errors,
            "early_stopped": len(distances) < max_rounds and "osrm" in sources
        }
        
        # If we have at least one successful OSRM measurement
        if "osrm" in sources:
            # Get only OSRM distances
            osrm_distances = [d for i, d in enumerate(distances) if sources[i] == "osrm"]
            
            if len(osrm_distances) > 0:
                # Check variance (consistency)
                if len(osrm_distances) > 1:
                    variance = max(osrm_distances) - min(osrm_distances)
                    verification_meta["variance"] = variance
                    
                    if variance > self.settings["VARIANCE_THRESHOLD"]:
                        self.verification_stats["high_variance_count"] += 1
                        verification_meta["note"] = f"High variance detected: {variance:.3f} km"
                        
                        # Untuk kasus variasi tinggi, tambahkan log detail
                        tps1_name = self.get_tps_name(tps1_id)
                        tps2_name = self.get_tps_name(tps2_id)
                        print(f"PERHATIAN: Variasi tinggi untuk {tps1_name} ke {tps2_name}: {variance:.3f} km")
                        print(f"  Pengukuran: {[round(d, 3) for d in osrm_distances]} km")
                    else:
                        verification_meta["note"] = "Consistent measurements"
                        if verification_meta.get("early_stopped"):
                            verification_meta["note"] += " (early stopping)"
                
                # Use median for 3+ measurements, mean for 2
                if len(osrm_distances) >= 3:
                    verified_distance = statistics.median(osrm_distances)
                    verification_meta["source"] = "osrm_verified_median"
                else:
                    verified_distance = statistics.mean(osrm_distances)
                    verification_meta["source"] = "osrm_verified_mean"
                
                self.verification_stats["successful_verifications"] += 1
                return verified_distance, verification_meta
        
        # If all failed, use improved haversine estimation
        self.verification_stats["failed_verifications"] += 1
        haversine_est = haversine_distance(lat1, lon1, lat2, lon2) * 1.3  # Apply road factor
        verification_meta["source"] = "haversine_estimated"
        verification_meta["note"] = "All OSRM requests failed, using Haversine with road factor"
        return haversine_est, verification_meta
    
    def get_road_distance(self, tps1_id, tps2_id, lat1, lon1, lat2, lon2, vehicle_type="truck", force_refresh=False):
        """
        Mendapatkan jarak jalan antara dua TPS.
        
        Args:
            tps1_id, tps2_id: ID TPS
            lat1, lon1, lat2, lon2: Koordinat dua titik
            vehicle_type: Jenis kendaraan ("car" atau "truck")
            force_refresh: Jika True, hitung ulang meskipun sudah ada di database
            
        Returns:
            float: Jarak jalan dalam kilometer
        """
        # Jika titiknya sama, kembalikan 0
        if tps1_id == tps2_id or (lat1 == lat2 and lon1 == lon2):
            return 0.0
        
        # Dapatkan kunci untuk pasangan TPS ini
        tps_key = self.get_distance_key(tps1_id, tps2_id)
        
        # Check manual validations first
        if tps_key in self.manual_validations and not force_refresh:
            # Update database if needed
            if tps_key in self.distance_db:
                self.distance_db[tps_key]['distance'] = self.manual_validations[tps_key]
                self.distance_db[tps_key]['source'] = 'manual_validation'
            return self.manual_validations[tps_key]
        
        # Jika sudah ada di database dan tidak dipaksa refresh, gunakan nilai yang ada
        if tps_key in self.distance_db and not force_refresh:
            return self.distance_db[tps_key]['distance']
        
        # Jika belum ada atau force_refresh, hitung dengan verifikasi berganda
        if self.settings["USE_VERIFICATION"]:
            distance, verification_meta = self.get_verified_road_distance(
                tps1_id, tps2_id, lat1, lon1, lat2, lon2, vehicle_type
            )
            
            # Simpan ke database dengan metadata verifikasi
            self.distance_db[tps_key] = {
                'distance': distance,
                'duration': verification_meta.get('durations', [0])[0] if verification_meta.get('durations') else 0,
                'tps1_id': tps1_id,
                'tps2_id': tps2_id,
                'lat1': lat1,
                'lon1': lon1,
                'lat2': lat2,
                'lon2': lon2,
                'vehicle': vehicle_type,
                'source': verification_meta.get('source', 'unknown'),
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'verification': {
                    'iterations': verification_meta.get('iterations', 0),
                    'variance': verification_meta.get('variance', 0),
                    'note': verification_meta.get('note', '')
                }
            }
        else:
            # Gunakan metode lama tanpa verifikasi berganda
            max_retries = self.settings["MAX_RETRIES"] if self.settings["RETRY_FAILED"] else 1
            
            for attempt in range(max_retries):
                try:
                    # Format koordinat untuk OSRM (longitude dulu, lalu latitude)
                    osrm_profile = "driving"  # Selalu gunakan "driving" untuk mobil
                    
                    url = f"http://router.project-osrm.org/route/v1/{osrm_profile}/{lon1},{lat1};{lon2},{lat2}"
                    params = {
                        "overview": "false",
                        "alternatives": "false",
                        "steps": "false"
                    }
                    
                    response = requests.get(url, params=params, timeout=15)
                    data = response.json()
                    
                    if data.get("code") == "Ok" and len(data.get("routes", [])) > 0:
                        # Jarak dalam kilometer
                        distance = data["routes"][0]["distance"] / 1000
                        duration = data["routes"][0]["duration"] / 60  # dalam menit
                        
                        # Simpan ke database
                        self.distance_db[tps_key] = {
                            'distance': distance,
                            'duration': duration,
                            'tps1_id': tps1_id,
                            'tps2_id': tps2_id,
                            'lat1': lat1,
                            'lon1': lon1,
                            'lat2': lat2,
                            'lon2': lon2,
                            'vehicle': vehicle_type,
                            'source': 'osrm',
                            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        }
                        
                        break  # Keluar dari loop jika berhasil
                    else:
                        # Jika gagal mendapatkan rute, tunggu sebelum mencoba lagi
                        if attempt < max_retries - 1:
                            time.sleep(self.settings["RETRY_DELAY"])
                        else:
                            # Gunakan Haversine sebagai fallback
                            distance = haversine_distance(lat1, lon1, lat2, lon2) * 1.3  # Apply road factor
                            self.distance_db[tps_key] = {
                                'distance': distance,
                                'tps1_id': tps1_id,
                                'tps2_id': tps2_id,
                                'lat1': lat1,
                                'lon1': lon1,
                                'lat2': lat2,
                                'lon2': lon2,
                                'vehicle': 'haversine',
                                'source': 'haversine',
                                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                'note': 'Fallback to Haversine: OSRM route not found'
                            }
                            
                            # Track failure
                            self.failures.append({
                                'tps1_id': tps1_id,
                                'tps2_id': tps2_id,
                                'reason': 'OSRM route not found'
                            })
                except Exception as e:
                    # Jika error, tunggu sebelum mencoba lagi
                    if attempt < max_retries - 1:
                        time.sleep(self.settings["RETRY_DELAY"])
                    else:
                        # Gunakan Haversine sebagai fallback
                        distance = haversine_distance(lat1, lon1, lat2, lon2) * 1.3  # Apply road factor
                        self.distance_db[tps_key] = {
                            'distance': distance,
                            'tps1_id': tps1_id,
                            'tps2_id': tps2_id,
                            'lat1': lat1,
                            'lon1': lon1,
                            'lat2': lat2,
                            'lon2': lon2,
                            'vehicle': 'haversine',
                            'source': 'haversine',
                            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            'note': f'Fallback to Haversine: {str(e)}'
                        }
                        
                        # Track failure
                        self.failures.append({
                            'tps1_id': tps1_id,
                            'tps2_id': tps2_id,
                            'reason': str(e)
                        })
        
        # Simpan database secara periodik
        if len(self.distance_db) % self.settings["BATCH_SAVE"] == 0:
            self.save_database()
            
        return self.distance_db[tps_key]['distance']
    
    def collect_all_distances(self, df, vehicle_type="truck", num_threads=4):
        """
        Mengumpulkan jarak jalan antara semua pasangan TPS.
        
        Args:
            df: DataFrame dengan data TPS
            vehicle_type: Jenis kendaraan
            num_threads: Jumlah thread untuk proses paralel
            
        Returns:
            numpy.ndarray: Matriks jarak antar TPS
        """
        # Store original TPS data for later reference
        self.original_tps_data = df.copy()
        
        n = len(df)
        total_pairs = n * (n - 1) // 2
        
        print(f"Mengumpulkan jarak jalan untuk {n} TPS ({total_pairs} pasangan)...")
        
        # Prepare all pairs
        all_pairs = []
        for i in range(n):
            for j in range(i+1, n):  # Hanya pasangan unik (i,j) dengan i < j
                all_pairs.append((i, j))
        
        # Get existing pairs
        existing_pairs = set()
        for key, data in self.distance_db.items():
            if 'tps1_id' in data and 'tps2_id' in data:
                i, j = min(int(data['tps1_id']), int(data['tps2_id'])), max(int(data['tps1_id']), int(data['tps2_id']))
                if i < n and j < n:  # Ensure indices are valid
                    existing_pairs.add((i, j))
        
        # Filter pairs that need to be processed
        remaining_pairs = [pair for pair in all_pairs if pair not in existing_pairs]
        
        if not remaining_pairs:
            print("Semua jarak sudah ada dalam database! Tidak perlu perhitungan tambahan.")
        else:
            print(f"Perlu menghitung {len(remaining_pairs)} pasangan baru dari total {total_pairs} pasangan.")
            
            # Split into batches for parallel processing
            if num_threads > 1:
                batch_size = max(1, len(remaining_pairs) // num_threads + 1)
                batches = [remaining_pairs[i:i+batch_size] for i in range(0, len(remaining_pairs), batch_size)]
                
                # Process each batch
                with ThreadPoolExecutor(max_workers=num_threads) as executor:
                    futures = []
                    
                    for batch_id, batch in enumerate(batches):
                        future = executor.submit(
                            self._process_batch, 
                            df, 
                            batch, 
                            vehicle_type, 
                            batch_id,
                            self.settings["SHOW_PROGRESS"]
                        )
                        futures.append(future)
                    
                    # Wait for all to complete
                    for future in futures:
                        future.result()
            else:
                # Process all in one batch
                self._process_batch(df, remaining_pairs, vehicle_type, 0, self.settings["SHOW_PROGRESS"])
            
            # Save final database
            self.save_database()
            
            # Report verification statistics
            if self.settings["USE_VERIFICATION"]:
                print("\nVerifikasi Jarak:")
                print(f"Total verifikasi: {self.verification_stats['total_verifications']}")
                print(f"Verifikasi berhasil: {self.verification_stats['successful_verifications']} " +
                      f"({self.verification_stats['successful_verifications']/max(1, self.verification_stats['total_verifications'])*100:.1f}%)")
                print(f"Verifikasi gagal: {self.verification_stats['failed_verifications']} " +
                      f"({self.verification_stats['failed_verifications']/max(1, self.verification_stats['total_verifications'])*100:.1f}%)")
                print(f"Jarak dengan variasi tinggi: {self.verification_stats['high_variance_count']} " +
                      f"({self.verification_stats['high_variance_count']/max(1, self.verification_stats['total_verifications'])*100:.1f}%)")
        
        # Build distance matrix
        distance_matrix = np.zeros((n, n))
        
        for key, data in self.distance_db.items():
            if 'tps1_id' in data and 'tps2_id' in data:
                i, j = int(data['tps1_id']), int(data['tps2_id'])
                if i < n and j < n:  # Ensure indices are valid
                    distance_matrix[i, j] = data['distance']
                    distance_matrix[j, i] = data['distance']  # Symmetric
        
        # Return the distance matrix and information about failed requests
        return distance_matrix
    
    def _process_batch(self, df, pairs, vehicle_type, batch_id, show_progress):
        """
        Process a batch of TPS pairs.
        
        Args:
            df: DataFrame dengan data TPS
            pairs: List pasangan TPS yang akan diproses
            vehicle_type: Jenis kendaraan
            batch_id: ID batch untuk progress bar
            show_progress: Tampilkan progress bar atau tidak
        """
        if show_progress:
            pairs_iter = tqdm(pairs, desc=f"Batch {batch_id+1}", position=batch_id)
        else:
            pairs_iter = pairs
        
        for i, j in pairs_iter:
            tps1_id, tps2_id = i, j
            lat1, lon1 = df.iloc[i]['latitude'], df.iloc[i]['longitude']
            lat2, lon2 = df.iloc[j]['latitude'], df.iloc[j]['longitude']
            
            # Get road distance
            self.get_road_distance(
                tps1_id, tps2_id, lat1, lon1, lat2, lon2, 
                vehicle_type=vehicle_type
            )
            
            # Small delay to avoid overloading the API
            time.sleep(self.settings["REQUEST_DELAY"])
    
    def save_database(self):
        """
        Save the distance database to file.
        
        Returns:
            bool: True jika berhasil, False jika gagal
        """
        try:
            with open(self.db_file, 'wb') as f:
                pickle.dump(self.distance_db, f)
            return True
        except Exception as e:
            print(f"Error menyimpan database: {e}")
            return False
    
    def get_tps_name(self, tps_id):
        """
        Mendapatkan nama TPS dari ID.
        
        Args:
            tps_id: ID TPS
            
        Returns:
            str: Nama TPS
        """
        if self.original_tps_data is not None and int(tps_id) < len(self.original_tps_data):
            return self.original_tps_data.iloc[int(tps_id)]['nama']
        return f"TPS {tps_id}"
    
    def export_to_excel(self, df, filename=None):
        """
        Export the distance database to Excel file with TPS names.
        
        Args:
            df: DataFrame dengan data TPS
            filename: Nama file Excel (opsional)
            
        Returns:
            bool: True jika berhasil, False jika gagal
        """
        if not self.distance_db:
            print("Database kosong, tidak ada yang diekspor")
            return False
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(self.database_dir, f"tps_distances_{timestamp}.xlsx")
        
        try:
            # Pastikan kita menggunakan openpyxl untuk Excel
            try:
                import openpyxl
                excel_engine = 'openpyxl'
            except ImportError:
                print("PERHATIAN: Pustaka openpyxl tidak tersedia. Mencoba engine lain...")
                excel_engine = None
            
            # Prepare data for export
            export_data = []
            
            for key, data in self.distance_db.items():
                if 'tps1_id' in data and 'tps2_id' in data:
                    tps1_id, tps2_id = data['tps1_id'], data['tps2_id']
                    
                    # Skip self-distances
                    if tps1_id == tps2_id:
                        continue
                    
                    # Get TPS names - prioritize original names from stored data
                    tps1_name = self.get_tps_name(tps1_id)
                    tps2_name = self.get_tps_name(tps2_id)
                    
                    # Pastikan format angka yang benar
                    distance = round(data.get('distance', 0), self.settings["DECIMAL_PRECISION"])
                    duration = round(data.get('duration', 0), self.settings["DECIMAL_PRECISION"])
                    
                    # Get verification info
                    verification_note = ""
                    verification_variance = 0
                    if 'verification' in data:
                        verification_note = data['verification'].get('note', '')
                        verification_variance = data['verification'].get('variance', 0)
                    
                    # Create row with TPS information
                    row = {
                        'TPS_1_ID': tps1_id,
                        'TPS_1_Name': tps1_name,
                        'TPS_1_Latitude': data.get('lat1', df.iloc[int(tps1_id)]['latitude'] if int(tps1_id) < len(df) else 0),
                        'TPS_1_Longitude': data.get('lon1', df.iloc[int(tps1_id)]['longitude'] if int(tps1_id) < len(df) else 0),
                        'TPS_2_ID': tps2_id,
                        'TPS_2_Name': tps2_name,
                        'TPS_2_Latitude': data.get('lat2', df.iloc[int(tps2_id)]['latitude'] if int(tps2_id) < len(df) else 0),
                        'TPS_2_Longitude': data.get('lon2', df.iloc[int(tps2_id)]['longitude'] if int(tps2_id) < len(df) else 0),
                        'Distance_km': distance,
                        'Duration_min': duration,
                        'Source': data.get('source', 'unknown'),
                        'Vehicle_Type': data.get('vehicle', 'unknown'),
                        'Timestamp': data.get('timestamp', ''),
                        'Verification_Note': verification_note,
                        'Verification_Variance': verification_variance,
                        'Notes': data.get('note', '')
                    }
                    export_data.append(row)
            
            # Create DataFrame from the exported data
            export_df = pd.DataFrame(export_data)
            
            # Format angka dengan benar
            format_cols = ['TPS_1_Latitude', 'TPS_1_Longitude', 'TPS_2_Latitude', 'TPS_2_Longitude', 
                          'Distance_km', 'Duration_min', 'Verification_Variance']
            for col in format_cols:
                if col in export_df.columns:
                    export_df[col] = export_df[col].map(lambda x: round(float(x), self.settings["DECIMAL_PRECISION"]))
            
            # Export distance matrix if requested
            if self.settings["EXPORT_MATRIX"]:
                # Create a distance matrix with TPS names
                n = len(df)
                tps_names = df['nama'].tolist()
                
                # Initialize the matrix with zeros
                matrix_data = np.zeros((n, n))
                
                # Fill the matrix with distances
                for key, data in self.distance_db.items():
                    if 'tps1_id' in data and 'tps2_id' in data:
                        i, j = int(data['tps1_id']), int(data['tps2_id'])
                        if i < n and j < n:  # Ensure indices are valid
                            matrix_data[i, j] = round(data['distance'], self.settings["DECIMAL_PRECISION"])
                            matrix_data[j, i] = round(data['distance'], self.settings["DECIMAL_PRECISION"])  # Symmetric
                
                # Create DataFrame with TPS names
                matrix_df = pd.DataFrame(matrix_data, index=tps_names, columns=tps_names)
                
                # Try to export with openpyxl
                try:
                    # Write multiple sheets to Excel
                    with pd.ExcelWriter(filename, engine=excel_engine) as writer:
                        export_df.to_excel(writer, sheet_name='Distance_Pairs', index=False)
                        matrix_df.to_excel(writer, sheet_name='Distance_Matrix')
                        
                        # Also export the original TPS data
                        df.to_excel(writer, sheet_name='TPS_Data', index=False)
                        
                        # Export any failures
                        if self.failures:
                            failures_df = pd.DataFrame(self.failures)
                            failures_df.to_excel(writer, sheet_name='Failures', index=False)
                        
                        # Export verification stats
                        if self.settings["USE_VERIFICATION"]:
                            verification_stats_df = pd.DataFrame([self.verification_stats])
                            verification_stats_df.to_excel(writer, sheet_name='Verification_Stats', index=False)
                    
                    print(f"Data jarak jalan diekspor ke {filename}")
                    
                except Exception as excel_err:
                    print(f"Error pada ekspor Excel multi-sheet: {excel_err}")
                    print("Mencoba ekspor ke file terpisah...")
                    
                    # Fallback to separate files
                    base_name = os.path.splitext(filename)[0]
                    
                    # Export pairs
                    pairs_file = f"{base_name}_pairs.xlsx"
                    export_df.to_excel(pairs_file, index=False)
                    print(f"Data pasangan jarak diekspor ke {pairs_file}")
                    
                    # Export matrix
                    matrix_file = f"{base_name}_matrix.xlsx"
                    matrix_df.to_excel(matrix_file)
                    print(f"Matriks jarak diekspor ke {matrix_file}")
                    
                    # Export original data
                    data_file = f"{base_name}_tps_data.xlsx"
                    df.to_excel(data_file, index=False)
                    print(f"Data TPS diekspor ke {data_file}")
            else:
                # Just export pairs data
                try:
                    export_df.to_excel(filename, index=False, engine=excel_engine)
                    print(f"Data jarak jalan diekspor ke {filename}")
                except Exception as e:
                    print(f"Error pada ekspor Excel: {e}")
                    # Try with different engine
                    base_name = os.path.splitext(filename)[0]
                    csv_filename = f"{base_name}.csv"
                    export_df.to_csv(csv_filename, index=False)
                    print(f"Data jarak jalan diekspor ke CSV: {csv_filename}")
            
            # Also export to CSV if requested
            if self.settings["EXPORT_CSV"]:
                csv_filename = filename.replace('.xlsx', '.csv')
                export_df.to_csv(csv_filename, index=False)
                print(f"Data jarak jalan juga diekspor ke {csv_filename}")
            
            return True
            
        except Exception as e:
            print(f"Error mengekspor data: {e}")
            
            # Try to export to CSV as last resort
            try:
                csv_filename = filename.replace('.xlsx', '.csv')
                export_df.to_csv(csv_filename, index=False)
                print(f"Fallback: Data diekspor ke CSV {csv_filename}")
                return True
            except:
                return False
    
    def get_statistics(self):
        """
        Get statistics about the distance database.
        
        Returns:
            dict: Statistik database jarak
        """
        if not self.distance_db:
            return {
                'total_pairs': 0,
                'osrm_routes': 0,
                'haversine_fallbacks': 0,
                'manual_validations': 0,
                'verified_routes': 0,
                'early_stopped_verifications': 0,
                'avg_distance': 0,
                'max_distance': 0,
                'min_distance': 0,
                'avg_duration': 0,
                'high_variance_pairs': 0,
                'avg_iterations': 0
            }
        
        stats = {
            'total_pairs': len(self.distance_db),
            'osrm_routes': 0,
            'haversine_fallbacks': 0,
            'manual_validations': 0,
            'verified_routes': 0,
            'early_stopped_verifications': 0,
            'distances': [],
            'durations': [],
            'high_variance_pairs': 0,
            'iterations': []
        }
        
        for key, data in self.distance_db.items():
            source = data.get('source', '')
            
            if source == 'osrm':
                stats['osrm_routes'] += 1
            elif source == 'haversine':
                stats['haversine_fallbacks'] += 1
            elif source == 'manual_validation':
                stats['manual_validations'] += 1
            elif source in ['osrm_verified', 'osrm_verified_median', 'osrm_verified_mean']:
                stats['verified_routes'] += 1
            
            # Track early stopped
            if 'verification' in data:
                if data['verification'].get('early_stopped', False):
                    stats['early_stopped_verifications'] += 1
                
                # Track high variance
                if data['verification'].get('variance', 0) > self.settings["VARIANCE_THRESHOLD"]:
                    stats['high_variance_pairs'] += 1
                
                # Track iterations
                if 'iterations' in data['verification']:
                    stats['iterations'].append(data['verification']['iterations'])
            
            if 'distance' in data:
                stats['distances'].append(data['distance'])
            
            if 'duration' in data:
                stats['durations'].append(data['duration'])
        
        if stats['distances']:
            stats['avg_distance'] = sum(stats['distances']) / len(stats['distances'])
            stats['max_distance'] = max(stats['distances'])
            stats['min_distance'] = min(stats['distances'])
        else:
            stats['avg_distance'] = 0
            stats['max_distance'] = 0
            stats['min_distance'] = 0
        
        if stats['durations']:
            stats['avg_duration'] = sum(stats['durations']) / len(stats['durations'])
        else:
            stats['avg_duration'] = 0
            
        if stats['iterations']:
            stats['avg_iterations'] = sum(stats['iterations']) / len(stats['iterations'])
        else:
            stats['avg_iterations'] = 0
        
        return stats

    def find_inconsistent_pairs(self, threshold=1.0):
        """
        Mencari pasangan TPS dengan jarak yang tidak konsisten (variasi tinggi).
        
        Args:
            threshold: Ambang batas perbedaan jarak (km)
            
        Returns:
            list: Daftar pasangan TPS dengan jarak tidak konsisten
        """
        inconsistent_pairs = []
        
        for key, data in self.distance_db.items():
            if 'verification' in data and data['verification'].get('variance', 0) > threshold:
                tps1_name = self.get_tps_name(data['tps1_id'])
                tps2_name = self.get_tps_name(data['tps2_id'])
                
                inconsistent_pairs.append({
                    'tps1_id': data['tps1_id'],
                    'tps1_name': tps1_name,
                    'tps2_id': data['tps2_id'],
                    'tps2_name': tps2_name,
                    'distance': data['distance'],
                    'variance': data['verification'].get('variance', 0),
                    'measurements': data['verification'].get('distances', [])
                })
        
        # Sort by variance (highest first)
        inconsistent_pairs.sort(key=lambda x: x['variance'], reverse=True)
        
        return inconsistent_pairs
    
    def add_manual_validation_interactive(self):
        """
        Menambahkan validasi manual secara interaktif untuk pasangan TPS.
        
        Returns:
            bool: True jika berhasil, False jika gagal
        """
        if self.original_tps_data is None:
            print("Data TPS tidak tersedia.")
            return False
        
        print("\n=== TAMBAHKAN VALIDASI MANUAL JARAK ===")
        
        # Tampilkan daftar TPS
        print("\nDaftar TPS:")
        for i, row in self.original_tps_data.iterrows():
            print(f"{i}: {row['nama']}")
        
        try:
            # Minta input dari pengguna
            tps1_id = int(input("\nMasukkan ID TPS pertama: "))
            tps2_id = int(input("Masukkan ID TPS kedua: "))
            
            # Tampilkan jarak saat ini jika ada
            tps_key = self.get_distance_key(tps1_id, tps2_id)
            current_distance = None
            
            if tps_key in self.distance_db:
                current_distance = self.distance_db[tps_key]['distance']
                print(f"\nJarak saat ini: {current_distance:.3f} km")
                
                if 'verification' in self.distance_db[tps_key]:
                    verification = self.distance_db[tps_key]['verification']
                    if 'distances' in verification:
                        print("Hasil pengukuran:")
                        for i, d in enumerate(verification['distances']):
                            print(f"  Iterasi {i+1}: {d:.3f} km")
            
            # Minta jarak yang valid
            known_distance = float(input("\nMasukkan jarak yang valid (km): "))
            
            # Konfirmasi
            tps1_name = self.get_tps_name(tps1_id)
            tps2_name = self.get_tps_name(tps2_id)
            
            confirmation = input(f"\nKonfirmasi validasi jarak dari {tps1_name} ke {tps2_name} = {known_distance} km? (y/n): ")
            
            if confirmation.lower() == 'y':
                self.add_manual_validation(tps1_id, tps2_id, known_distance)
                return True
            else:
                print("Validasi dibatalkan.")
                return False
            
        except Exception as e:
            print(f"Error: {e}")
            return False
    
    def validate_most_inconsistent(self, n=5):
        """
        Menampilkan dan meminta validasi untuk pasangan TPS yang paling tidak konsisten.
        
        Args:
            n: Jumlah pasangan yang akan ditampilkan
            
        Returns:
            bool: True jika berhasil, False jika gagal
        """
        inconsistent_pairs = self.find_inconsistent_pairs()
        
        if not inconsistent_pairs:
            print("Tidak ditemukan pasangan TPS dengan jarak tidak konsisten.")
            return False
        
        print("\n=== PASANGAN TPS DENGAN JARAK TIDAK KONSISTEN ===")
        
        for i, pair in enumerate(inconsistent_pairs[:n]):
            print(f"\n{i+1}. {pair['tps1_name']} - {pair['tps2_name']}")
            print(f"   Jarak: {pair['distance']:.3f} km")
            print(f"   Variasi: {pair['variance']:.3f} km")
            print(f"   Pengukuran: {[round(d, 3) for d in pair['measurements']]}")
            
            try:
                validate = input("   Validasi pasangan ini? (y/n/skip): ")
                
                if validate.lower() == 'y':
                    known_distance = float(input("   Masukkan jarak yang valid (km): "))
                    self.add_manual_validation(pair['tps1_id'], pair['tps2_id'], known_distance)
                elif validate.lower() == 'skip':
                    break
            except Exception as e:
                print(f"   Error: {e}")
        
        return True
    
    def recheck_selected_pairs(self, force_refresh=False):
        """
        Menghitung ulang pasangan TPS yang dipilih.
        
        Args:
            force_refresh: Paksa ulang meskipun sudah ada di database
            
        Returns:
            bool: True jika berhasil, False jika gagal
        """
        if self.original_tps_data is None:
            print("Data TPS tidak tersedia.")
            return False
        
        print("\n=== HITUNG ULANG PASANGAN TPS ===")
        
        # Tampilkan daftar TPS
        print("\nDaftar TPS:")
        for i, row in self.original_tps_data.iterrows():
            print(f"{i}: {row['nama']}")
        
        try:
            # Minta input dari pengguna
            pairs_str = input("\nMasukkan pasangan TPS yang akan dihitung ulang (format: '0,1 2,3'): ")
            pairs = []
            
            for pair_str in pairs_str.split():
                tps1_id, tps2_id = map(int, pair_str.split(','))
                pairs.append((tps1_id, tps2_id))
            
            # Hitung ulang pasangan
            print(f"\nMenghitung ulang {len(pairs)} pasangan...")
            
            for tps1_id, tps2_id in pairs:
                tps1_name = self.get_tps_name(tps1_id)
                tps2_name = self.get_tps_name(tps2_id)
                
                print(f"Menghitung ulang: {tps1_name} - {tps2_name}")
                
                lat1, lon1 = self.original_tps_data.iloc[tps1_id]['latitude'], self.original_tps_data.iloc[tps1_id]['longitude']
                lat2, lon2 = self.original_tps_data.iloc[tps2_id]['latitude'], self.original_tps_data.iloc[tps2_id]['longitude']
                
                # Get road distance with force_refresh
                self.get_road_distance(
                    tps1_id, tps2_id, lat1, lon1, lat2, lon2, 
                    vehicle_type=self.settings["VEHICLE_TYPE"],
                    force_refresh=True
                )
            
            # Save database
            self.save_database()
            return True
            
        except Exception as e:
            print(f"Error: {e}")
            return False

# ==========================================================
# FUNGSI UNTUK KONVERSI DATABASE PKL KE EXCEL
# ==========================================================

def convert_pkl_to_excel(pkl_file, output_file=None, tps_data_file=None):
    """
    Konversi database pkl yang sudah ada ke Excel.
    
    Args:
        pkl_file: Path ke file pkl
        output_file: Path ke file output Excel
        tps_data_file: Path ke file data TPS
    """
    print(f"Konversi database {pkl_file} ke Excel...")
    
    # Load database
    try:
        with open(pkl_file, 'rb') as f:
            distance_db = pickle.load(f)
        print(f"Database dimuat: {len(distance_db)} pasangan")
    except Exception as e:
        print(f"Error memuat database: {e}")
        return False
    
    # Load TPS data if provided
    df_tps = None
    if tps_data_file:
        try:
            if tps_data_file.endswith('.csv'):
                df_tps = pd.read_csv(tps_data_file)
            elif tps_data_file.endswith(('.xlsx', '.xls')):
                df_tps = pd.read_excel(tps_data_file)
            print(f"Data TPS dimuat: {len(df_tps)} TPS")
        except Exception as e:
            print(f"Error memuat data TPS: {e}")
    else:
        # Try to load TPS data from our function
        try:
            df_tps = input_data()
            print(f"Data TPS dimuat dari fungsi input_data(): {len(df_tps)} TPS")
        except:
            print("Tidak bisa memuat data TPS")
    
    # Set default output file
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"tps_distances_converted_{timestamp}.xlsx"
    
    # Create a temporary collector object to use its export function
    collector = TPSDistanceCollector(get_settings())
    collector.distance_db = distance_db
    
    if df_tps is not None:
        collector.original_tps_data = df_tps
    
    # Export to Excel
    success = collector.export_to_excel(df_tps if df_tps is not None else pd.DataFrame(), output_file)
    
    if success:
        print(f"Konversi berhasil! File Excel dibuat: {output_file}")
    else:
        print("Konversi gagal!")
    
    return success

# ==========================================================
# MENU INTERAKTIF
# ==========================================================

def interactive_menu():
    """
    Menampilkan menu interaktif untuk program.
    """
    settings = get_settings()
    collector = TPSDistanceCollector(settings)
    
    # Load data TPS
    df = input_data()
    print(f"Data loaded: {len(df)} TPS")
    collector.original_tps_data = df
    
    while True:
        print("\n========== MENU TPS DISTANCE COLLECTOR ==========")
        print("1. Kumpulkan Semua Jarak")
        print("2. Tampilkan Statistik")
        print("3. Ekspor ke Excel")
        print("4. Tambah Validasi Manual")
        print("5. Validasi Pasangan Tidak Konsisten")
        print("6. Hitung Ulang Pasangan Tertentu")
        print("7. Konversi Database PKL ke Excel")
        print("8. Keluar")
        
        choice = input("\nPilih menu: ")
        
        if choice == '1':
            # Kumpulkan semua jarak
            num_threads = int(input("Jumlah thread (1-10): ") or settings["NUM_THREADS"])
            num_threads = max(1, min(10, num_threads))
            
            start_time = time.time()
            collector.collect_all_distances(df, vehicle_type=settings["VEHICLE_TYPE"], num_threads=num_threads)
            elapsed_time = time.time() - start_time
            print(f"\nProses pengumpulan jarak selesai dalam {elapsed_time:.2f} detik")
            
        elif choice == '2':
            # Tampilkan statistik
            stats = collector.get_statistics()
            print("\nStatistik Database Jarak Jalan:")
            print(f"Total pasangan TPS: {stats['total_pairs']}")
            print(f"Rute OSRM: {stats['osrm_routes']} ({stats['osrm_routes']/max(1, stats['total_pairs'])*100:.1f}%)")
            print(f"Rute OSRM terverifikasi: {stats['verified_routes']} ({stats['verified_routes']/max(1, stats['total_pairs'])*100:.1f}%)")
            print(f"Fallback ke Haversine: {stats['haversine_fallbacks']} ({stats['haversine_fallbacks']/max(1, stats['total_pairs'])*100:.1f}%)")
            print(f"Validasi manual: {stats['manual_validations']} ({stats['manual_validations']/max(1, stats['total_pairs'])*100:.1f}%)")
            print(f"Pasangan dengan variasi tinggi: {stats['high_variance_pairs']} ({stats['high_variance_pairs']/max(1, stats['total_pairs'])*100:.1f}%)")
            print(f"Jarak rata-rata: {stats['avg_distance']:.2f} km")
            print(f"Jarak minimum: {stats['min_distance']:.2f} km")
            print(f"Jarak maksimum: {stats['max_distance']:.2f} km")
            print(f"Durasi rata-rata: {stats['avg_duration']:.2f} menit")
            
        elif choice == '3':
            # Ekspor ke Excel
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            default_filename = os.path.join(settings["DATABASE_DIR"], f"tps_distances_{timestamp}.xlsx")
            filename = input(f"Nama file Excel [{default_filename}]: ") or default_filename
            
            start_time = time.time()
            collector.export_to_excel(df, filename)
            elapsed_time = time.time() - start_time
            print(f"Ekspor selesai dalam {elapsed_time:.2f} detik")
            
        elif choice == '4':
            # Tambah validasi manual
            collector.add_manual_validation_interactive()
            
        elif choice == '5':
            # Validasi pasangan tidak konsisten
            n = int(input("Jumlah pasangan yang akan ditampilkan: ") or 5)
            collector.validate_most_inconsistent(n)
            
        elif choice == '6':
            # Hitung ulang pasangan tertentu
            collector.recheck_selected_pairs(force_refresh=True)
            
        elif choice == '7':
            # Konversi database PKL ke Excel
            pkl_file = input(f"Path ke file PKL [{collector.db_file}]: ") or collector.db_file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            default_output = f"tps_distances_converted_{timestamp}.xlsx"
            output_file = input(f"Path ke file output Excel [{default_output}]: ") or default_output
            
            convert_pkl_to_excel(pkl_file, output_file)
            
        elif choice == '8':
            # Keluar
            print("Terima kasih telah menggunakan TPS Distance Collector!")
            break
            
        else:
            print("Pilihan tidak valid. Silakan coba lagi.")

# ==========================================================
# FUNGSI UTAMA
# ==========================================================

def main():
    """Fungsi utama program."""
    print("\n======================================================")
    print("    TPS DISTANCE COLLECTOR - PENGUMPUL JARAK JALAN TPS")
    print("    VERSI DENGAN VERIFIKASI BERGANDA")
    print("======================================================\n")
    
    # Check for command line arguments
    import sys
    if len(sys.argv) > 1:
        if sys.argv[1] == "--convert":
            # Conversion mode - convert existing pkl to Excel
            pkl_file = "tps_distances/tps_distances.pkl"
            if len(sys.argv) > 2:
                pkl_file = sys.argv[2]
            
            output_file = None
            if len(sys.argv) > 3:
                output_file = sys.argv[3]
            
            convert_pkl_to_excel(pkl_file, output_file)
            return
        elif sys.argv[1] == "--interactive":
            # Interactive mode
            interactive_menu()
            return
    
    # Normal mode - run the collector
    
    # Load settings
    settings = get_settings()
    
    # Load data TPS
    df = input_data()
    print(f"Data loaded: {len(df)} TPS")
    
    # Remove duplicates based on coordinates (with small tolerance)
    df['coord_id'] = df.apply(
        lambda row: f"{round(row['latitude'], 5)}_{round(row['longitude'], 5)}", axis=1
    )
    df_dedup = df.drop_duplicates(subset=['coord_id'])
    
    if len(df) != len(df_dedup):
        print(f"Removed {len(df) - len(df_dedup)} duplicate locations")
        df = df_dedup.drop(columns=['coord_id'])
    else:
        df = df.drop(columns=['coord_id'])
    
    # Create TPS Distance Collector
    collector = TPSDistanceCollector(settings)
    
    # Start time for whole process
    start_time = time.time()
    
    # Collect distances between all TPS pairs
    collector.collect_all_distances(
        df, 
        vehicle_type=settings["VEHICLE_TYPE"],
        num_threads=settings["NUM_THREADS"]
    )
    
    # End time
    elapsed_time = time.time() - start_time
    print(f"\nProses pengumpulan jarak selesai dalam {elapsed_time:.2f} detik")
    
    # Get statistics
    stats = collector.get_statistics()
    print("\nStatistik Database Jarak Jalan:")
    print(f"Total pasangan TPS: {stats['total_pairs']}")
    print(f"Rute OSRM berhasil: {stats['osrm_routes']} ({stats['osrm_routes']/stats['total_pairs']*100:.1f}%)")
    print(f"Rute OSRM terverifikasi: {stats['verified_routes']} ({stats['verified_routes']/stats['total_pairs']*100:.1f}%)")
    print(f"Early stopping: {stats['early_stopped_verifications']} ({stats['early_stopped_verifications']/max(1, stats['verified_routes'])*100:.1f}% dari terverifikasi)")
    print(f"Rata-rata iterasi: {stats['avg_iterations']:.1f}")
    print(f"Fallback ke Haversine: {stats['haversine_fallbacks']} ({stats['haversine_fallbacks']/stats['total_pairs']*100:.1f}%)")
    print(f"Validasi manual: {stats['manual_validations']} ({stats['manual_validations']/stats['total_pairs']*100:.1f}%)")
    print(f"Pasangan dengan variasi tinggi: {stats['high_variance_pairs']} ({stats['high_variance_pairs']/stats['total_pairs']*100:.1f}%)")
    print(f"Jarak rata-rata: {stats['avg_distance']:.2f} km")
    print(f"Jarak minimum: {stats['min_distance']:.2f} km")
    print(f"Jarak maksimum: {stats['max_distance']:.2f} km")
    print(f"Durasi rata-rata: {stats['avg_duration']:.2f} menit")
    
    # Export to Excel
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    excel_file = os.path.join(settings["DATABASE_DIR"], f"tps_distances_{timestamp}.xlsx")
    collector.export_to_excel(df, excel_file)
    
    print("\nProses selesai! Data jarak jalan antar TPS telah disimpan dan diekspor.")
    print("Anda dapat menjalankan program dalam mode interaktif dengan:")
    print("python tps_distance_collector.py --interactive")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nProgram dihentikan oleh pengguna.")
        print("Data yang sudah dihitung tetap tersimpan dan dapat digunakan nanti.")
    except Exception as e:
        print(f"\n\nTerjadi kesalahan: {e}")
        print("Data yang sudah dihitung tetap tersimpan dan dapat digunakan nanti.")