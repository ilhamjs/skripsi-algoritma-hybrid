import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os
import random
import time
import math
from collections import defaultdict
import threading
import folium
from folium.plugins import MarkerCluster
import webbrowser
import tempfile
import traceback
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import pickle
from sklearn.cluster import KMeans

class TPSClusteringApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Aplikasi Clustering TPS dengan GA-VNS dan Optimasi Rute ACO")
        self.root.geometry("1400x900")  # Window lebih besar
        self.root.minsize(1200, 800)    # Minimum size lebih besar juga
        
        self.file_path = None
        self.results = None
        self.data_info = None
        self.distance_matrix = None
        self.route_figures = []  # For storing route figures
        
        # Multi-start related variables
        self.multi_start_results = None
        self.start_time = None
        self.timer_id = None
        
        # Buat frame utama dengan 3 kolom untuk parameter
        main_frame = ttk.Frame(root)
        main_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        # === ATUR GRID UNTUK 3 KOLOM ===
        # Kolom 0: Parameter Clustering
        # Kolom 1: Multi-Start Parameters & Armada
        # Kolom 2: Upload File
        
        # === KOLOM KIRI: PARAMETER CLUSTERING ===
        param_frame = ttk.LabelFrame(main_frame, text="Parameter Clustering")
        param_frame.grid(row=0, column=0, padx=5, pady=5, sticky="nw")
        
        # Min capacity
        ttk.Label(param_frame, text="Kapasitas Minimum Cluster (m³):").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.min_capacity_var = tk.DoubleVar(value=10.0)
        ttk.Entry(param_frame, textvariable=self.min_capacity_var, width=10).grid(row=0, column=1, padx=5, pady=5, sticky="w")
        
        # Max capacity
        ttk.Label(param_frame, text="Kapasitas Maksimum Cluster (m³):").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.max_capacity_var = tk.DoubleVar(value=13.0)
        ttk.Entry(param_frame, textvariable=self.max_capacity_var, width=10).grid(row=1, column=1, padx=5, pady=5, sticky="w")
        
        # Jarak maksimal (opsional, bukan batasan keras)
        ttk.Label(param_frame, text="Penalti Jarak (bobot):").grid(row=2, column=0, padx=5, pady=5, sticky="w")
        self.distance_penalty_var = tk.DoubleVar(value=1.0)
        ttk.Entry(param_frame, textvariable=self.distance_penalty_var, width=10).grid(row=2, column=1, padx=5, pady=5, sticky="w")
        
        # GA-VNS Parameters
        ttk.Label(param_frame, text="Ukuran Populasi GA:").grid(row=0, column=2, padx=5, pady=5, sticky="w")
        self.population_size_var = tk.IntVar(value=75)
        ttk.Entry(param_frame, textvariable=self.population_size_var, width=10).grid(row=0, column=3, padx=5, pady=5, sticky="w")
        
        ttk.Label(param_frame, text="Jumlah Iterasi:").grid(row=1, column=2, padx=5, pady=5, sticky="w")
        self.max_iterations_var = tk.IntVar(value=200)
        ttk.Entry(param_frame, textvariable=self.max_iterations_var, width=10).grid(row=1, column=3, padx=5, pady=5, sticky="w")
        
        ttk.Label(param_frame, text="Mutation Rate:").grid(row=2, column=2, padx=5, pady=5, sticky="w")
        self.mutation_rate_var = tk.DoubleVar(value=0.2)
        ttk.Entry(param_frame, textvariable=self.mutation_rate_var, width=10).grid(row=2, column=3, padx=5, pady=5, sticky="w")
        
        ttk.Label(param_frame, text="Crossover Rate:").grid(row=3, column=2, padx=5, pady=5, sticky="w")
        self.crossover_rate_var = tk.DoubleVar(value=0.8)
        ttk.Entry(param_frame, textvariable=self.crossover_rate_var, width=10).grid(row=3, column=3, padx=5, pady=5, sticky="w")
        
        # Optimasi rute
        self.optimize_routes_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(param_frame, text="Optimasi Rute (ACO)", variable=self.optimize_routes_var).grid(row=3, column=0, padx=5, pady=5, sticky="w")
        
        # ACO Parameters
        ttk.Label(param_frame, text="ACO Iterations:").grid(row=4, column=0, padx=5, pady=5, sticky="w")
        self.aco_iterations_var = tk.IntVar(value=50)
        ttk.Entry(param_frame, textvariable=self.aco_iterations_var, width=10).grid(row=4, column=1, padx=5, pady=5, sticky="w")
        
        ttk.Label(param_frame, text="ACO Ants:").grid(row=4, column=2, padx=5, pady=5, sticky="w")
        self.aco_ants_var = tk.IntVar(value=10)
        ttk.Entry(param_frame, textvariable=self.aco_ants_var, width=10).grid(row=4, column=3, padx=5, pady=5, sticky="w")
        
        # Titik awal dan akhir rute
        route_endpoints_frame = ttk.LabelFrame(param_frame, text="Titik Awal & Akhir Rute")
        route_endpoints_frame.grid(row=5, column=0, columnspan=4, padx=5, pady=5, sticky="we")

        # Titik awal (Garasi)
        ttk.Label(route_endpoints_frame, text="Titik Awal (Garasi):").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.start_point_var = tk.StringVar(value="Garasi")
        ttk.Entry(route_endpoints_frame, textvariable=self.start_point_var, width=20).grid(row=0, column=1, padx=5, pady=5, sticky="w")

        # Titik akhir (TPA)
        ttk.Label(route_endpoints_frame, text="Titik Akhir (TPA):").grid(row=0, column=2, padx=5, pady=5, sticky="w")
        self.end_point_var = tk.StringVar(value="TPA Troketon")
        ttk.Entry(route_endpoints_frame, textvariable=self.end_point_var, width=20).grid(row=0, column=3, padx=5, pady=5, sticky="w")
        
        # === KOLOM TENGAH: MULTI-START & ARMADA PARAMETERS ===
        middle_frame = ttk.Frame(main_frame)
        middle_frame.grid(row=0, column=1, padx=5, pady=5, sticky="n")
        
        # Multi-Start Parameters
        multi_frame = ttk.LabelFrame(middle_frame, text="Multi-Start Parameters")
        multi_frame.pack(fill="x", padx=5, pady=5)
        
        # Jumlah runs
        ttk.Label(multi_frame, text="Jumlah Run:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.num_runs_var = tk.IntVar(value=10)
        ttk.Entry(multi_frame, textvariable=self.num_runs_var, width=10).grid(row=0, column=1, padx=5, pady=5, sticky="w")
        
        # Parallel processing
        ttk.Label(multi_frame, text="Parallel Processing:").grid(row=0, column=2, padx=5, pady=5, sticky="w")
        self.parallel_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(multi_frame, variable=self.parallel_var).grid(row=0, column=3, padx=5, pady=5, sticky="w")
        
        # Number of workers
        ttk.Label(multi_frame, text="Jumlah Worker:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.num_workers_var = tk.IntVar(value=8)
        ttk.Entry(multi_frame, textvariable=self.num_workers_var, width=10).grid(row=1, column=1, padx=5, pady=5, sticky="w")
        
        # Adaptive stopping
        ttk.Label(multi_frame, text="Adaptive Stopping:").grid(row=1, column=2, padx=5, pady=5, sticky="w")
        self.adaptive_stop_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(multi_frame, variable=self.adaptive_stop_var).grid(row=1, column=3, padx=5, pady=5, sticky="w")
        
        # Max no improvement
        ttk.Label(multi_frame, text="Max Tanpa Perbaikan:").grid(row=2, column=0, padx=5, pady=5, sticky="w")
        self.max_no_improvement_var = tk.IntVar(value=3)
        ttk.Entry(multi_frame, textvariable=self.max_no_improvement_var, width=10).grid(row=2, column=1, padx=5, pady=5, sticky="w")
        
        # Parameter Armada
        fleet_frame = ttk.LabelFrame(middle_frame, text="Parameter Armada Pengangkutan")
        fleet_frame.pack(fill="x", padx=5, pady=5)

        # Jumlah armada/truk
        ttk.Label(fleet_frame, text="Jumlah Armada Pengangkutan:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.num_trucks_var = tk.IntVar(value=6)
        ttk.Entry(fleet_frame, textvariable=self.num_trucks_var, width=10).grid(row=0, column=1, padx=5, pady=5, sticky="w")
        
        # === KOLOM KANAN: FILE UPLOAD ===
        file_frame = ttk.LabelFrame(main_frame, text="Upload File Excel")
        file_frame.grid(row=0, column=2, padx=5, pady=5, sticky="ne")
        
        ttk.Label(file_frame, text="Format file Excel:").grid(row=0, column=0, columnspan=3, padx=5, pady=5, sticky="w")
        ttk.Label(file_frame, text="Sheet 1 (Data) = Data TPS (nama, lat, long, volume)").grid(row=1, column=0, columnspan=3, padx=5, pady=2, sticky="w")
        ttk.Label(file_frame, text="Sheet 2 (Matrix) = Matriks Jarak").grid(row=2, column=0, columnspan=3, padx=5, pady=2, sticky="w")
        
        # File path display
        self.file_path_var = tk.StringVar()
        ttk.Entry(file_frame, textvariable=self.file_path_var, width=40, state="readonly").grid(row=3, column=0, padx=5, pady=10, sticky="w")
        
        # Browse button
        ttk.Button(file_frame, text="Browse", command=self.browse_file).grid(row=3, column=1, padx=5, pady=10)
        
        # Process button
        ttk.Button(file_frame, text="Proses Clustering", command=self.process_clustering).grid(row=4, column=0, columnspan=2, padx=5, pady=5)
        
        # Use fixed endpoints checkbox
        self.use_fixed_endpoints_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(file_frame, text="Gunakan titik awal dan akhir tetap", 
                        variable=self.use_fixed_endpoints_var).grid(row=5, column=0, columnspan=2, padx=5, pady=5, sticky="w")

        # Debug Mode checkbox
        self.debug_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(file_frame, text="Debug Mode", variable=self.debug_var).grid(row=6, column=0, columnspan=2, padx=5, pady=5, sticky="w")
        
        # === HASIL CLUSTERING (BARIS BAWAH) ===
        # Results frame - sekarang menempati seluruh lebar 3 kolom
        self.results_frame = ttk.LabelFrame(main_frame, text="Hasil Clustering")
        self.results_frame.grid(row=1, column=0, columnspan=3, padx=5, pady=5, sticky="nsew")
        
        # Konfigurasikan grid weight untuk memastikan results_frame mendapat ruang maksimal
        main_frame.grid_rowconfigure(1, weight=1)
        main_frame.grid_columnconfigure(0, weight=1)
        main_frame.grid_columnconfigure(1, weight=1)
        main_frame.grid_columnconfigure(2, weight=1)
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.results_frame)
        self.notebook.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Summary tab
        self.summary_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.summary_tab, text="Ringkasan")
        
        # Visualization tab
        self.viz_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.viz_tab, text="Visualisasi")
        
        # Map visualization tab
        self.map_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.map_tab, text="Visualisasi Peta")
        
        # Route visualization tab
        self.route_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.route_tab, text="Visualisasi Rute")
        
        # Details tab
        self.details_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.details_tab, text="Detail Cluster")
        
        # Data View tab
        self.data_view_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.data_view_tab, text="Lihat Data")
        
        # Matrix View tab
        self.matrix_view_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.matrix_view_tab, text="Lihat Matriks")
        
        # Debug tab
        self.debug_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.debug_tab, text="Debug")
        
        # Progress tab (for tracking algorithm progress)
        self.progress_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.progress_tab, text="Progress")
        
        # Multi-Start Results tab
        self.multi_start_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.multi_start_tab, text="Multi-Start Results")

        # Titik Tengah tab
        self.centroid_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.centroid_tab, text="Titik Tengah")
        
        # Status bar with progress
        status_frame = ttk.Frame(root)
        status_frame.pack(side="bottom", fill="x")
        
        self.status_var = tk.StringVar()
        self.status_var.set("Siap. Silakan pilih file Excel.")
        self.status_label = ttk.Label(status_frame, textvariable=self.status_var, relief="sunken", anchor="w")
        self.status_label.pack(side="left", fill="x", expand=True)
        
        # Progress bar
        self.progress_var = tk.DoubleVar(value=0)
        self.progress_bar = ttk.Progressbar(status_frame, variable=self.progress_var, maximum=100, length=200)
        self.progress_bar.pack(side="right", padx=10, fill="x", expand=False)
        
        # Progress percentage
        self.progress_percent_var = tk.StringVar(value="0%")
        self.progress_percent_label = ttk.Label(status_frame, textvariable=self.progress_percent_var, width=5)
        self.progress_percent_label.pack(side="right", padx=5)


    def create_scrollable_frame(self, parent_tab):
        """Membuat frame dengan scrollbar vertikal dan horizontal"""
        # Buat frame utama
        main_frame = ttk.Frame(parent_tab)
        main_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Buat canvas untuk scrolling
        canvas = tk.Canvas(main_frame)
        v_scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
        h_scrollbar = ttk.Scrollbar(main_frame, orient="horizontal", command=canvas.xview)
        
        # Buat frame dalam canvas untuk konten
        content_frame = ttk.Frame(canvas)
        content_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        
        # Buat window dalam canvas
        canvas_window = canvas.create_window((0, 0), window=content_frame, anchor="nw")
        
        # Configure canvas resize
        canvas.bind("<Configure>", lambda e: canvas.itemconfig(canvas_window, width=e.width))
        
        # Pack scrollbar dan canvas
        v_scrollbar.pack(side="right", fill="y")
        h_scrollbar.pack(side="bottom", fill="x")
        canvas.pack(side="left", fill="both", expand=True)
        
        # Configure scrollbar command
        canvas.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
        
        # Tambahkan dukungan untuk scrolling dengan mouse wheel
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)
        
        return content_frame

    def browse_file(self):
        """Open file dialog to select Excel file"""
        file_path = filedialog.askopenfilename(
            title="Pilih File Excel",
            filetypes=[("Excel files", "*.xlsx *.xls")],
        )
        if file_path:
            self.file_path = file_path
            self.file_path_var.set(file_path)
            self.status_var.set(f"File dipilih: {os.path.basename(file_path)}")
    
    def update_progress(self, percentage, message=None):
        """Update the progress bar and status message"""
        self.progress_var.set(percentage)
        self.progress_percent_var.set(f"{int(percentage)}%")
        if message:
            self.status_var.set(message)
        self.root.update_idletasks()
    
    def update_timer_display(self, start_time):
        """Update timer for display"""
        if self.start_time is None:
            self.start_time = start_time
            
        elapsed = time.time() - start_time
        hours, remainder = divmod(elapsed, 3600)
        minutes, seconds = divmod(remainder, 60)
        time_str = f"Waktu berjalan: {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"
        
        # Update status bar
        self.status_var.set(f"Multi-start berjalan... {time_str}")
            
        # Force update
        self.root.update_idletasks()
        
        # Update every second
        self.timer_id = self.root.after(1000, lambda: self.update_timer_display(start_time))
    
    def optimize_route_aco(self, cluster, distance_matrix, start_idx=None, end_idx=None, num_ants=10, num_iterations=50, alpha=1.0, beta=5.0, update_progress=None):
        """
        Optimize route using Ant Colony Optimization with optional fixed start and end points
        """
        # Handle special cases of empty or very small clusters
        if len(cluster) == 0:
            if start_idx is not None and end_idx is not None:
                # Jika tidak ada TPS, cukup rute dari Garasi ke TPA
                return [start_idx, end_idx], distance_matrix[start_idx][end_idx]
            return cluster, 0
        elif len(cluster) == 1:
            if start_idx is not None and end_idx is not None:
                # Jika hanya ada 1 TPS, buat rute Garasi -> TPS -> TPA
                tps_idx = cluster[0]
                route = [start_idx, tps_idx, end_idx]
                distance = distance_matrix[start_idx][tps_idx] + distance_matrix[tps_idx][end_idx]
                return route, distance
            return cluster, 0
        elif len(cluster) == 2:
            if start_idx is not None and end_idx is not None:
                # Jika ada 2 TPS, coba 2 kemungkinan urutan dan pilih yang terbaik
                tps1, tps2 = cluster
                route1 = [start_idx, tps1, tps2, end_idx]
                dist1 = (distance_matrix[start_idx][tps1] + 
                        distance_matrix[tps1][tps2] + 
                        distance_matrix[tps2][end_idx])
                        
                route2 = [start_idx, tps2, tps1, end_idx]
                dist2 = (distance_matrix[start_idx][tps2] + 
                        distance_matrix[tps2][tps1] + 
                        distance_matrix[tps1][end_idx])
                        
                return route1 if dist1 <= dist2 else route2, min(dist1, dist2)
            else:
                # Tanpa titik awal/akhir, hanya optimasi 2 TPS
                dist = distance_matrix[cluster[0]][cluster[1]]
                return cluster, dist
        
        # Flag untuk menandakan apakah menggunakan titik awal/akhir
        use_fixed_points = start_idx is not None and end_idx is not None
        
        # Susun node untuk optimasi
        if use_fixed_points:
            # Jika menggunakan titik awal/akhir, tambahkan ke dalam nodes
            all_nodes = [start_idx] + cluster + [end_idx]
        else:
            # Jika tidak, hanya gunakan TPS saja
            all_nodes = cluster.copy()
        
        # Ekstrak submatriks jarak untuk nodes yang relevan
        route_size = len(all_nodes)
        sub_matrix = np.zeros((route_size, route_size))
        for i in range(route_size):
            for j in range(route_size):
                sub_matrix[i][j] = distance_matrix[all_nodes[i]][all_nodes[j]]
        
        # Parameter ACO
        evaporation_rate = 0.1
        Q = 100  # Pheromone deposit factor
        
        # Inisialisasi matriks feromon
        pheromone = np.ones((route_size, route_size))
        
        # Variabel untuk melacak solusi terbaik
        best_route_indices = list(range(route_size))
        best_distance = float('inf')
        
        # Progress tracking
        progress_increment = 100.0 / num_iterations if num_iterations > 0 else 0
        
        # Loop utama ACO
        for iteration in range(num_iterations):
            all_routes = []
            all_distances = []
            
            # Setiap semut membangun solusi
            for ant in range(num_ants):
                if use_fixed_points:
                    # Jika gunakan fixed points, selalu mulai dari titik awal (Garasi)
                    current_idx = 0  # Indeks 0 adalah start_idx di all_nodes
                    # TPS (bukan Garasi/TPA) yang belum dikunjungi
                    unvisited = list(range(1, route_size-1))  # Skip indeks 0 (start) dan terakhir (end)
                else:
                    # Tanpa fixed points, mulai dari TPS acak
                    current_idx = random.randint(0, route_size-1)
                    unvisited = list(range(route_size))
                    unvisited.remove(current_idx)
                
                # Membangun rute
                route = [current_idx]
                total_distance = 0
                
                # Pilih TPS berikutnya sampai semua dikunjungi
                while unvisited:
                    # Hitung probabilitas transisi
                    probabilities = []
                    for next_idx in unvisited:
                        # Level feromon
                        tau = pheromone[current_idx][next_idx] ** alpha
                        # Informasi heuristik (inverse jarak)
                        dist_val = sub_matrix[current_idx][next_idx]
                        eta = (1.0 / dist_val) ** beta if dist_val > 0 else 0
                        probabilities.append(tau * eta)
                    
                    # Normalisasi probabilitas
                    total = sum(probabilities)
                    if total > 0:
                        probabilities = [p/total for p in probabilities]
                    else:
                        probabilities = [1.0/len(unvisited)] * len(unvisited)
                    
                    # Pilih node berikutnya
                    if len(unvisited) > 0:
                        next_idx = np.random.choice(unvisited, p=probabilities)
                        
                        # Tambahkan ke rute
                        route.append(next_idx)
                        total_distance += sub_matrix[current_idx][next_idx]
                        current_idx = next_idx
                        unvisited.remove(next_idx)
                
                # Jika menggunakan fixed endpoints, tambahkan titik akhir (TPA)
                if use_fixed_points:
                    # Tambahkan titik akhir (TPA) ke rute
                    route.append(route_size-1)  # Indeks terakhir adalah end_idx
                    total_distance += sub_matrix[current_idx][route_size-1]
                
                # Simpan rute ini
                all_routes.append(route)
                all_distances.append(total_distance)
                
                # Update solusi terbaik
                if total_distance < best_distance:
                    best_route_indices = route.copy()
                    best_distance = total_distance
            
            # Evaporasi feromon
            pheromone *= (1 - evaporation_rate)
            
            # Update feromon berdasarkan rute-rute yang dibangun
            for route, dist in zip(all_routes, all_distances):
                deposit = Q / dist if dist > 0 else 0
                for i in range(len(route)-1):
                    pheromone[route[i]][route[i+1]] += deposit
            
            # Update progress jika callback disediakan
            if update_progress:
                current_progress = (iteration + 1) * progress_increment
                update_progress(current_progress, f"ACO Iterasi {iteration+1}/{num_iterations}")
        
        # Konversi indeks rute ke indeks node asli
        best_route = [all_nodes[i] for i in best_route_indices]
        
        return best_route, best_distance
    
    def ga_vns_clustering(self, distance_matrix, volumes, min_capacity, max_capacity, 
                        population_size=75, max_iterations=200, mutation_rate=0.2, crossover_rate=0.8,
                        distance_penalty=1.0, optimize_routes=True, 
                        aco_ants=10, aco_iterations=50, debug_mode=True):
        """
        Improved GA-VNS hybrid algorithm for TPS clustering with route optimization.
        Genetic Algorithm (GA) is used as the primary search method with Variable Neighborhood
        Search (VNS) applied to refine promising solutions.
        """
        n_points = len(volumes)
        debug_info = []
        
        # Store distance matrix for later use
        self.distance_matrix = distance_matrix
        
        # Setup progress tracking for UI
        progress_frame = ttk.Frame(self.progress_tab)
        progress_frame.pack(fill="x", padx=10, pady=10)
        
        self.progress_label = ttk.Label(progress_frame, text="Inisialisasi...")
        self.progress_label.pack(fill="x", anchor="w")
        
        self.detail_progress_var = tk.DoubleVar(value=0)
        self.detail_progress_bar = ttk.Progressbar(progress_frame, variable=self.detail_progress_var, maximum=100, length=300)
        self.detail_progress_bar.pack(fill="x", pady=5)
        
        progress_text = tk.Text(self.progress_tab, wrap="word")
        progress_text.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Add scrollbar to progress text
        progress_scrollbar = ttk.Scrollbar(progress_text, command=progress_text.yview)
        progress_scrollbar.pack(side="right", fill="y")
        progress_text.config(yscrollcommand=progress_scrollbar.set)
        
        # Function to update detailed progress
        def update_detail_progress(percentage, message=None):
            self.detail_progress_var.set(percentage)
            if message:
                self.progress_label.config(text=message)
            self.root.update_idletasks()
        
        # Explicit check for size match
        if distance_matrix.shape[0] != n_points or distance_matrix.shape[1] != n_points:
            raise ValueError(f"Ukuran matriks jarak ({distance_matrix.shape[0]}x{distance_matrix.shape[1]}) tidak sesuai dengan jumlah TPS ({n_points})")
        
        # Calculate required number of clusters based on total volume and capacity
        total_volume = sum(volumes)
        min_clusters_needed = math.ceil(total_volume / max_capacity)
        expected_clusters = max(min_clusters_needed, 1)
        
        progress_log = f"Total volume: {total_volume:.2f} m³\n"
        progress_log += f"Kapasitas minimum cluster: {min_capacity:.2f} m³\n"
        progress_log += f"Kapasitas maksimum cluster: {max_capacity:.2f} m³\n"
        progress_log += f"Perkiraan jumlah cluster: {expected_clusters}\n"
        progress_log += f"Optimasi rute: {'Aktif' if optimize_routes else 'Nonaktif'}\n"
        progress_log += f"Algoritma rute: ACO\n\n"
        progress_text.insert("end", progress_log)
        self.root.update_idletasks()
        
        # Update progress
        self.update_progress(1, "Memulai clustering...")
        
        # Helper function to check if a cluster is valid
        def is_valid_cluster(cluster, check_min_capacity=True):
            if not cluster:
                return False
                
            # Check capacity constraint
            cluster_volume = sum(volumes[idx] for idx in cluster)
            
            if cluster_volume > max_capacity:
                return False
                
            if check_min_capacity and cluster_volume < min_capacity:
                return False
            
            return True
        
        # Helper function to calculate fitness (lower is better)
        def calculate_fitness(solution):
            # Check if all TPS are assigned once
            all_tps = []
            for cluster in solution:
                all_tps.extend(cluster)
            
            # Penalize missing or duplicate TPS assignments
            missing_tps = set(range(n_points)) - set(all_tps)
            duplicate_tps = len(all_tps) - len(set(all_tps))
            missing_penalty = len(missing_tps) * 1000000
            duplicate_penalty = duplicate_tps * 1000000
            
            # Check capacity constraints
            capacity_violations = 0
            for cluster in solution:
                cluster_volume = sum(volumes[idx] for idx in cluster)
                if cluster_volume < min_capacity:
                    # Penalize under capacity
                    capacity_violations += (min_capacity - cluster_volume) * 10000
                if cluster_volume > max_capacity:
                    # Heavily penalize over capacity
                    capacity_violations += (cluster_volume - max_capacity) * 100000
            
            # Calculate total route distance for each cluster using ACO
            total_route_distance = 0
            
            for cluster in solution:
                if len(cluster) > 1:
                    # Use TSP/route optimization if enabled
                    if optimize_routes:
                        # Simple approximation for fitness calculation 
                        # (full ACO would be too slow for every fitness evaluation)
                        if len(cluster) == 2:
                            cluster_distance = distance_matrix[cluster[0], cluster[1]]
                        else:
                            # Use a greedy nearest neighbor for quick evaluation
                            current = cluster[0]
                            unvisited = set(cluster[1:])
                            dist = 0
                            while unvisited:
                                # Find nearest neighbor
                                next_node = min(unvisited, key=lambda x: distance_matrix[current, x])
                                dist += distance_matrix[current, next_node]
                                current = next_node
                                unvisited.remove(next_node)
                            cluster_distance = dist
                    else:
                        # If not optimizing routes, just sum all intra-cluster distances
                        cluster_distance = 0
                        for i in range(len(cluster)):
                            for j in range(i+1, len(cluster)):
                                cluster_distance += distance_matrix[cluster[i], cluster[j]]
                    
                    total_route_distance += cluster_distance
            
            # Apply distance penalty weight
            weighted_distance = total_route_distance * distance_penalty
            
            # Total fitness is weighted sum of route distance and penalties
            total_fitness = weighted_distance + capacity_violations + missing_penalty + duplicate_penalty
            
            return total_fitness, total_route_distance, capacity_violations, 0, missing_penalty, duplicate_penalty
        
        # Function to create an initial solution with greedy bin packing
        def create_greedy_solution():
            # Sort TPS by volume (descending)
            tps_indices = sorted(range(n_points), key=lambda i: volumes[i], reverse=True)
            
            # Initialize clusters
            solution = []
            remaining_tps = set(tps_indices)
            
            # Process each TPS
            while remaining_tps:
                # Create new cluster
                current_cluster = []
                current_volume = 0
                
                # Start with largest remaining TPS
                sorted_remaining = sorted(remaining_tps, key=lambda i: volumes[i], reverse=True)
                first_tps = sorted_remaining[0]
                current_cluster.append(first_tps)
                current_volume += volumes[first_tps]
                remaining_tps.remove(first_tps)
                
                # Process remaining TPS by distance
                while remaining_tps and current_volume < max_capacity:
                    # Find closest remaining TPS to cluster
                    closest_tps = None
                    min_dist = float('inf')
                    
                    for tps_idx in remaining_tps:
                        # Check if adding this TPS exceeds capacity
                        if current_volume + volumes[tps_idx] > max_capacity:
                            continue
                            
                        # Find minimum distance to any TPS in current cluster
                        for existing_idx in current_cluster:
                            dist = distance_matrix[existing_idx, tps_idx]
                            if dist < min_dist:
                                min_dist = dist
                                closest_tps = tps_idx
                    
                    if closest_tps is not None:
                        # Add closest TPS to cluster
                        current_cluster.append(closest_tps)
                        current_volume += volumes[closest_tps]
                        remaining_tps.remove(closest_tps)
                        
                        # Break if we reached minimum capacity and still have TPS left
                        if current_volume >= min_capacity and len(remaining_tps) > 0:
                            break
                    else:
                        # If we can't find any TPS to add, break
                        break
                
                # Add cluster to solution if not empty
                if current_cluster:
                    solution.append(current_cluster)
                    
                # If we've formed enough clusters but still have TPS, try to add them
                if len(solution) >= expected_clusters and remaining_tps:
                    # Try to add remaining TPS to existing clusters
                    for tps_idx in list(remaining_tps):
                        added = False
                        
                        # Try adding to each cluster
                        for cluster in solution:
                            cluster_volume = sum(volumes[idx] for idx in cluster)
                            
                            if cluster_volume + volumes[tps_idx] <= max_capacity:
                                cluster.append(tps_idx)
                                remaining_tps.remove(tps_idx)
                                added = True
                                break
                        
                        if not added:
                            # If can't add to any existing cluster, create a new one
                            break
                
                # Emergency break to prevent infinite loop
                if not remaining_tps or (len(solution) > 0 and len(current_cluster) == 0):
                    break
                    
            # If we still have remaining TPS, create singleton clusters
            for tps_idx in remaining_tps:
                solution.append([tps_idx])
            
            return solution
        
        # Function to create a random solution
        def create_random_solution():
            # Shuffle TPS indices
            tps_indices = list(range(n_points))
            random.shuffle(tps_indices)
            
            # Initialize solution
            solution = []
            current_cluster = []
            current_volume = 0
            
            # Process each TPS
            for tps_idx in tps_indices:
                # Check if adding to current cluster would exceed capacity
                if current_volume + volumes[tps_idx] <= max_capacity:
                    # Add to current cluster
                    current_cluster.append(tps_idx)
                    current_volume += volumes[tps_idx]
                else:
                    # If we can't add to current cluster, check if current cluster is valid
                    if current_cluster and current_volume >= min_capacity:
                        solution.append(current_cluster)
                    
                    # Start new cluster with this TPS
                    current_cluster = [tps_idx]
                    current_volume = volumes[tps_idx]
            
            # Add last cluster if not empty
            if current_cluster:
                solution.append(current_cluster)
            
            return solution
        
        # Function to create a k-means style solution
        def create_kmeans_solution(k=expected_clusters):
            # Choose k random centroids
            all_indices = list(range(n_points))
            centroids_indices = random.sample(all_indices, k)
            
            # Assign each TPS to nearest centroid
            clusters = [[] for _ in range(k)]
            
            for i in range(n_points):
                # Find closest centroid
                min_dist = float('inf')
                closest_centroid = 0
                
                for c, centroid_idx in enumerate(centroids_indices):
                    dist = distance_matrix[i, centroid_idx]
                    if dist < min_dist:
                        min_dist = dist
                        closest_centroid = c
                
                # Add to cluster
                clusters[closest_centroid].append(i)
            
            # Fix clusters that violate capacity constraints
            valid_clusters = []
            unassigned_tps = []
            
            for cluster in clusters:
                # Check if cluster exceeds capacity
                cluster_volume = sum(volumes[idx] for idx in cluster)
                
                if cluster_volume <= max_capacity:
                    valid_clusters.append(cluster)
                else:
                    # Split cluster if it exceeds capacity
                    # Sort by volume
                    cluster.sort(key=lambda idx: volumes[idx], reverse=True)
                    
                    # Add TPS until capacity limit
                    current_cluster = []
                    current_volume = 0
                    
                    for tps_idx in cluster:
                        if current_volume + volumes[tps_idx] <= max_capacity:
                            current_cluster.append(tps_idx)
                            current_volume += volumes[tps_idx]
                        else:
                            unassigned_tps.append(tps_idx)
                    
                    if current_cluster:
                        valid_clusters.append(current_cluster)
            
            # Handle unassigned TPS
            if unassigned_tps:
                # Try to add to existing clusters
                for tps_idx in list(unassigned_tps):
                    added = False
                    
                    for cluster in valid_clusters:
                        # Check capacity
                        cluster_volume = sum(volumes[idx] for idx in cluster)
                        
                        if cluster_volume + volumes[tps_idx] <= max_capacity:
                            cluster.append(tps_idx)
                            unassigned_tps.remove(tps_idx)
                            added = True
                            break
                    
                    if not added:
                        continue
                
                # Create new clusters for remaining unassigned TPS
                if unassigned_tps:
                    unassigned_tps.sort(key=lambda idx: volumes[idx], reverse=True)
                    
                    current_cluster = []
                    current_volume = 0
                    
                    for tps_idx in unassigned_tps:
                        if current_volume + volumes[tps_idx] <= max_capacity:
                            current_cluster.append(tps_idx)
                            current_volume += volumes[tps_idx]
                        else:
                            # If can't add due to capacity, create a new cluster
                            if current_cluster:
                                valid_clusters.append(current_cluster)
                            
                            current_cluster = [tps_idx]
                            current_volume = volumes[tps_idx]
                    
                    # Add the last cluster
                    if current_cluster:
                        valid_clusters.append(current_cluster)
            
            return valid_clusters
        
        # Function to create a PSO-inspired solution
        def create_pso_solution():
            # Encode problem as particle position (TPS to cluster assignments)
            particle_pos = np.random.random(n_points) * expected_clusters
            cluster_assignments = np.floor(particle_pos).astype(int)
            
            # Convert to solution format
            max_cluster = max(cluster_assignments) if len(cluster_assignments) > 0 else 0
            solution = [[] for _ in range(max_cluster + 1)]
            for i, cluster_id in enumerate(cluster_assignments):
                solution[cluster_id].append(i)
            
            # Filter empty clusters
            solution = [cluster for cluster in solution if cluster]
            
            # Split clusters that exceed capacity
            fixed_solution = []
            for cluster in solution:
                cluster_volume = sum(volumes[idx] for idx in cluster)
                if cluster_volume <= max_capacity:
                    fixed_solution.append(cluster)
                else:
                    # Sort by volume for better packing
                    sorted_tps = sorted(cluster, key=lambda idx: volumes[idx], reverse=True)
                    current_cluster = []
                    current_volume = 0
                    
                    for tps_idx in sorted_tps:
                        if current_volume + volumes[tps_idx] <= max_capacity:
                            current_cluster.append(tps_idx)
                            current_volume += volumes[tps_idx]
                        else:
                            if current_cluster:
                                fixed_solution.append(current_cluster)
                            current_cluster = [tps_idx]
                            current_volume = volumes[tps_idx]
                    
                    if current_cluster:
                        fixed_solution.append(current_cluster)
            
            return fixed_solution
        
        # Initialize population - GA FIRST APPROACH
        self.update_progress(5, "Membuat populasi awal GA...")
        update_detail_progress(0, "Membuat solusi dengan pendekatan GA...")
        population = []
        
        # Add greedy solution (enhanced initialization for GA)
        greedy_solution = create_greedy_solution()
        population.append(greedy_solution)
        update_detail_progress(20, "Membuat solusi dengan pendekatan k-means...")
        progress_text.insert("end", "Membuat solusi awal dengan pendekatan greedy...\n")
        self.root.update_idletasks()
        
        # Add K-means solutions with different k values
        for k in range(max(2, expected_clusters-2), expected_clusters+5):
            try:
                kmeans_solution = create_kmeans_solution(k)
                population.append(kmeans_solution)
            except Exception as e:
                progress_text.insert("end", f"Warning: Gagal membuat solusi k-means dengan k={k}: {str(e)}\n")
        
        update_detail_progress(40, "Membuat solusi secara acak...")
        progress_text.insert("end", "Membuat solusi awal dengan pendekatan k-means...\n")
        self.root.update_idletasks()
        
        # Add random solutions
        while len(population) < population_size * 0.5:  # Reduce number of random solutions for PSO
            try:
                random_solution = create_random_solution()
                population.append(random_solution)
            except Exception as e:
                progress_text.insert("end", f"Warning: Gagal membuat solusi acak: {str(e)}\n")
        
        # Add PSO-inspired solutions
        update_detail_progress(60, "Membuat solusi dengan pendekatan PSO...")
        progress_text.insert("end", "Membuat solusi awal dengan pendekatan PSO...\n")
        self.root.update_idletasks()
        
        for _ in range(min(10, population_size // 5)):
            try:
                pso_solution = create_pso_solution()
                population.append(pso_solution)
            except Exception as e:
                progress_text.insert("end", f"Warning: Gagal membuat solusi PSO: {str(e)}\n")
        
        update_detail_progress(70, "Membuat solusi tambahan...")
        progress_text.insert("end", "Membuat solusi awal lainnya...\n")
        self.root.update_idletasks()
        
        # Make sure we have enough solutions for GA
        while len(population) < population_size:
            # Clone and mutate existing solutions
            idx = random.randint(0, len(population)-1)
            new_solution = [cluster.copy() for cluster in population[idx]]
            
            # Apply random mutation
            if random.random() < 0.5:
                # Try to merge two clusters
                if len(new_solution) >= 2:
                    c1 = random.randint(0, len(new_solution)-1)
                    c2 = random.randint(0, len(new_solution)-1)
                    
                    while c2 == c1 and len(new_solution) > 1:
                        c2 = random.randint(0, len(new_solution)-1)
                    
                    if c1 != c2:
                        merged = new_solution[c1] + new_solution[c2]
                        # If merged cluster is valid, use it
                        if sum(volumes[idx] for idx in merged) <= max_capacity:
                            # Replace first cluster with merged
                            new_solution[c1] = merged
                            # Remove second cluster
                            new_solution.pop(c2 if c2 < c1 else c1)
            else:
                # Try to split a cluster
                if new_solution:
                    c = random.randint(0, len(new_solution)-1)
                    if len(new_solution[c]) >= 2:
                        # Randomly split
                        split_point = random.randint(1, len(new_solution[c])-1)
                        c1 = new_solution[c][:split_point]
                        c2 = new_solution[c][split_point:]
                        
                        # Replace with split clusters
                        new_solution[c] = c1
                        new_solution.append(c2)
            
            population.append(new_solution)
            
            # Ensure we don't go into an infinite loop
            if len(population) >= population_size * 2:
                break
        
        # Trim population to desired size
        population = population[:population_size]
        
        update_detail_progress(80, "Menyelesaikan populasi awal GA...")
        
        # Log initial population
        progress_text.insert("end", f"Populasi awal GA terbentuk dengan {len(population)} solusi\n")
        best_initial = min(population, key=lambda sol: calculate_fitness(sol)[0])
        initial_fitness, initial_distance, _, _, _, _ = calculate_fitness(best_initial)
        progress_text.insert("end", f"Solusi terbaik awal: {len(best_initial)} cluster, fitness={initial_fitness:.2f}, jarak={initial_distance:.2f}\n\n")
        self.root.update_idletasks()
        
        update_detail_progress(100, "Populasi awal GA selesai dibentuk.")
        self.update_progress(10, "Mulai iterasi algoritma hybrid GA-VNS...")
        
        # Initialize for adaptive parameters
        adaptive_mutation_rate = mutation_rate  
        adaptive_crossover_rate = crossover_rate
        diversity_history = []
        
        # Function to calculate population diversity
        def calculate_diversity(pop):
            # Handle small populations
            if len(pop) <= 1:
                return 0
            
            # Calculate similarity between solutions
            total_dist = 0
            count = 0
            
            # Sample pairs for efficiency (max 50 pairs)
            max_pairs = min(50, len(pop) * (len(pop) - 1) // 2)
            if len(pop) > 10:
                # Sample pairs
                pairs = []
                for _ in range(max_pairs):
                    i, j = random.sample(range(len(pop)), 2)
                    pairs.append((i, j))
            else:
                # For small populations, evaluate all pairs
                pairs = [(i, j) for i in range(len(pop)) for j in range(i+1, len(pop))]
            
            # Process each pair
            for i, j in pairs:
                # Convert cluster assignments to TPS sets
                tps_set_i = set()
                for cluster in pop[i]:
                    tps_set_i.update(cluster)
                    
                tps_set_j = set()
                for cluster in pop[j]:
                    tps_set_j.update(cluster)
            
                # Calculate Jaccard distance
                if tps_set_i and tps_set_j:
                    intersection = len(tps_set_i.intersection(tps_set_j))
                    union = len(tps_set_i.union(tps_set_j))
                    
                    if union > 0:
                        similarity = intersection / union
                        total_dist += (1 - similarity)  # Convert to distance
                        count += 1
            
            # Return average diversity
            return total_dist / count if count > 0 else 0
        
        # Main GA-VNS hybrid algorithm loop
        best_solution = None
        best_fitness = float('inf')
        best_distance = float('inf')
        
        stagnation_count = 0
        last_improvement = 0
        
        # Progress calculation for iterations
        iteration_progress_start = 10
        iteration_progress_end = 80
        progress_per_iteration = (iteration_progress_end - iteration_progress_start) / max_iterations
        
        for iteration in range(max_iterations):
            # Update progress
            current_progress = iteration_progress_start + (iteration * progress_per_iteration)
            self.update_progress(current_progress, f"Iterasi {iteration+1}/{max_iterations}")
            update_detail_progress(
                (iteration * 100) / max_iterations,
                f"Iterasi {iteration+1}/{max_iterations}"
            )
            
            # Sort population by fitness - GA approach
            population.sort(key=lambda sol: calculate_fitness(sol)[0])
            
            # Store the best solution
            current_best = population[0]
            current_fitness, current_distance, cv, dv, mp, dp = calculate_fitness(current_best)
            
            # Calculate population diversity and adapt parameters
            diversity = calculate_diversity(population)
            diversity_history.append(diversity)
            
            # Adaptive parameter tuning based on diversity
            if len(diversity_history) >= 2:
                diversity_change = diversity_history[-1] - diversity_history[-2]
                
                # If diversity decreasing rapidly, increase mutation to encourage exploration
                if diversity_change < -0.05:
                    adaptive_mutation_rate = min(0.9, adaptive_mutation_rate * 1.5)
                # If diversity stable or increasing, gradually return to base rate
                else:
                    adaptive_mutation_rate = mutation_rate + (adaptive_mutation_rate - mutation_rate) * 0.9
                
                # Adjust crossover rate inversely to balance exploration/exploitation
                adaptive_crossover_rate = max(0.5, min(0.95, 1.0 - adaptive_mutation_rate/2))
            
            # Log progress
            if debug_mode and iteration % 10 == 0:
                log = f"Iterasi {iteration}/{max_iterations} ({(iteration*100/max_iterations):.1f}%): "
                log += f"Fitness={current_fitness:.2f}, Jarak={current_distance:.2f}, "
                log += f"Clusters={len(current_best)}, Diversitas={diversity:.4f}\n"
                log += f"  MutRate={adaptive_mutation_rate:.4f}, CrossRate={adaptive_crossover_rate:.4f}\n"
                
                if cv > 0 or dv > 0 or mp > 0 or dp > 0:
                    log += f"  Pelanggaran: Kapasitas={cv:.2f}, Jarak={dv:.2f}, TPS Hilang={mp:.2f}, TPS Duplikat={dp:.2f}\n"
                
                progress_text.insert("end", log)
                self.root.update_idletasks()
                
                # Add to debug info
                debug_info.append({
                    'iteration': iteration,
                    'fitness': current_fitness,
                    'distance': current_distance,
                    'num_clusters': len(current_best),
                    'diversity': diversity,
                    'mutation_rate': adaptive_mutation_rate,
                    'crossover_rate': adaptive_crossover_rate
                })
            
            if current_fitness < best_fitness:
                best_solution = [cluster.copy() for cluster in current_best]
                best_fitness = current_fitness
                best_distance = current_distance
                last_improvement = iteration
                stagnation_count = 0
                
                # Log improvement
                progress_text.insert("end", f"✓ Solusi baru terbaik: {len(best_solution)} cluster, fitness={best_fitness:.2f}, jarak={best_distance:.2f}\n")
                self.root.update_idletasks()
            else:
                stagnation_count += 1
            
            # Early stopping if no improvement for a while
            if stagnation_count > max_iterations // 4:
                progress_text.insert("end", f"\nBerhenti lebih awal karena tidak ada peningkatan selama {stagnation_count} iterasi\n")
                break
            
            # Create new population using GA operations
            new_population = [population[0]]  # Keep the best solution (elitism)
            
            while len(new_population) < population_size:
                # GA OPERATIONS
                # Selection (tournament selection)
                tournament_size = min(3, len(population))
                parent1 = min(random.sample(population, tournament_size), key=lambda sol: calculate_fitness(sol)[0])
                parent2 = min(random.sample(population, tournament_size), key=lambda sol: calculate_fitness(sol)[0])
                
                # Crossover
                if random.random() < adaptive_crossover_rate:
                    # Order clusters by their total volume
                    parent1_ordered = sorted(parent1, key=lambda c: sum(volumes[idx] for idx in c), reverse=True)
                    parent2_ordered = sorted(parent2, key=lambda c: sum(volumes[idx] for idx in c), reverse=True)
                    
                    # Initialize child with empty clusters
                    child = []
                    
                    # Track assigned TPS
                    assigned_tps = set()
                    
                    # First, randomly select some clusters from parent1
                    for cluster in parent1_ordered:
                        if random.random() < 0.5:  # 50% chance to include each cluster
                            # Only include unassigned TPS
                            valid_cluster = [idx for idx in cluster if idx not in assigned_tps]
                            
                            if valid_cluster:
                                child.append(valid_cluster)
                                assigned_tps.update(valid_cluster)
                    
                    # Then add clusters from parent2 that don't overlap
                    for cluster in parent2_ordered:
                        # Only include unassigned TPS
                        valid_cluster = [idx for idx in cluster if idx not in assigned_tps]
                        
                        if valid_cluster:
                            # Check if cluster meets minimum capacity
                            if sum(volumes[idx] for idx in valid_cluster) >= min_capacity / 2:  # Relaxed constraint
                                child.append(valid_cluster)
                                assigned_tps.update(valid_cluster)
                    
                    # Check for unassigned TPS
                    unassigned = set(range(n_points)) - assigned_tps
                    
                    if unassigned:
                        # Try to assign remaining TPS to existing clusters
                        for tps_idx in list(unassigned):
                            added = False
                            
                            # Try to add to existing clusters
                            for cluster in child:
                                # Check capacity
                                cluster_volume = sum(volumes[idx] for idx in cluster)
                                
                                if cluster_volume + volumes[tps_idx] <= max_capacity:
                                    cluster.append(tps_idx)
                                    unassigned.remove(tps_idx)
                                    added = True
                                    break
                            
                            if not added:
                                continue
                        
                        # Create new clusters for remaining unassigned TPS
                        if unassigned:
                            # Sort by volume for better packing
                            remaining = sorted(list(unassigned), key=lambda idx: volumes[idx], reverse=True)
                            
                            current_cluster = []
                            current_volume = 0
                            
                            for tps_idx in remaining:
                                if current_volume + volumes[tps_idx] <= max_capacity:
                                    current_cluster.append(tps_idx)
                                    current_volume += volumes[tps_idx]
                                else:
                                    # If can't add due to capacity, create a new cluster
                                    if current_cluster:
                                        child.append(current_cluster)
                                    
                                    current_cluster = [tps_idx]
                                    current_volume = volumes[tps_idx]
                            
                            # Add the last cluster if not empty
                            if current_cluster:
                                child.append(current_cluster)
                    
                    # Remove empty clusters
                    child = [cluster for cluster in child if cluster]
                    
                else:
                    # No crossover, just copy one parent
                    if random.random() < 0.5:
                        child = [cluster.copy() for cluster in parent1]
                    else:
                        child = [cluster.copy() for cluster in parent2]
                
                # Mutation - GA operation
                if random.random() < adaptive_mutation_rate:
                    # Choose mutation type
                    mutation_type = random.choice(['swap', 'move', 'split', 'merge'])
                    
                    if mutation_type == 'swap':
                        # Swap two random TPS between clusters
                        if len(child) >= 2:
                            # Select two random clusters
                            cluster1_idx = random.randint(0, len(child) - 1)
                            cluster2_idx = random.randint(0, len(child) - 1)
                            
                            # Make sure they're different
                            while cluster2_idx == cluster1_idx and len(child) > 1:
                                cluster2_idx = random.randint(0, len(child) - 1)
                            
                            if len(child[cluster1_idx]) > 0 and len(child[cluster2_idx]) > 0:
                                # Select random TPS from each cluster
                                tps1_idx = random.randint(0, len(child[cluster1_idx]) - 1)
                                tps2_idx = random.randint(0, len(child[cluster2_idx]) - 1)
                                
                                # Get TPS values
                                tps1 = child[cluster1_idx][tps1_idx]
                                tps2 = child[cluster2_idx][tps2_idx]
                                
                                # Check if swap would be valid
                                vol1 = sum(volumes[idx] for idx in child[cluster1_idx] if idx != tps1)
                                vol2 = sum(volumes[idx] for idx in child[cluster2_idx] if idx != tps2)
                                
                                new_vol1 = vol1 + volumes[tps2]
                                new_vol2 = vol2 + volumes[tps1]
                                
                                if new_vol1 <= max_capacity and new_vol2 <= max_capacity:
                                    # Perform swap
                                    child[cluster1_idx][tps1_idx] = tps2
                                    child[cluster2_idx][tps2_idx] = tps1
                    
                    elif mutation_type == 'move':
                        # Move a random TPS from one cluster to another
                        if len(child) >= 2:
                            # Select source and destination clusters
                            from_cluster_idx = random.randint(0, len(child) - 1)
                            to_cluster_idx = random.randint(0, len(child) - 1)
                            
                            # Make sure they're different
                            while to_cluster_idx == from_cluster_idx and len(child) > 1:
                                to_cluster_idx = random.randint(0, len(child) - 1)
                            
                            # Make sure source cluster has more than one TPS
                            if len(child[from_cluster_idx]) > 1:
                                # Select a random TPS to move
                                tps_idx = random.randint(0, len(child[from_cluster_idx]) - 1)
                                tps = child[from_cluster_idx][tps_idx]
                                
                                # Check if move would be valid
                                to_vol = sum(volumes[idx] for idx in child[to_cluster_idx])
                                from_vol = sum(volumes[idx] for idx in child[from_cluster_idx])
                                
                                new_to_vol = to_vol + volumes[tps]
                                new_from_vol = from_vol - volumes[tps]
                                
                                if new_to_vol <= max_capacity and (new_from_vol >= min_capacity or len(child) > expected_clusters):
                                    # Perform move
                                    child[from_cluster_idx].pop(tps_idx)
                                    child[to_cluster_idx].append(tps)
                                    
                                    # Remove empty clusters
                                    if not child[from_cluster_idx]:
                                        child.pop(from_cluster_idx)
                    
                    elif mutation_type == 'split':
                        # Split a cluster
                        if child:
                            # Choose a cluster to split (prefer larger ones)
                            cluster_weights = [len(cluster) for cluster in child]
                            total_weight = sum(cluster_weights)
                            
                            if total_weight > 0:
                                probs = [w/total_weight for w in cluster_weights]
                                cluster_idx = random.choices(range(len(child)), weights=probs)[0]
                                
                                # Only split if cluster has more than one TPS
                                if len(child[cluster_idx]) > 3:
                                    # Randomly split
                                    split_point = random.randint(1, len(child[cluster_idx]) - 1)
                                    
                                    # Create two new clusters
                                    cluster1 = child[cluster_idx][:split_point]
                                    cluster2 = child[cluster_idx][split_point:]
                                    
                                    # Check if both parts are valid
                                    vol1 = sum(volumes[idx] for idx in cluster1)
                                    vol2 = sum(volumes[idx] for idx in cluster2)
                                    
                                    # If both are valid or we're below expected clusters, perform split
                                    if (vol1 >= min_capacity and vol2 >= min_capacity) or len(child) < expected_clusters:
                                        # Replace original with first part
                                        child[cluster_idx] = cluster1
                                        # Add second part as new cluster
                                        child.append(cluster2)
                    
                    elif mutation_type == 'merge':
                        # Merge two clusters
                        if len(child) >= 2:
                            # Select two random clusters
                            cluster1_idx = random.randint(0, len(child) - 1)
                            cluster2_idx = random.randint(0, len(child) - 1)
                            
                            # Make sure they're different
                            while cluster2_idx == cluster1_idx and len(child) > 1:
                                cluster2_idx = random.randint(0, len(child) - 1)
                            
                            if cluster1_idx != cluster2_idx:
                                # Check if merge would be valid
                                merged = child[cluster1_idx] + child[cluster2_idx]
                                merged_vol = sum(volumes[idx] for idx in merged)
                                
                                if merged_vol <= max_capacity:
                                    # Perform merge
                                    # Replace first cluster with merged
                                    child[cluster1_idx] = merged
                                    # Remove second cluster
                                    child.pop(cluster2_idx if cluster2_idx < cluster1_idx else cluster1_idx)
                
                # NOW APPLY VNS AS A REFINEMENT STEP TO PROMISING SOLUTIONS
                # Apply VNS with probability based on solution quality
                # Apply more intensive VNS to better solutions
                apply_vns_prob = 0.3 - (len(new_population) / (population_size * 3))  # Higher prob for early solutions
                apply_vns_prob = max(0.05, min(0.5, apply_vns_prob))  # Between 5% and 50%
                
                if random.random() < apply_vns_prob:
                    # Define neighborhood structures for VNS
                    neighborhood_structures = ['swap_tps', 'relocate_tps', 'exchange_tps', 'split_merge']
                    
                    # Apply VNS using different neighborhood structures
                    for neighborhood in neighborhood_structures:
                        # Create a copy of the current solution
                        current_solution = [cluster.copy() for cluster in child]
                        current_fitness, _, _, _, _, _ = calculate_fitness(current_solution)
                        
                        if neighborhood == 'swap_tps':
                            # Swap two TPS between different clusters
                            if len(current_solution) >= 2:
                                # Choose two random clusters
                                idx1 = random.randint(0, len(current_solution) - 1)
                                idx2 = random.randint(0, len(current_solution) - 1)
                                
                                # Ensure they're different clusters
                                while idx2 == idx1 and len(current_solution) > 1:
                                    idx2 = random.randint(0, len(current_solution) - 1)
                                
                                if len(current_solution[idx1]) > 0 and len(current_solution[idx2]) > 0:
                                    # Choose a random TPS from each cluster
                                    tps1_pos = random.randint(0, len(current_solution[idx1]) - 1)
                                    tps2_pos = random.randint(0, len(current_solution[idx2]) - 1)
                                    
                                    # Get TPS IDs
                                    tps1 = current_solution[idx1][tps1_pos]
                                    tps2 = current_solution[idx2][tps2_pos]
                                    
                                    # Check if swap is valid (capacity constraints)
                                    cluster1_vol = sum(volumes[idx] for idx in current_solution[idx1])
                                    cluster2_vol = sum(volumes[idx] for idx in current_solution[idx2])
                                    
                                    new_vol1 = cluster1_vol - volumes[tps1] + volumes[tps2]
                                    new_vol2 = cluster2_vol - volumes[tps2] + volumes[tps1]
                                    
                                    if new_vol1 <= max_capacity and new_vol2 <= max_capacity:
                                        # Perform swap
                                        current_solution[idx1][tps1_pos] = tps2
                                        current_solution[idx2][tps2_pos] = tps1
                        
                        elif neighborhood == 'relocate_tps':
                            # Move a TPS from one cluster to another
                            if len(current_solution) >= 2:
                                # Choose source cluster
                                source_idx = random.randint(0, len(current_solution) - 1)
                                
                                # Make sure source cluster has at least two TPS
                                if len(current_solution[source_idx]) >= 2:
                                    # Choose a random TPS to relocate
                                    tps_pos = random.randint(0, len(current_solution[source_idx]) - 1)
                                    tps = current_solution[source_idx][tps_pos]
                                    
                                    # Choose destination cluster
                                    dest_idx = random.randint(0, len(current_solution) - 1)
                                    while dest_idx == source_idx and len(current_solution) > 1:
                                        dest_idx = random.randint(0, len(current_solution) - 1)
                                    
                                    # Check capacity constraint
                                    dest_vol = sum(volumes[idx] for idx in current_solution[dest_idx])
                                    source_vol = sum(volumes[idx] for idx in current_solution[source_idx])
                                    
                                    if dest_vol + volumes[tps] <= max_capacity:
                                        # Check if source cluster will remain valid
                                        remaining_vol = source_vol - volumes[tps]
                                        if remaining_vol >= min_capacity or len(current_solution[source_idx]) <= 2:
                                            # Move TPS
                                            current_solution[source_idx].remove(tps)
                                            current_solution[dest_idx].append(tps)
                                            
                                            # Remove empty clusters
                                            if not current_solution[source_idx]:
                                                current_solution.pop(source_idx)
                        
                        elif neighborhood == 'exchange_tps':
                            # Exchange groups of TPS between clusters
                            if len(current_solution) >= 2:
                                # Choose two clusters
                                idx1 = random.randint(0, len(current_solution) - 1)
                                idx2 = random.randint(0, len(current_solution) - 1)
                                
                                # Ensure they're different clusters
                                while idx2 == idx1 and len(current_solution) > 1:
                                    idx2 = random.randint(0, len(current_solution) - 1)
                                
                                if len(current_solution[idx1]) >= 2 and len(current_solution[idx2]) >= 2:
                                    # Choose a subset of TPS from each cluster
                                    size1 = min(len(current_solution[idx1]) // 2, random.randint(1, 3))
                                    size2 = min(len(current_solution[idx2]) // 2, random.randint(1, 3))
                                    
                                    subset1 = random.sample(current_solution[idx1], size1)
                                    subset2 = random.sample(current_solution[idx2], size2)
                                    
                                    # Calculate volumes
                                    vol1 = sum(volumes[idx] for idx in current_solution[idx1])
                                    vol2 = sum(volumes[idx] for idx in current_solution[idx2])
                                    
                                    subset1_vol = sum(volumes[idx] for idx in subset1)
                                    subset2_vol = sum(volumes[idx] for idx in subset2)
                                    
                                    # Check capacity constraints
                                    new_vol1 = vol1 - subset1_vol + subset2_vol
                                    new_vol2 = vol2 - subset2_vol + subset1_vol
                                    
                                    if new_vol1 <= max_capacity and new_vol2 <= max_capacity:
                                        # Perform exchange
                                        for tps in subset1:
                                            current_solution[idx1].remove(tps)
                                        
                                        for tps in subset2:
                                            current_solution[idx2].remove(tps)
                                        
                                        current_solution[idx1].extend(subset2)
                                        current_solution[idx2].extend(subset1)
                        
                        elif neighborhood == 'split_merge':
                            # Randomly choose to split or merge clusters
                            if random.random() < 0.5 and current_solution:
                                # Split a cluster
                                if any(len(cluster) > 3 for cluster in current_solution):
                                    # Choose a larger cluster
                                    large_clusters = [i for i, c in enumerate(current_solution) if len(c) > 3]
                                    idx = random.choice(large_clusters)
                                    
                                    # Determine split point
                                    split_point = random.randint(1, len(current_solution[idx]) - 1)
                                    
                                    # Split cluster
                                    cluster1 = current_solution[idx][:split_point]
                                    cluster2 = current_solution[idx][split_point:]
                                    
                                    # Check validity of split clusters
                                    vol1 = sum(volumes[idx] for idx in cluster1)
                                    vol2 = sum(volumes[idx] for idx in cluster2)
                                    
                                    if vol1 <= max_capacity and vol2 <= max_capacity:
                                        if vol1 >= min_capacity and vol2 >= min_capacity:
                                            # Perform split
                                            current_solution[idx] = cluster1
                                            current_solution.append(cluster2)
                            else:
                                # Merge clusters
                                if len(current_solution) >= 2:
                                    # Choose two random clusters
                                    idx1 = random.randint(0, len(current_solution) - 1)
                                    idx2 = random.randint(0, len(current_solution) - 1)
                                    
                                    # Ensure they're different clusters
                                    while idx2 == idx1 and len(current_solution) > 1:
                                        idx2 = random.randint(0, len(current_solution) - 1)
                                    
                                    # Check if merge is valid
                                    merged = current_solution[idx1] + current_solution[idx2]
                                    merged_vol = sum(volumes[idx] for idx in merged)
                                    
                                    if merged_vol <= max_capacity:
                                        # Perform merge
                                        current_solution[idx1] = merged
                                        if idx1 != idx2:  # safeguard
                                            current_solution.pop(idx2 if idx2 < idx1 else idx2)
                        
                        # Check if the neighborhood move improved the solution
                        new_fitness, _, _, _, _, _ = calculate_fitness(current_solution)
                        
                        if new_fitness < current_fitness:
                            # Accept improved solution
                            child = current_solution
                            break  # Exit VNS early if we found an improvement
                
                # Clean up and validate the solution - common to both GA and VNS
                # Remove empty clusters
                child = [cluster for cluster in child if cluster]
                
                # Check if all TPS are assigned
                all_assigned_tps = set()
                for cluster in child:
                    for idx in cluster:
                        all_assigned_tps.add(idx)
                
                # If not all TPS are assigned, fix the solution
                if len(all_assigned_tps) < n_points:
                    missing_tps = set(range(n_points)) - all_assigned_tps
                    
                    # Try to add missing TPS to existing clusters
                    for tps_idx in list(missing_tps):
                        added = False
                        
                        # Try to add to existing clusters
                        for cluster in child:
                            # Check capacity
                            cluster_volume = sum(volumes[idx] for idx in cluster)
                            
                            if cluster_volume + volumes[tps_idx] <= max_capacity:
                                cluster.append(tps_idx)
                                missing_tps.remove(tps_idx)
                                added = True
                                break
                        
                        if not added:
                            continue
                    
                    # Create new clusters for remaining missing TPS
                    if missing_tps:
                        remaining = sorted(list(missing_tps), key=lambda idx: volumes[idx], reverse=True)
                        
                        current_cluster = []
                        current_volume = 0
                        
                        for tps_idx in remaining:
                            if current_volume + volumes[tps_idx] <= max_capacity:
                                current_cluster.append(tps_idx)
                                current_volume += volumes[tps_idx]
                            else:
                                # If can't add due to capacity, create a new cluster
                                if current_cluster:
                                    child.append(current_cluster)
                                
                                current_cluster = [tps_idx]
                                current_volume = volumes[tps_idx]
                        
                        # Add the last cluster if not empty
                        if current_cluster:
                            child.append(current_cluster)
                
                # Check for duplicate TPS assignments
                all_tps = []
                for cluster in child:
                    all_tps.extend(cluster)
                
                if len(all_tps) != len(set(all_tps)):
                    # Fix duplicates by keeping only first occurrence
                    seen = set()
                    for i, cluster in enumerate(child):
                        new_cluster = []
                        for tps in cluster:
                            if tps not in seen:
                                new_cluster.append(tps)
                                seen.add(tps)
                        child[i] = new_cluster
                    
                    # Remove empty clusters
                    child = [cluster for cluster in child if cluster]
                
                # Add to new population
                new_population.append(child)
                
                # Break if we've reached population size
                if len(new_population) >= population_size:
                    break
            
            # If new population is still smaller than desired, add random solutions
            while len(new_population) < population_size:
                try:
                    new_population.append(create_random_solution())
                except:
                    # Clone best solution as fallback
                    new_population.append([cluster.copy() for cluster in population[0]])
            
            # Replace population
            population = new_population[:population_size]
        
        # Final logging
        progress_text.insert("end", f"\nOptimisasi selesai setelah {iteration+1} iterasi\n")
        progress_text.insert("end", f"Solusi terbaik: {len(best_solution)} cluster dengan jarak total {best_distance:.2f}\n")
        
        # Update progress for final ACO route optimization
        self.update_progress(85, "Melakukan optimasi rute final dengan ACO...")
        
        # Optimize routes for final solution (existing code)
        if optimize_routes:
            progress_text.insert("end", "\nMelakukan optimasi akhir pada rute dengan ACO...\n")
            
            # Cek apakah menggunakan fixed endpoints
            use_fixed_endpoints = self.use_fixed_endpoints_var.get()
            start_point_name = self.start_point_var.get() if use_fixed_endpoints else None
            end_point_name = self.end_point_var.get() if use_fixed_endpoints else None
            
            # Cari indeks titik awal dan akhir
            start_idx = None
            end_idx = None
            
            if use_fixed_endpoints:
                # Rest of the fixed endpoints code (tidak berubah)
                # ...
                names = self.data_info['names']
                
                # Coba temukan titik awal (Garasi)
                if start_point_name in names:
                    # Match persis
                    for i, name in enumerate(names):
                        if name == start_point_name:
                            start_idx = i
                            break
                else:
                    # Match parsial
                    for i, name in enumerate(names):
                        if start_point_name.lower() in name.lower():
                            start_idx = i
                            break
                
                # Coba temukan titik akhir (TPA)
                if end_point_name in names:
                    # Match persis
                    for i, name in enumerate(names):
                        if name == end_point_name:
                            end_idx = i
                            break
                else:
                    # Match parsial
                    for i, name in enumerate(names):
                        if end_point_name.lower() in name.lower():
                            end_idx = i
                            break
                
                # Log hasil
                if start_idx is not None:
                    progress_text.insert("end", f"Titik awal rute (Garasi): {names[start_idx]} (index {start_idx})\n")
                else:
                    progress_text.insert("end", f"PERINGATAN: Titik awal '{start_point_name}' tidak ditemukan dalam data!\n")
                    use_fixed_endpoints = False
                
                if end_idx is not None:
                    progress_text.insert("end", f"Titik akhir rute (TPA): {names[end_idx]} (index {end_idx})\n")
                else:
                    progress_text.insert("end", f"PERINGATAN: Titik akhir '{end_point_name}' tidak ditemukan dalam data!\n")
                    use_fixed_endpoints = False
            
            # ACO route optimization code (tidak berubah)
            optimized_solution = []
            total_improvement = 0
            
            # Calculate total clusters for progress tracking
            total_clusters = len(best_solution)
            progress_increment = 15.0 / total_clusters if total_clusters > 0 else 0
            
            for i, cluster in enumerate(best_solution):
                # Update progress
                current_progress = 85 + (i * progress_increment)
                self.update_progress(current_progress, f"Optimasi ACO untuk cluster {i+1}/{total_clusters}")
                update_detail_progress(
                    (i * 100) / total_clusters,
                    f"Optimasi rute cluster {i+1}/{total_clusters}"
                )
                
                if len(cluster) >= 1:  # Bahkan 1 TPS perlu optimasi dengan titik awal/akhir
                    # Hitung jarak rute asli
                    original_dist = 0
                    
                    if use_fixed_endpoints and start_idx is not None and end_idx is not None:
                        # Jarak Garasi -> TPS pertama
                        original_dist += distance_matrix[start_idx][cluster[0]]
                        
                        # Jarak antar TPS
                        for j in range(len(cluster)-1):
                            original_dist += distance_matrix[cluster[j]][cluster[j+1]]
                        
                        # Jarak TPS terakhir -> TPA
                        original_dist += distance_matrix[cluster[-1]][end_idx]
                    else:
                        # Hanya jarak antar TPS
                        for j in range(len(cluster)-1):
                            original_dist += distance_matrix[cluster[j]][cluster[j+1]]
                    
                    # Optimasi dengan ACO
                    optimized_route, route_distance = self.optimize_route_aco(
                        cluster, distance_matrix, 
                        start_idx=start_idx if use_fixed_endpoints else None,
                        end_idx=end_idx if use_fixed_endpoints else None,
                        num_ants=aco_ants, 
                        num_iterations=aco_iterations,
                        update_progress=None  # Don't show nested progress to avoid UI clutter
                    )
                    
                    # Hitung peningkatan
                    improvement = original_dist - route_distance
                    
                    # Gunakan rute optimal jika lebih baik atau jika titik awal/akhir diperlukan
                    if improvement > 0 or use_fixed_endpoints:
                        optimized_solution.append(optimized_route)
                        total_improvement += improvement
                        
                        # Pesan log disesuaikan dengan tipe rute
                        if use_fixed_endpoints and start_idx is not None and end_idx is not None:
                            # Rute dengan titik awal dan akhir
                            route_desc = f"Garasi → {len(cluster)} TPS → TPA"
                        else:
                            # Rute hanya TPS
                            route_desc = f"{len(cluster)} TPS"
                        
                        progress_text.insert("end", f"Cluster {i+1} ({route_desc}): Peningkatan rute sebesar {improvement:.2f} unit\n")
                    else:
                        optimized_solution.append(cluster)
                else:
                    optimized_solution.append(cluster)
            
            # Use the optimized solution if we have one
            if optimized_solution:
                best_solution = optimized_solution
        
        # Complete progress
        self.update_progress(100, "Validasi solusi akhir...")
        
        # Validate final solution
        all_tps = []
        for cluster in best_solution:
            all_tps.extend(cluster)
        
        missing = set(range(n_points)) - set(all_tps)
        duplicates = len(all_tps) - len(set(all_tps))
        
        if missing:
            progress_text.insert("end", f"PERINGATAN: {len(missing)} TPS tidak masuk cluster: {missing}\n")
        
        if duplicates:
            progress_text.insert("end", f"PERINGATAN: {duplicates} TPS duplikat dalam cluster\n")
        
        # Check cluster capacities
        for i, cluster in enumerate(best_solution):
            cluster_vol = sum(volumes[idx] for idx in cluster)
            if cluster_vol < min_capacity:
                progress_text.insert("end", f"PERINGATAN: Cluster {i+1} di bawah kapasitas minimum: {cluster_vol:.2f} m³\n")
            if cluster_vol > max_capacity:
                progress_text.insert("end", f"PERINGATAN: Cluster {i+1} melebihi kapasitas maksimum: {cluster_vol:.2f} m³\n")
        
        self.root.update_idletasks()
        
        # Filter out Garage and TPA if requested
        if self.use_fixed_endpoints_var.get():
            # Filter clusters to remove any occurrence of Garage or TPA
            filtered_solution = []
            names = self.data_info['names']
            
            # Daftar nama yang akan diabaikan
            ignore_names = ["Garasi", "TPA", "TPA Troketon"]
            
            # Identifikasi indeks yang harus difilter
            ignore_indices = []
            for i, name in enumerate(names):
                if isinstance(name, str) and any(ignore.lower() in name.lower() for ignore in ignore_names):
                    ignore_indices.append(i)
            
            # Filter tiap cluster
            for cluster in best_solution:
                filtered_cluster = [idx for idx in cluster if idx not in ignore_indices]
                if filtered_cluster:  # Hanya tambahkan jika cluster tidak kosong
                    filtered_solution.append(filtered_cluster)
            
            # Gunakan solution yang telah difilter
            best_solution = filtered_solution
        
        # Return the best solution
        return best_solution if best_solution else [list(range(n_points))], debug_info
    
    def calculate_solution_metrics(self, clusters, distance_matrix, volumes, optimize_routes=True):
        """Calculate metrics for a clustering solution including Garage-TPA distances"""
        # Calculate total volume
        total_volume = sum(volumes)
        
        # Get Garage and TPA indices if they are used
        use_fixed_endpoints = self.use_fixed_endpoints_var.get()
        start_point_name = self.start_point_var.get() if use_fixed_endpoints else None
        end_point_name = self.end_point_var.get() if use_fixed_endpoints else None
        
        # Find indices of start and end points
        start_idx = None
        end_idx = None
        
        if use_fixed_endpoints and 'names' in self.data_info:
            names = self.data_info['names']
            
            # Look for Garage
            for i, name in enumerate(names):
                if isinstance(name, str) and (start_point_name.lower() == name.lower() or 
                                            start_point_name.lower() in name.lower()):
                    start_idx = i
                    break
            
            # Look for TPA
            for i, name in enumerate(names):
                if isinstance(name, str) and (end_point_name.lower() == name.lower() or 
                                            end_point_name.lower() in name.lower()):
                    end_idx = i
                    break
        
        # Calculate cluster metrics
        cluster_metrics = []
        total_distance = 0
        total_route_distance = 0
        total_complete_route_distance = 0  # Including Garage and TPA
        
        for i, cluster in enumerate(clusters):
            cluster_volume = sum(volumes[idx] for idx in cluster)
            volume_percentage = (cluster_volume / total_volume) * 100
            
            # Calculate intra-cluster distances
            intra_distances = []
            for j in range(len(cluster)):
                for k in range(j+1, len(cluster)):
                    dist = distance_matrix[cluster[j]][cluster[k]]
                    intra_distances.append(dist)
            
            avg_intra_distance = np.mean(intra_distances) if intra_distances else 0
            max_intra_distance = np.max(intra_distances) if intra_distances else 0
            total_distance += sum(intra_distances)
            
            # Calculate route distance (TSP only between TPS)
            route_distance = 0
            if len(cluster) > 1 and optimize_routes:
                # Calculate route distance along the ordered cluster
                for j in range(len(cluster)-1):
                    route_distance += distance_matrix[cluster[j]][cluster[j+1]]
            else:
                route_distance = sum(intra_distances) if intra_distances else 0
            
            # Calculate COMPLETE route including Garage and TPA
            complete_route_distance = route_distance
            garage_to_first_distance = 0
            last_to_tpa_distance = 0
            
            if use_fixed_endpoints and start_idx is not None and end_idx is not None and len(cluster) > 0:
                # Add distance from Garage to first TPS
                garage_to_first_distance = distance_matrix[start_idx][cluster[0]]
                complete_route_distance += garage_to_first_distance
                
                # Add distance from last TPS to TPA
                last_to_tpa_distance = distance_matrix[cluster[-1]][end_idx]
                complete_route_distance += last_to_tpa_distance
            
            # Add to totals
            total_route_distance += route_distance
            total_complete_route_distance += complete_route_distance
            
            # Calculate between-node distances for this cluster
            node_distances = []
            for j in range(len(cluster)-1):
                node_distances.append(distance_matrix[cluster[j]][cluster[j+1]])
            
            cluster_metrics.append({
                'cluster_id': i+1,
                'size': len(cluster),
                'volume': cluster_volume,
                'volume_percentage': volume_percentage,
                'avg_distance': avg_intra_distance,
                'max_distance': max_intra_distance,
                'route_distance': route_distance,  # TPS only
                'garage_to_first': garage_to_first_distance,  # Garage to first TPS
                'last_to_tpa': last_to_tpa_distance,  # Last TPS to TPA
                'complete_route_distance': complete_route_distance,  # Total including Garage & TPA
                'node_distances': node_distances
            })
        
        # Calculate solution metrics
        solution_metrics = {
            'num_clusters': len(clusters),
            'total_distance': total_distance,
            'total_route_distance': total_route_distance,  # TPS only
            'total_complete_route_distance': total_complete_route_distance,  # Including Garage & TPA
            'avg_distance_per_cluster': total_distance / len(clusters) if clusters else 0,
            'avg_route_distance_per_cluster': total_route_distance / len(clusters) if clusters else 0,
            'avg_complete_route_distance_per_cluster': total_complete_route_distance / len(clusters) if clusters else 0,
            'cluster_metrics': cluster_metrics,
            'use_fixed_endpoints': use_fixed_endpoints,
            'garage_index': start_idx,
            'tpa_index': end_idx
        }
        
        return solution_metrics
    
    def calculate_cluster_centroids(self):
        """Calculate the centroid (middle point) coordinates for each cluster"""
        if not self.results or not self.data_info or not self.data_info.get('has_coords', False):
            return None
            
        centroids = []
        
        for cluster_info in self.results['clusters']:
            cluster_num = cluster_info['cluster_number']
            cluster_indices = cluster_info['tps_indices']
            
            # Extract coordinates for this cluster
            lats = [self.data_info['lat'][idx] for idx in cluster_indices]
            longs = [self.data_info['long'][idx] for idx in cluster_indices]
            
            # Calculate average coordinates (centroid)
            if lats and longs:
                avg_lat = sum(lats) / len(lats)
                avg_long = sum(longs) / len(longs)
                
                centroids.append({
                    'cluster': cluster_num,
                    'long': avg_long,
                    'lat': avg_lat
                })
        
        return centroids

    def cluster_centroids_by_fleet(self):
        """
        Group cluster centroids into fleet routes with balanced assignment (5-6 clusters per fleet)
        where each cluster represents one day of service in a week
        """
        # Calculate centroids if not already done
        centroids = self.calculate_cluster_centroids()
        
        if not centroids:
            messagebox.showerror("Error", "Tidak dapat mengelompokkan armada: Tidak ada data titik tengah")
            return None
        
        # Total clusters
        total_clusters = len(centroids)
        
        # Calculate required number of trucks based on 5-6 clusters per truck
        # For optimal distribution: aim for 6 clusters per truck, but never exceed it
        clusters_per_truck = 6
        num_trucks_needed = math.ceil(total_clusters / clusters_per_truck)
        
        # Get user-specified number of trucks
        user_trucks = self.num_trucks_var.get()
        
        # If user specified too few trucks, warn and adjust
        if user_trucks < num_trucks_needed:
            messagebox.showwarning("Peringatan", 
                                f"Jumlah armada ({user_trucks}) terlalu sedikit untuk melayani {total_clusters} cluster "
                                f"dengan batas 6 cluster per armada.\n\n"
                                f"Jumlah armada disesuaikan menjadi {num_trucks_needed}.")
            num_trucks = num_trucks_needed
        else:
            num_trucks = user_trucks
        
        # Create a distance matrix between centroids
        coords = np.array([[c['long'], c['lat']] for c in centroids])
        
        # Simple Euclidean distance matrix between all centroids
        from scipy.spatial.distance import cdist
        distance_matrix = cdist(coords, coords, 'euclidean')
        
        # Balanced assignment algorithm
        # Calculate ideal number of clusters per truck
        base_clusters_per_truck = total_clusters // num_trucks
        extra_clusters = total_clusters % num_trucks
        
        # Each truck gets base_clusters_per_truck, and some get one extra
        truck_capacities = [base_clusters_per_truck + (1 if i < extra_clusters else 0) 
                            for i in range(num_trucks)]
        
        # Start with empty assignments
        fleet_assignments = [[] for _ in range(num_trucks)]
        assigned_centroids = set()
        
        # Implement a greedy algorithm that tries to balance distance and cluster count
        # First, assign one "seed" cluster to each truck (preferably distant from each other)
        # For the first truck, pick a random centroid
        if len(centroids) > 0:
            first_centroid_idx = random.randint(0, len(centroids)-1)
            fleet_assignments[0].append(centroids[first_centroid_idx])
            assigned_centroids.add(first_centroid_idx)
            
            # For subsequent trucks, pick centroids that are farthest from already assigned ones
            for truck_idx in range(1, num_trucks):
                if len(assigned_centroids) >= len(centroids):
                    break
                    
                # Calculate average distance from each unassigned centroid to all assigned ones
                max_dist = -1
                farthest_idx = -1
                
                for i in range(len(centroids)):
                    if i not in assigned_centroids:
                        # Find minimum distance to any assigned centroid
                        min_dist_to_assigned = float('inf')
                        for j in assigned_centroids:
                            if distance_matrix[i][j] < min_dist_to_assigned:
                                min_dist_to_assigned = distance_matrix[i][j]
                        
                        # Keep the centroid with maximum minimum distance
                        if min_dist_to_assigned > max_dist:
                            max_dist = min_dist_to_assigned
                            farthest_idx = i
                
                if farthest_idx != -1:
                    fleet_assignments[truck_idx].append(centroids[farthest_idx])
                    assigned_centroids.add(farthest_idx)
        
        # Now assign remaining centroids to the nearest truck that has capacity
        for i in range(len(centroids)):
            if i not in assigned_centroids:
                # Find the best truck to assign this centroid to
                best_truck = -1
                best_dist = float('inf')
                
                for truck_idx in range(num_trucks):
                    # Skip trucks that have reached capacity
                    if len(fleet_assignments[truck_idx]) >= truck_capacities[truck_idx]:
                        continue
                    
                    # Calculate average distance to centroids already assigned to this truck
                    if not fleet_assignments[truck_idx]:  # If truck has no centroids yet
                        avg_dist = 0  # Prefer empty trucks
                    else:
                        total_dist = 0
                        for assigned_centroid in fleet_assignments[truck_idx]:
                            assigned_idx = centroids.index(assigned_centroid)
                            total_dist += distance_matrix[i][assigned_idx]
                        avg_dist = total_dist / len(fleet_assignments[truck_idx])
                    
                    if avg_dist < best_dist:
                        best_dist = avg_dist
                        best_truck = truck_idx
                
                # Assign to best truck if found, otherwise force assign to any truck with space
                if best_truck != -1:
                    fleet_assignments[best_truck].append(centroids[i])
                else:
                    # Find any truck with space
                    for truck_idx in range(num_trucks):
                        if len(fleet_assignments[truck_idx]) < truck_capacities[truck_idx]:
                            fleet_assignments[truck_idx].append(centroids[i])
                            break
        
        # Add fleet assignment to centroids
        for truck_idx, assigned_centroids in enumerate(fleet_assignments):
            for centroid in assigned_centroids:
                centroid['fleet'] = truck_idx + 1  # 1-based indexing for fleets
        
        # Group clusters by fleet
        fleet_clusters = {}
        for i in range(1, num_trucks + 1):
            fleet_clusters[i] = [c for c in centroids if c.get('fleet') == i]
        
        return {
            'centroids': centroids,
            'fleet_clusters': fleet_clusters,
            'num_trucks': num_trucks,
            'clusters_per_truck': truck_capacities
        }

    def process_clustering(self):
        """Process clustering using multi-start optimization"""
        if not self.file_path:
            messagebox.showerror("Error", "Pilih file Excel terlebih dahulu")
            return
        
        # Reset progress display
        self.update_progress(0, "Memulai proses clustering dengan multi-start...")
        
        try:
            # Clear tabs
            for tab in [self.debug_tab, self.data_view_tab, self.matrix_view_tab, 
                        self.summary_tab, self.viz_tab, self.details_tab, 
                        self.progress_tab, self.route_tab, self.map_tab, 
                        self.multi_start_tab]:
                for widget in tab.winfo_children():
                    widget.destroy()
            
            # Get parameters
            min_capacity = self.min_capacity_var.get()
            max_capacity = self.max_capacity_var.get()
            distance_penalty = self.distance_penalty_var.get()
            
            # VNS-GA parameters
            population_size = self.population_size_var.get()
            max_iterations = self.max_iterations_var.get()
            mutation_rate = self.mutation_rate_var.get()
            crossover_rate = self.crossover_rate_var.get()
            optimize_routes = self.optimize_routes_var.get()
            aco_ants = self.aco_ants_var.get()
            aco_iterations = self.aco_iterations_var.get()
            debug_mode = self.debug_var.get()
            
            # Multi-start parameters
            num_runs = self.num_runs_var.get()
            parallel = self.parallel_var.get()
            num_workers = self.num_workers_var.get()
            adaptive_stop = self.adaptive_stop_var.get()
            max_no_improvement = self.max_no_improvement_var.get()
            
            # Validate parameters
            if min_capacity <= 0 or max_capacity <= 0 or distance_penalty < 0:
                messagebox.showerror("Error", "Parameter harus lebih besar dari 0")
                return
                
            if min_capacity > max_capacity:
                messagebox.showerror("Error", "Kapasitas minimum harus lebih kecil dari kapasitas maksimum")
                return
            
            if num_runs <= 0:
                messagebox.showerror("Error", "Jumlah run multi-start harus lebih besar dari 0")
                return
                
            if num_workers <= 0:
                messagebox.showerror("Error", "Jumlah worker harus lebih besar dari 0")
                return
            
            self.status_var.set("Memproses data...")
            self.update_progress(2, "Membaca file Excel...")
            
            # Read data
            data_info, distance_matrix, debug_info = self.read_excel_data_direct(self.file_path)
            self.data_info = data_info  # Store for visualization
            
            # Display debug info
            debug_text = tk.Text(self.debug_tab, wrap="word")
            debug_text.insert("1.0", debug_info)
            debug_text.pack(fill="both", expand=True, padx=10, pady=10)
            
            if data_info is None or distance_matrix is None:
                messagebox.showerror("Error", "Gagal membaca data. Lihat tab Debug untuk informasi.")
                self.status_var.set("Error: Gagal membaca data")
                self.notebook.select(self.debug_tab)
                return
            
            # Update progress
            self.update_progress(5, "Data berhasil dibaca, mempersiapkan multi-start...")
            
            # Extract TPS info
            volumes = data_info['volumes']
            names = data_info['names']
            
            # Display data in Data View tab
            try:
                # Create DataFrame for TPS data
                tps_data = pd.DataFrame({
                    'Nama TPS': names,
                    'Volume (m³)': volumes
                })
                
                # Add coordinates if available
                if data_info.get('has_coords', False):
                    tps_data['Latitude'] = data_info['lat']
                    tps_data['Longitude'] = data_info['long']
                
                self.display_dataframe(tps_data, "Data TPS", self.data_view_tab)
            except Exception as e:
                error_label = ttk.Label(self.data_view_tab, text=f"Error menampilkan data: {str(e)}")
                error_label.pack(padx=10, pady=10)
            
            # Display matrix in Matrix View tab
            try:
                self.display_matrix(distance_matrix, names, "Matriks Jarak", self.matrix_view_tab)
            except Exception as e:
                error_label = ttk.Label(self.matrix_view_tab, text=f"Error menampilkan matriks: {str(e)}")
                error_label.pack(padx=10, pady=10)
            
            # Create progress display in multi-start tab 
            multi_start_frame = ttk.Frame(self.multi_start_tab)
            multi_start_frame.pack(fill="both", expand=True, padx=10, pady=10)
            
            ms_progress_frame = ttk.LabelFrame(multi_start_frame, text="Status Multi-Start")
            ms_progress_frame.pack(fill="x", padx=10, pady=10)
            
            ms_status_var = tk.StringVar(value="Memulai proses multi-start...")
            ms_status_label = ttk.Label(ms_progress_frame, textvariable=ms_status_var, font=("Arial", 10, "bold"))
            ms_status_label.pack(pady=5)
            
            ms_progress_var = tk.DoubleVar(value=0)
            ms_progress_bar = ttk.Progressbar(ms_progress_frame, variable=ms_progress_var, maximum=100, length=400)
            ms_progress_bar.pack(fill="x", padx=10, pady=5)
            
            ms_progress_text_var = tk.StringVar(value="0%")
            ms_progress_text = ttk.Label(ms_progress_frame, textvariable=ms_progress_text_var)
            ms_progress_text.pack()
            
            ms_time_var = tk.StringVar(value="Waktu: 00:00:00")
            ms_time_label = ttk.Label(ms_progress_frame, textvariable=ms_time_var)
            ms_time_label.pack(pady=2)
            
            ms_best_var = tk.StringVar(value="Belum ada solusi terbaik")
            ms_best_label = ttk.Label(ms_progress_frame, textvariable=ms_best_var, font=("Arial", 10))
            ms_best_label.pack(pady=5)
            
            # Create structured progress area in progress tab
            progress_text = tk.Text(self.progress_tab, wrap="word")
            progress_text.pack(fill="both", expand=True, padx=10, pady=10)
            
            # Add scrollbar
            scrollbar = ttk.Scrollbar(self.progress_tab, orient="vertical", command=progress_text.yview)
            scrollbar.pack(side="right", fill="y")
            progress_text.config(yscrollcommand=scrollbar.set)
            
            # Add heading
            progress_text.insert("end", "PROSES MULTI-START\n" + "="*50 + "\n\n")
            progress_text.insert("end", f"Waktu mulai: {time.strftime('%H:%M:%S')}\n")
            progress_text.insert("end", f"Jumlah run: {num_runs}\n")
            progress_text.insert("end", f"Paralel: {'Ya' if parallel else 'Tidak'}, Workers: {num_workers}\n")
            progress_text.insert("end", f"Adaptive stopping: {'Ya' if adaptive_stop else 'Tidak'}, " +
                               f"Max tanpa perbaikan: {max_no_improvement}\n\n")
            progress_text.insert("end", "Menjalankan multi-start...\n" + "-"*50 + "\n\n")
            
            # Show the multi-start tab
            self.notebook.select(self.multi_start_tab)
            
            # Run multi-start in a separate thread
            thread = threading.Thread(
                target=self._run_multi_start,
                args=(
                    distance_matrix, volumes, min_capacity, max_capacity,
                    population_size, max_iterations, mutation_rate, crossover_rate,
                    distance_penalty, optimize_routes, aco_ants, aco_iterations,
                    debug_mode, num_runs, parallel, num_workers, adaptive_stop, max_no_improvement,
                    progress_text, ms_status_var, ms_progress_var, ms_progress_text_var, 
                    ms_time_var, ms_best_var
                )
            )
            thread.daemon = True
            thread.start()
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            
            self.update_progress(0, f"Error: {str(e)}")
            messagebox.showerror("Error", f"Terjadi kesalahan: {str(e)}")
            
            # Display error details in debug tab
            debug_text = tk.Text(self.debug_tab, wrap="word")
            debug_text.insert("1.0", f"ERROR: {str(e)}\n\nDetail error:\n{error_details}")
            debug_text.pack(fill="both", expand=True, padx=10, pady=10)
            
            self.notebook.select(self.debug_tab)
    
    def _run_multi_start(self, distance_matrix, volumes, min_capacity, max_capacity,
                         population_size, max_iterations, mutation_rate, crossover_rate,
                         distance_penalty, optimize_routes, aco_ants, aco_iterations,
                         debug_mode, num_runs, parallel, num_workers, adaptive_stop, max_no_improvement,
                         progress_text, ms_status_var, ms_progress_var, ms_progress_text_var, 
                         ms_time_var, ms_best_var):
        """
        Run multi-start optimization for clustering
        """
        try:
            # Start timer
            self.start_time = time.time()
            
            # Setup timer update function
            def update_timer():
                if self.start_time is None:
                    return
                    
                elapsed = time.time() - self.start_time
                hours, remainder = divmod(elapsed, 3600)
                minutes, seconds = divmod(remainder, 60)
                time_str = f"Waktu: {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"
                
                ms_time_var.set(time_str)
                self.root.update_idletasks()
                
                # Schedule next update
                self.timer_id = self.root.after(1000, update_timer)
            
            # Start timer updates
            update_timer()
            
            # Initialize variables
            all_results = []
            best_solution = None
            best_fitness = float('inf')
            best_distance = float('inf')
            no_improvement_count = 0
            
            # Function to run a single optimization
            def run_single(seed):
                # Set random seed
                np.random.seed(seed)
                
                # Log progress
                if not parallel:
                    run_index = len(all_results) + 1
                    self.update_progress(
                        5 + (90 * run_index / num_runs),
                        f"Multi-start run {run_index}/{num_runs} dengan seed {seed}"
                    )
                    
                    progress_text.insert("end", f"Run {run_index}/{num_runs} (seed={seed}): Memulai clustering...\n")
                    progress_text.see("end")
                    self.root.update_idletasks()
                
                # Run clustering algorithm
                start_time = time.time()
                # PERBAIKAN: Ubah ini dari vns_ga_clustering menjadi ga_vns_clustering
                clusters, debug_info = self.ga_vns_clustering(
                    distance_matrix, volumes, min_capacity, max_capacity,
                    population_size=population_size, max_iterations=max_iterations,
                    mutation_rate=mutation_rate, crossover_rate=crossover_rate,
                    distance_penalty=distance_penalty, optimize_routes=optimize_routes,
                    aco_ants=aco_ants, aco_iterations=aco_iterations,
                    debug_mode=debug_mode
                )
                execution_time = time.time() - start_time
                
                # Calculate metrics
                metrics = self.calculate_solution_metrics(
                    clusters, distance_matrix, volumes, optimize_routes
                )
                
                # Check feasibility
                feasible = True
                for cluster in clusters:
                    cluster_vol = sum(volumes[idx] for idx in cluster)
                    if cluster_vol > max_capacity:
                        feasible = False
                        break
                
                # Get total distance
                total_distance_key = 'total_complete_route_distance' if 'total_complete_route_distance' in metrics else 'total_route_distance'
                distance = metrics.get(total_distance_key, float('inf'))
                
                # Create full result
                result = {
                    'seed': seed,
                    'clusters': clusters,
                    'metrics': metrics,
                    'execution_time': execution_time,
                    'feasible': feasible,
                    'distance': distance,
                    'num_clusters': len(clusters),
                    'debug_info': debug_info
                }
                
                return result
            
            if parallel and num_runs > 1:
                # Run in parallel
                progress_text.insert("end", f"Memulai {num_runs} run secara paralel dengan {num_workers} workers...\n")
                progress_text.see("end")
                ms_status_var.set(f"Menjalankan {num_runs} run secara paralel...")
                self.root.update_idletasks()
                
                self.update_progress(10, f"Menjalankan {num_runs} run secara paralel...")
                
                # Determine executor based on complexity
                executor_class = ThreadPoolExecutor  # Default 
                executor_name = "ThreadPoolExecutor"
                
                if distance_matrix.shape[0] > 100:
                    # For large problems, use ProcessPoolExecutor
                    try:
                        from concurrent.futures import ProcessPoolExecutor
                        executor_class = ProcessPoolExecutor
                        executor_name = "ProcessPoolExecutor"
                    except:
                        # Fall back to ThreadPoolExecutor if ProcessPoolExecutor is not available
                        pass
                
                progress_text.insert("end", f"Menggunakan {executor_name} dengan {num_workers} workers\n\n")
                
                with executor_class(max_workers=num_workers) as executor:
                    # Submit all tasks
                    futures = {executor.submit(run_single, seed): seed for seed in range(num_runs)}
                    
                    # Process results as they complete
                    for i, future in enumerate(as_completed(futures)):
                        seed = futures[future]
                        try:
                            result = future.result()
                            all_results.append(result)
                            
                            # Update progress
                            progress_percentage = 10 + (85 * (i + 1) / num_runs)
                            self.update_progress(
                                progress_percentage,
                                f"Selesai run {i+1}/{num_runs} (seed={seed}): {result['num_clusters']} cluster, jarak={result['distance']:.2f}"
                            )
                            
                            # Update multi-start progress
                            ms_progress_var.set((i+1) * 100 / num_runs)
                            ms_progress_text_var.set(f"{int((i+1) * 100 / num_runs)}%")
                            ms_status_var.set(f"Run {i+1}/{num_runs} (seed={seed}): {result['num_clusters']} cluster, jarak={result['distance']:.2f}")
                            
                            # Log result
                            feasible_text = "FEASIBLE" if result['feasible'] else "TIDAK FEASIBLE"
                            progress_text.insert("end", f"Run {i+1}/{num_runs} (seed={seed}): {result['num_clusters']} cluster, " +
                                              f"jarak={result['distance']:.2f}, {feasible_text}, waktu={result['execution_time']:.2f}s\n")
                            progress_text.see("end")
                            
                            # Update best solution
                            if ((result['feasible'] and result['distance'] < best_distance) or 
                                (best_solution is None and result['feasible'])):
                                best_solution = result
                                best_distance = result['distance']
                                ms_best_var.set(f"Solusi terbaik: Seed {seed}, {result['num_clusters']} cluster, jarak={result['distance']:.2f}")
                                progress_text.insert("end", f"✓ Run {i+1} (seed={seed}) menghasilkan solusi terbaik baru: {result['distance']:.2f}\n")
                                progress_text.see("end")
                            
                            self.root.update_idletasks()
                        except Exception as e:
                            progress_text.insert("end", f"Error pada run {i+1} (seed={seed}): {str(e)}\n")
                            progress_text.see("end")
                            self.root.update_idletasks()
            else:
                # Run sequentially
                for run in range(num_runs):
                    # Update status
                    ms_status_var.set(f"Menjalankan run {run+1}/{num_runs} (seed={run})...")
                    
                    # Run clustering
                    result = run_single(run)
                    all_results.append(result)
                    
                    # Update progress display
                    progress_percentage = 10 + (85 * (run + 1) / num_runs)
                    self.update_progress(
                        progress_percentage,
                        f"Selesai run {run+1}/{num_runs} (seed={run}): {result['num_clusters']} cluster, jarak={result['distance']:.2f}"
                    )
                    
                    ms_progress_var.set((run+1) * 100 / num_runs)
                    ms_progress_text_var.set(f"{int((run+1) * 100 / num_runs)}%")
                    
                    # Log result
                    feasible_text = "FEASIBLE" if result['feasible'] else "TIDAK FEASIBLE"
                    progress_text.insert("end", f"Run {run+1}/{num_runs} (seed={run}): {result['num_clusters']} cluster, " +
                                      f"jarak={result['distance']:.2f}, {feasible_text}, waktu={result['execution_time']:.2f}s\n")
                    progress_text.see("end")
                    
                    # Update best solution
                    if ((result['feasible'] and result['distance'] < best_distance) or 
                        (best_solution is None and result['feasible'])):
                        best_solution = result
                        best_distance = result['distance']
                        no_improvement_count = 0
                        ms_best_var.set(f"Solusi terbaik: Seed {run}, {result['num_clusters']} cluster, jarak={result['distance']:.2f}")
                        progress_text.insert("end", f"✓ Run {run+1} (seed={run}) menghasilkan solusi terbaik baru: {result['distance']:.2f}\n")
                        progress_text.see("end")
                    else:
                        no_improvement_count += 1
                    
                    self.root.update_idletasks()
                    
                    # Check adaptive stopping
                    if adaptive_stop and no_improvement_count >= max_no_improvement and run >= num_runs // 3:
                        progress_text.insert("end", f"\nADAPTIVE STOP: Berhenti setelah {no_improvement_count} run tanpa perbaikan\n")
                        progress_text.see("end")
                        break
            
            # Stop timer updates
            if self.timer_id:
                self.root.after_cancel(self.timer_id)
                self.timer_id = None
            
            # Store results for later use
            self.multi_start_results = all_results
            
            # If no solution was found, try to find best non-feasible solution
            if best_solution is None and all_results:
                # Sort by distance
                all_results.sort(key=lambda x: x.get('distance', float('inf')))
                best_solution = all_results[0]
                progress_text.insert("end", f"\nPERINGATAN: Tidak ada solusi feasible. " +
                                   f"Menggunakan solusi terbaik yang tidak feasible.\n")
                progress_text.see("end")
            
            # Display summary
            total_elapsed = time.time() - self.start_time
            ms_status_var.set(f"Multi-start selesai. Waktu total: {total_elapsed:.2f} detik")
            
            progress_text.insert("end", f"\nMulti-start selesai pada: {time.strftime('%H:%M:%S')}\n")
            progress_text.insert("end", f"Total waktu proses: {total_elapsed:.2f} detik\n")
            progress_text.insert("end", "Menampilkan hasil di tab Multi-Start Results...\n")
            progress_text.see("end")
            
            # Visualize multi-start results
            self.visualize_multi_start_results(all_results)
            
            if best_solution:
                # Set up results for display
                vnsga_clusters = best_solution['clusters']
                vnsga_metrics = best_solution['metrics']
                debug_tracking = best_solution.get('debug_info', [])
                
                # Set results for application
                self.results = {
                    'algorithm': f'Multi-Start ({len(all_results)}/{num_runs} run) VNS-GA + ACO',
                    'total_tps': len(volumes),
                    'total_volume': sum(volumes),
                    'num_clusters': len(vnsga_clusters),
                    'clusters': [],
                    'metrics': vnsga_metrics,
                    'execution_time': best_solution['execution_time'],
                    'optimize_routes': optimize_routes,
                    'multi_start_info': {
                        'best_seed': best_solution['seed'],
                        'num_runs': len(all_results),
                        'best_distance': best_solution['distance'],
                        'best_feasible': best_solution['feasible'],
                        'total_process_time': total_elapsed
                    }
                }
                
                # Process cluster details
                for i, cluster in enumerate(vnsga_clusters, 1):
                    cluster_volumes = [volumes[idx] for idx in cluster]
                    
                    # Make sure we get names correctly
                    tps_names = []
                    if 'names' in self.data_info:
                        tps_names = [self.data_info['names'][idx] for idx in cluster]
                    else:
                        tps_names = [f"TPS-{idx}" for idx in cluster]
                    
                    cluster_info = {
                        'cluster_number': i,
                        'num_tps': len(cluster),
                        'total_volume': sum(cluster_volumes),
                        'tps_names': tps_names,
                        'tps_volumes': cluster_volumes,
                        'tps_indices': cluster
                    }
                    self.results['clusters'].append(cluster_info)
                
                # Add multi-start info to debug
                progress_text.insert("end", f"\nMulti-start selesai. Solusi terbaik dari seed {best_solution['seed']}: " +
                                   f"{best_solution['num_clusters']} cluster, jarak={best_solution['distance']:.2f}\n")
                progress_text.see("end")
                
                # Visualize algorithm progress
                try:
                    self.app.visualize_algorithm_progress(debug_tracking)
                except Exception as e:
                    progress_text.insert("end", f"Error visualisasi algoritma: {str(e)}\n")
                
                # Display results
                self.update_progress(95, "Mempersiapkan tampilan hasil...")
                
                try:
                    self.display_results()
                except Exception as e:
                    progress_text.insert("end", f"Error display results: {str(e)}\n")
                
                # Create map visualization
                try:
                    self.create_map_visualization()
                except Exception as e:
                    progress_text.insert("end", f"Error visualisasi peta: {str(e)}\n")
                
                # Visualize routes
                try:
                    self.visualize_routes()
                except Exception as e:
                    progress_text.insert("end", f"Error visualisasi rute: {str(e)}\n")
                
                # Visualize titik tengah
                try:
                    self.display_centroids_tab()
                except Exception as e:
                    progress_text.insert("end", f"Error visualisasi titik tengah: {str(e)}\n")

                # Update progress
                self.update_progress(100, "Selesai!")
                ms_status_var.set(f"Solusi terbaik diterapkan (seed {best_solution['seed']})")
                
                # Update status bar
                self.status_var.set(f"Proses selesai. Solusi terbaik (seed {best_solution['seed']}): " +
                                   f"{self.results['num_clusters']} cluster, jarak {best_solution['distance']:.2f}")
                
                # Open multi-start tab
                self.notebook.select(self.multi_start_tab)
            else:
                self.update_progress(100, "Tidak ada solusi yang ditemukan")
                ms_status_var.set("Error: Tidak ada solusi yang ditemukan")
                self.status_var.set("Error: Tidak ada solusi yang ditemukan dari semua run")
                progress_text.insert("end", f"\nERROR: Tidak ada solusi yang ditemukan dari semua run\n")
                progress_text.see("end")
        
        except Exception as e:
            # Stop timer if there was an error
            if self.timer_id:
                self.root.after_cancel(self.timer_id)
                self.timer_id = None
            
            trace = traceback.format_exc()
            ms_status_var.set(f"Error: {str(e)}")
            self.update_progress(0, f"Error multi-start: {str(e)}")
            
            # Display error details
            debug_text = tk.Text(self.debug_tab, wrap="word")
            debug_text.insert("1.0", f"ERROR MULTI-START: {str(e)}\n\nDetail error:\n{trace}")
            debug_text.pack(fill="both", expand=True, padx=10, pady=10)
            
            self.notebook.select(self.debug_tab)
    
    def visualize_multi_start_results(self, results):
        """
        Visualize results from multi-start optimization with improved scrollable frames
        
        Parameters:
        -----------
        results : list
            List of results from different runs
        """
        if not results:
            # Show message if no results
            ttk.Label(
                self.multi_start_tab, 
                text="Tidak ada hasil multi-start yang ditemukan.",
                font=("Arial", 12, "bold"),
                foreground="red"
            ).pack(expand=True, pady=20)
            return
        
        # Clear tab
        for widget in self.multi_start_tab.winfo_children():
            widget.destroy()
        
        # Create scrollable frame using the helper function
        scrollable_frame = self.create_scrollable_frame(self.multi_start_tab)
        
        # Title
        ttk.Label(
            scrollable_frame, 
            text=f"Hasil Multi-Start ({len(results)} run)", 
            font=("Arial", 14, "bold")
        ).pack(pady=5)
        
        # Best solution section
        best_frame = ttk.LabelFrame(scrollable_frame, text="Solusi Terbaik")
        best_frame.pack(fill="x", expand=True, padx=5, pady=5)
        
        # Find best solution
        feasible_results = [r for r in results if r.get('feasible', False)]
        if feasible_results:
            feasible_results.sort(key=lambda x: x.get('distance', float('inf')))
            best_result = feasible_results[0]
        else:
            # If no feasible solutions, find best by distance
            results.sort(key=lambda x: x.get('distance', float('inf')))
            best_result = results[0]
        
        # Display best solution info
        best_info_text = (
            f"Seed: {best_result.get('seed', '-')}\n"
            f"Jumlah Cluster: {best_result.get('num_clusters', 0)}\n"
            f"Jarak Total: {best_result.get('distance', float('inf')):.2f}\n"
            f"Feasible: {'Ya' if best_result.get('feasible', False) else 'Tidak'}\n"
            f"Waktu Eksekusi: {best_result.get('execution_time', 0):.2f} detik"
        )
        
        ttk.Label(best_frame, text=best_info_text, justify="left").pack(anchor="w", padx=10, pady=5)
        
        # Comparison table
        table_frame = ttk.LabelFrame(scrollable_frame, text="Perbandingan Hasil")
        table_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Create treeview container with horizontal scrollbar
        tree_container = ttk.Frame(table_frame)
        tree_container.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Create treeview for table
        columns = ("Run", "Seed", "Clusters", "Distance", "Feasible", "Time")
        tree = ttk.Treeview(tree_container, columns=columns, show="headings", height=10)
        
        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, width=80, anchor="center")
        
        # Adjust column widths
        tree.column("Run", width=40)
        tree.column("Seed", width=40)
        tree.column("Clusters", width=60)
        tree.column("Distance", width=100)
        tree.column("Feasible", width=60)
        tree.column("Time", width=60)
        
        # Add scrollbars to treeview
        tree_v_scrollbar = ttk.Scrollbar(tree_container, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=tree_v_scrollbar.set)
        tree_v_scrollbar.pack(side="right", fill="y")
        
        tree_h_scrollbar = ttk.Scrollbar(tree_container, orient="horizontal", command=tree.xview)
        tree.configure(xscrollcommand=tree_h_scrollbar.set)
        tree_h_scrollbar.pack(side="bottom", fill="x")
        
        tree.pack(fill="both", expand=True)
        
        # Add data to table
        for i, result in enumerate(results):
            feasible_text = "Ya" if result.get('feasible', False) else "Tidak"
            tree.insert(
                "", "end", 
                values=(
                    i+1, 
                    result.get('seed', '-'), 
                    result.get('num_clusters', 0),
                    f"{result.get('distance', float('inf')):.2f}",
                    feasible_text,
                    f"{result.get('execution_time', 0):.2f}s"
                )
            )
        
        # Visualize with scatter plot
        fig1 = plt.Figure(figsize=(10, 5), dpi=100)
        ax1 = fig1.add_subplot(111)
        
        # Extract data for visualization
        seeds = [r.get('seed', i) for i, r in enumerate(results)]
        distances = [r.get('distance', float('inf')) for r in results]
        num_clusters = [r.get('num_clusters', 0) for r in results]
        feasible = [r.get('feasible', False) for r in results]
        
        # Colors based on feasibility
        colors = ['green' if f else 'red' for f in feasible]
        
        # Create scatter plot
        scatter = ax1.scatter(num_clusters, distances, c=colors, alpha=0.7)
        
        # Add seed labels
        for i, seed in enumerate(seeds):
            ax1.annotate(str(seed), (num_clusters[i], distances[i]), fontsize=8)
        
        # Add legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='Feasible'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Tidak Feasible')
        ]
        ax1.legend(handles=legend_elements)
        
        # Labels and title
        ax1.set_xlabel('Jumlah Cluster')
        ax1.set_ylabel('Total Distance')
        ax1.set_title('Perbandingan Hasil Multi-Start')
        ax1.grid(True)
        
        # Adjust layout
        fig1.tight_layout()
        
        # Add to UI
        chart_frame1 = ttk.Frame(scrollable_frame)
        chart_frame1.pack(fill="both", expand=True, padx=5, pady=5)
        
        canvas1 = FigureCanvasTkAgg(fig1, master=chart_frame1)
        canvas1.draw()
        canvas1.get_tk_widget().pack(fill="both", expand=True)
        
        # Bar chart for feasible results
        feasible_indices = [i for i, f in enumerate(feasible) if f]
        
        if feasible_indices:
            fig2 = plt.Figure(figsize=(10, 5), dpi=100)
            ax2 = fig2.add_subplot(111)
            
            # Filter for feasible solutions
            feasible_seeds = [seeds[i] for i in feasible_indices]
            feasible_distances = [distances[i] for i in feasible_indices]
            
            # Sort by distance
            sorted_indices = np.argsort(feasible_distances)
            sorted_seeds = [feasible_seeds[i] for i in sorted_indices]
            sorted_distances = [feasible_distances[i] for i in sorted_indices]
            
            # Create bar chart
            bars = ax2.bar(range(len(sorted_seeds)), sorted_distances, color='skyblue')
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax2.annotate(f'{height:.2f}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom')
            
            # Set x-ticks as seeds
            ax2.set_xticks(range(len(sorted_seeds)))
            ax2.set_xticklabels([f"Seed {s}" for s in sorted_seeds])
            
            # Add best solution marker
            best_idx = sorted_indices[0]
            ax2.axhline(y=sorted_distances[0], color='r', linestyle='--', 
                    label=f'Best: {sorted_distances[0]:.2f} (Seed {sorted_seeds[0]})')
            
            # Labels and title
            ax2.set_xlabel('Run (Seeds)')
            ax2.set_ylabel('Total Distance')
            ax2.set_title('Perbandingan Jarak untuk Solusi Feasible')
            ax2.legend()
            ax2.grid(True, axis='y')
            
            # Adjust layout
            fig2.tight_layout()
            
            # Add to UI
            chart_frame2 = ttk.Frame(scrollable_frame)
            chart_frame2.pack(fill="both", expand=True, padx=5, pady=5)
            
            canvas2 = FigureCanvasTkAgg(fig2, master=chart_frame2)
            canvas2.draw()
            canvas2.get_tk_widget().pack(fill="both", expand=True)
        
        # Statistics
        stats_frame = ttk.LabelFrame(scrollable_frame, text="Statistik Multi-Start")
        stats_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Calculate statistics
        all_stats_text = (
            f"Total run: {len(results)}\n"
            f"Run feasible: {sum(feasible)}\n"
            f"Run tidak feasible: {len(results) - sum(feasible)}\n"
            f"Rentang jumlah cluster: {min(num_clusters)} - {max(num_clusters)}\n"
            f"Waktu total: {sum([r.get('execution_time', 0) for r in results]):.2f} detik\n"
            f"Waktu rata-rata per run: {sum([r.get('execution_time', 0) for r in results])/len(results):.2f} detik"
        )
        
        ttk.Label(stats_frame, text=all_stats_text, justify="left").pack(anchor="w", padx=10, pady=5)
        
        # Statistics for feasible results
        if feasible_indices:
            feasible_distances = [distances[i] for i in feasible_indices]
            
            feasible_stats_text = (
                f"\nStatistik untuk solusi feasible:\n"
                f"Jarak minimum: {min(feasible_distances):.2f}\n"
                f"Jarak maksimum: {max(feasible_distances):.2f}\n"
                f"Jarak rata-rata: {sum(feasible_distances)/len(feasible_distances):.2f}\n"
                f"Standar deviasi: {np.std(feasible_distances):.2f}\n"
                f"Koefisien variasi: {np.std(feasible_distances)/np.mean(feasible_distances)*100:.2f}%"
            )
            
            ttk.Label(stats_frame, text=feasible_stats_text, justify="left").pack(anchor="w", padx=10, pady=5)
        
        # Add export button
        button_frame = ttk.Frame(scrollable_frame)
        button_frame.pack(fill="x", padx=5, pady=10)
        
        ttk.Button(
            button_frame,
            text="Export Hasil Multi-Start ke Excel",
            command=self.export_multi_start_results
        ).pack(side="left", padx=5)
        
        # Add another button for exporting all results
        ttk.Button(
            button_frame,
            text="Export Semua Hasil ke Excel",
            command=self.export_results
        ).pack(side="left", padx=5)
    


    def export_multi_start_results(self):
        """Export multi-start results to Excel file"""
        if not self.multi_start_results:
            messagebox.showerror("Error", "Tidak ada hasil multi-start untuk diekspor")
            return
            
        file_path = filedialog.asksaveasfilename(
            defaultextension=".xlsx",
            filetypes=[("Excel files", "*.xlsx")],
            title="Simpan Hasil Multi-Start"
        )
        
        if not file_path:
            return
            
        try:
            # Create DataFrame for multi-start results
            results_data = []
            for i, result in enumerate(self.multi_start_results):
                results_data.append({
                    'Run': i+1,
                    'Seed': result.get('seed', i),
                    'Jumlah Cluster': result.get('num_clusters', 0),
                    'Jarak Total': result.get('distance', float('inf')),
                    'Feasible': 'Ya' if result.get('feasible', False) else 'Tidak',
                    'Waktu Eksekusi (s)': result.get('execution_time', 0),
                })
            
            results_df = pd.DataFrame(results_data)
            
            # Create DataFrame for statistics
            feasible_results = [r for r in self.multi_start_results if r.get('feasible', False)]
            feasible_distances = [r.get('distance', float('inf')) for r in feasible_results]
            
            stats_data = [{
                'Total Run': len(self.multi_start_results),
                'Run Feasible': len(feasible_results),
                'Run Tidak Feasible': len(self.multi_start_results) - len(feasible_results),
                'Jarak Minimum': min(feasible_distances) if feasible_distances else float('inf'),
                'Jarak Maksimum': max(feasible_distances) if feasible_distances else float('inf'),
                'Jarak Rata-rata': sum(feasible_distances)/len(feasible_distances) if feasible_distances else float('inf'),
                'Standar Deviasi': np.std(feasible_distances) if len(feasible_distances) > 1 else 0,
                'Koefisien Variasi (%)': np.std(feasible_distances)/np.mean(feasible_distances)*100 if len(feasible_distances) > 1 else 0,
                'Waktu Total (s)': sum([r.get('execution_time', 0) for r in self.multi_start_results]),
                'Waktu Rata-rata (s)': sum([r.get('execution_time', 0) for r in self.multi_start_results])/len(self.multi_start_results)
            }]
            
            stats_df = pd.DataFrame(stats_data)
            
            # Create Excel writer
            with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
                results_df.to_excel(writer, sheet_name='Hasil Multi-Start', index=False)
                stats_df.to_excel(writer, sheet_name='Statistik', index=False)
                
                # Add best solution details if available
                if hasattr(self, 'results') and self.results:
                    best_clusters_data = []
                    
                    for cluster in self.results['clusters']:
                        best_clusters_data.append({
                            'Cluster': cluster['cluster_number'],
                            'Jumlah TPS': cluster['num_tps'],
                            'Volume Total (m³)': cluster['total_volume'],
                            'Jarak Rute': 0  # Placeholder, can be filled with actual route distances if available
                        })
                    
                    best_df = pd.DataFrame(best_clusters_data)
                    best_df.to_excel(writer, sheet_name='Solusi Terbaik', index=False)
            
            messagebox.showinfo("Sukses", f"Hasil multi-start berhasil disimpan ke {file_path}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Gagal menyimpan file: {str(e)}")
    
    def export_to_kml(self):
        """Export the cluster data to KML/KMZ format for viewing in Google Earth"""
        if not self.results or not self.data_info or not self.data_info.get('has_coords', False):
            messagebox.showerror("Error", "Tidak dapat mengekspor KML: Tidak ada data koordinat")
            return
        
        # Ask user if they want KML or KMZ
        format_choice = messagebox.askyesno(
            "Format Export", 
            "Apakah Anda ingin menyimpan dalam format KMZ (terkompresi)?\n\n"
            "Pilih 'Ya' untuk KMZ atau 'Tidak' untuk KML"
        )
        
        file_ext = ".kmz" if format_choice else ".kml"
        
        file_path = filedialog.asksaveasfilename(
            defaultextension=file_ext,
            filetypes=[("KMZ files", "*.kmz"), ("KML files", "*.kml")],
            title="Simpan Peta sebagai KML/KMZ"
        )
        
        if not file_path:
            return
        
        try:
            # Try to import simplekml - needs to be installed with pip install simplekml
            import simplekml
            
            # Create a new KML document
            kml = simplekml.Kml()
            
            # Check if using fixed endpoints
            use_fixed_endpoints = self.use_fixed_endpoints_var.get()
            start_point_name = self.start_point_var.get() if use_fixed_endpoints else None
            end_point_name = self.end_point_var.get() if use_fixed_endpoints else None
            
            # Find indices of start and end points
            start_idx = None
            end_idx = None
            
            if use_fixed_endpoints and 'names' in self.data_info:
                names = self.data_info['names']
                
                # Find start point (Garage)
                for i, name in enumerate(names):
                    if isinstance(name, str) and start_point_name.lower() in name.lower():
                        start_idx = i
                        break
                
                # Find end point (TPA)
                for i, name in enumerate(names):
                    if isinstance(name, str) and end_point_name.lower() in name.lower():
                        end_idx = i
                        break
                
                # Add Garage marker if found
                if start_idx is not None:
                    garage_lat = self.data_info['lat'][start_idx]
                    garage_long = self.data_info['long'][start_idx]
                    
                    garage_point = kml.newpoint(
                        name="Garasi (Titik Awal)",
                        description=f"Garasi: {start_point_name}",
                        coords=[(garage_long, garage_lat)]
                    )
                    garage_point.style.iconstyle.icon.href = 'http://maps.google.com/mapfiles/kml/shapes/placemark_square.png'
                    garage_point.style.iconstyle.color = simplekml.Color.blue
                    garage_point.style.iconstyle.scale = 1.2
                
                # Add TPA marker if found
                if end_idx is not None:
                    tpa_lat = self.data_info['lat'][end_idx]
                    tpa_long = self.data_info['long'][end_idx]
                    
                    tpa_point = kml.newpoint(
                        name="TPA (Titik Akhir)",
                        description=f"TPA: {end_point_name}",
                        coords=[(tpa_long, tpa_lat)]
                    )
                    tpa_point.style.iconstyle.icon.href = 'http://maps.google.com/mapfiles/kml/shapes/garbage.png'
                    tpa_point.style.iconstyle.color = simplekml.Color.red
                    tpa_point.style.iconstyle.scale = 1.2
            
            # Define a set of distinct colors for clusters
            colors = [
                simplekml.Color.red,
                simplekml.Color.blue,
                simplekml.Color.green,
                simplekml.Color.purple,
                simplekml.Color.yellow,
                simplekml.Color.cyan,
                simplekml.Color.pink,
                simplekml.Color.orange,
                simplekml.Color.gray,
                simplekml.Color.black
            ]
            
            # Create folder for clusters
            clusters_folder = kml.newfolder(name="Clusters")
            
            # Add each cluster with routes
            for i, cluster in enumerate(self.results['clusters']):
                cluster_num = cluster['cluster_number']
                color = colors[i % len(colors)]
                
                # Create folder for this cluster
                cluster_folder = clusters_folder.newfolder(
                    name=f"Cluster {cluster_num} - {len(cluster['tps_indices'])} TPS, {cluster['total_volume']:.2f} m³"
                )
                
                # Extract cluster data
                cluster_indices = cluster['tps_indices']
                cluster_lats = [self.data_info['lat'][idx] for idx in cluster_indices]
                cluster_longs = [self.data_info['long'][idx] for idx in cluster_indices]
                names = [self.data_info['names'][idx] for idx in cluster_indices]
                volumes = [self.data_info['volumes'][idx] for idx in cluster_indices]
                
                # Add route line
                if len(cluster_indices) > 1:
                    route_coords = []
                    
                    # Add Garage as first point if using fixed endpoints
                    if use_fixed_endpoints and start_idx is not None:
                        route_coords.append((self.data_info['long'][start_idx], self.data_info['lat'][start_idx]))
                    
                    # Add all TPS points
                    for j in range(len(cluster_indices)):
                        route_coords.append((cluster_longs[j], cluster_lats[j]))
                    
                    # Add TPA as last point if using fixed endpoints
                    if use_fixed_endpoints and end_idx is not None:
                        route_coords.append((self.data_info['long'][end_idx], self.data_info['lat'][end_idx]))
                    
                    # Create route line
                    route = cluster_folder.newlinestring(
                        name=f"Route Cluster {cluster_num}",
                        description=f"Rute Cluster {cluster_num}: {len(cluster_indices)} TPS, {cluster['total_volume']:.2f} m³",
                        coords=route_coords
                    )
                    
                    # Style the route line
                    route.style.linestyle.color = color
                    route.style.linestyle.width = 4
                
                # Add markers for each TPS in this cluster
                for j, (lat, lng, name, vol, idx) in enumerate(zip(cluster_lats, cluster_longs, names, volumes, cluster_indices)):
                    # Create marker for this TPS
                    point = cluster_folder.newpoint(
                        name=f"({j+1}) {name}",
                        description=f"""
                        <b>Cluster {cluster_num}</b><br>
                        <b>TPS:</b> {name}<br>
                        <b>Index:</b> {idx}<br>
                        <b>Volume:</b> {vol:.2f} m³<br>
                        <b>Urutan Rute:</b> {j+1}/{len(cluster_indices)}
                        """,
                        coords=[(lng, lat)]
                    )
                    
                    # Style the marker
                    point.style.iconstyle.color = color
                    point.style.iconstyle.icon.href = 'http://maps.google.com/mapfiles/kml/shapes/shaded_dot.png'
                    
                    # Make first and last TPS markers larger
                    if j == 0 or j == len(cluster_indices) - 1:
                        point.style.iconstyle.scale = 1.2
                        if j == 0:
                            point.name = f"(Start) {name}"
                        else:
                            point.name = f"(End) {name}"
            
            # Save the KML file
            if file_path.lower().endswith('.kmz'):
                kml.savekmz(file_path)
            else:
                kml.save(file_path)
            
            messagebox.showinfo("Sukses", f"Peta berhasil disimpan sebagai {file_path}")
            
            # Ask if user wants to open the file
            if messagebox.askyesno("Buka File", "Apakah Anda ingin membuka file KML/KMZ sekarang?"):
                webbrowser.open('file://' + os.path.abspath(file_path))
                
        except ImportError:
            # If simplekml is not installed
            messagebox.showerror("Error", "Modul 'simplekml' tidak terinstall. Silakan install dengan menjalankan 'pip install simplekml'")
        except Exception as e:
            messagebox.showerror("Error", f"Gagal menyimpan file KML/KMZ: {str(e)}")
            
    def create_map_visualization(self):
        """Create a folium map to visualize clusters with different colors"""
        if not self.results or not self.data_info or not self.data_info.get('has_coords', False):
            ttk.Label(self.map_tab, text="Tidak dapat menampilkan peta: Tidak ada data koordinat").pack(padx=10, pady=10)
            return False
        
        # Clear map tab
        for widget in self.map_tab.winfo_children():
            widget.destroy()
        
        # Define a set of distinct colors for clusters
        colors = [
            'red', 'blue', 'green', 'purple', 'orange', 'darkred', 
            'darkblue', 'darkgreen', 'cadetblue', 'darkpurple', 'pink', 
            'lightblue', 'lightgreen', 'gray', 'black', 'lightgray'
        ]
        
        # Calculate center point of all TPS for map center
        lats = self.data_info['lat']
        longs = self.data_info['long']
        center_lat = np.mean(lats)
        center_long = np.mean(longs)
        
        # Check if using fixed endpoints
        use_fixed_endpoints = self.use_fixed_endpoints_var.get()
        start_point_name = self.start_point_var.get() if use_fixed_endpoints else None
        end_point_name = self.end_point_var.get() if use_fixed_endpoints else None

        # Find indices of start and end points
        start_idx = None
        end_idx = None

        if use_fixed_endpoints and 'names' in self.data_info:
            names = self.data_info['names']
            
            # Find start point (Garage)
            if start_point_name in names:
                # Exact match
                for i, name in enumerate(names):
                    if name == start_point_name:
                        start_idx = i
                        break
            else:
                # Partial match
                for i, name in enumerate(names):
                    if start_point_name.lower() in name.lower():
                        start_idx = i
                        break
            
            # Find end point (TPA)
            if end_point_name in names:
                # Exact match
                for i, name in enumerate(names):
                    if name == end_point_name:
                        end_idx = i
                        break
            else:
                # Partial match
                for i, name in enumerate(names):
                    if end_point_name.lower() in name.lower():
                        end_idx = i
                        break

        # Create a new map centered at the average location
        m = folium.Map(location=[center_lat, center_long], zoom_start=13)
        
        # Add markers for Garage and TPA if using fixed endpoints
        if use_fixed_endpoints and start_idx is not None and end_idx is not None:
            # Add Garage marker
            garage_lat = self.data_info['lat'][start_idx]
            garage_long = self.data_info['long'][start_idx]
            folium.Marker(
                location=[garage_lat, garage_long],
                popup=folium.Popup(f"Garasi: {start_point_name}", max_width=300),
                tooltip="Garasi (Titik Awal)",
                icon=folium.Icon(color='blue', icon='home')
            ).add_to(m)
            
            # Add TPA marker
            tpa_lat = self.data_info['lat'][end_idx]
            tpa_long = self.data_info['long'][end_idx]
            folium.Marker(
                location=[tpa_lat, tpa_long],
                popup=folium.Popup(f"TPA: {end_point_name}", max_width=300),
                tooltip="TPA (Titik Akhir)",
                icon=folium.Icon(color='red', icon='trash')
            ).add_to(m)

        # Add each cluster with a different color
        for i, cluster in enumerate(self.results['clusters']):
            cluster_num = cluster['cluster_number']
            color = colors[i % len(colors)]  # Cycle through colors
            
            # Extract cluster data
            cluster_indices = cluster['tps_indices']
            cluster_lats = [lats[idx] for idx in cluster_indices]
            cluster_longs = [longs[idx] for idx in cluster_indices]
            names = [self.data_info['names'][idx] for idx in cluster_indices]
            volumes = [self.data_info['volumes'][idx] for idx in cluster_indices]
            
            # Add markers for each TPS in this cluster
            for j, (lat, lng, name, vol, idx) in enumerate(zip(cluster_lats, cluster_longs, names, volumes, cluster_indices)):
                # Create a popup with TPS information
                popup_text = f"""
                <b>Cluster {cluster_num}</b><br>
                <b>TPS:</b> {name}<br>
                <b>Index:</b> {idx}<br>
                <b>Volume:</b> {vol:.2f} m³<br>
                <b>Urutan Rute:</b> {j+1}/{len(cluster_indices)}
                """
                
                # Add marker with popup
                folium.Marker(
                    location=[lat, lng],
                    popup=folium.Popup(popup_text, max_width=300),
                    tooltip=f"Cluster {cluster_num}: {name}",
                    icon=folium.Icon(color=color, icon='info-sign')
                ).add_to(m)
                
                # Add route lines between points
                if j < len(cluster_indices) - 1:
                    next_lat = cluster_lats[j+1]
                    next_lng = cluster_longs[j+1]
                    
                    # Create a line with the same color
                    folium.PolyLine(
                        locations=[[lat, lng], [next_lat, next_lng]],
                        color=color,
                        weight=2,
                        opacity=0.7,
                        popup=f"Jarak: {self.distance_matrix[cluster_indices[j]][cluster_indices[j+1]]:.2f}"
                    ).add_to(m)

                # Add lines from Garage to first TPS and from last TPS to TPA
                if use_fixed_endpoints and start_idx is not None and end_idx is not None:
                    # Line from Garage to first TPS
                    if j == 0:  # Only add once for first TPS
                        folium.PolyLine(
                            locations=[[garage_lat, garage_long], [lat, lng]],
                            color=color,
                            weight=3,
                            opacity=0.7,
                            popup=f"Garasi → TPS pertama: {self.distance_matrix[start_idx][cluster_indices[0]]:.2f}"
                        ).add_to(m)
                    
                    # Line from last TPS to TPA
                    if j == len(cluster_indices) - 1:
                        folium.PolyLine(
                            locations=[[lat, lng], [tpa_lat, tpa_long]],
                            color=color,
                            weight=3,
                            opacity=0.7,
                            popup=f"TPS terakhir → TPA: {self.distance_matrix[cluster_indices[-1]][end_idx]:.2f}"
                        ).add_to(m)   
        
        # Add a layer control to toggle clusters
        folium.LayerControl().add_to(m)
        
        # Save to a temporary HTML file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.html')
        map_filename = temp_file.name
        temp_file.close()
        
        try:
            m.save(map_filename)
            print(f"Map saved to: {os.path.abspath(map_filename)}")  # Debug info
        except Exception as e:
            print(f"Error saving map: {str(e)}")
            # Create a simple backup map
            backup_html = f"""
            <html><body>
            <h1>Map Visualization</h1>
            <p>Error creating interactive map. Using basic view instead.</p>
            <p>Clusters: {len(self.results['clusters'])}</p>
            </body></html>
            """
            with open(map_filename, 'w') as f:
                f.write(backup_html)
        
        # Create a frame for buttons and status
        control_frame = ttk.Frame(self.map_tab)
        control_frame.pack(fill="x", padx=10, pady=10)
        
        # Add info label
        ttk.Label(
            control_frame, 
            text=f"Visualisasi peta dengan {len(self.results['clusters'])} cluster. Setiap warna mewakili satu cluster."
        ).pack(side="left", padx=5)
        
        # Add button to open in browser
        ttk.Button(
            control_frame, 
            text="Buka di Browser", 
            command=lambda: webbrowser.open('file://' + os.path.abspath(map_filename))
        ).pack(side="right", padx=5)
        
        # Add button to save the map
        ttk.Button(
            control_frame, 
            text="Download Peta (HTML)", 
            command=lambda: self.save_map(map_filename)
        ).pack(side="right", padx=5)
        
        # Add button to save as KML/KMZ
        ttk.Button(
            control_frame, 
            text="Download sebagai KML/KMZ", 
            command=self.export_to_kml
        ).pack(side="right", padx=5)
        
        # Create a frame for the map
        map_frame = ttk.Frame(self.map_tab)
        map_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Create a message with instructions
        message = ttk.Label(
            map_frame, 
            text="Peta telah dibuat! Klik 'Buka di Browser' untuk melihat peta interaktif atau 'Download Peta' untuk menyimpannya.",
            wraplength=600,
            justify="center"
        )
        message.pack(expand=True, fill="both")
        
        return True
    
    def save_map(self, map_filename):
        """Save the map HTML file to a user-specified location"""
        if not map_filename or not os.path.exists(map_filename):
            messagebox.showerror("Error", "File peta tidak ditemukan")
            return
            
        file_path = filedialog.asksaveasfilename(
            defaultextension=".html",
            filetypes=[("HTML files", "*.html")],
            title="Simpan Peta"
        )
        
        if file_path:
            try:
                # Copy the temp file to the user-selected location
                with open(map_filename, 'r', encoding='utf-8') as src_file:
                    map_html = src_file.read()
                
                with open(file_path, 'w', encoding='utf-8') as dest_file:
                    dest_file.write(map_html)
                
                messagebox.showinfo("Sukses", f"Peta berhasil disimpan ke {file_path}")
                
                # Offer to open in browser
                if messagebox.askyesno("Buka Peta", "Apakah Anda ingin membuka peta di browser sekarang?"):
                    webbrowser.open('file://' + os.path.abspath(file_path))
                    
            except Exception as e:
                messagebox.showerror("Error", f"Gagal menyimpan peta: {str(e)}")

    def read_excel_data_direct(self, excel_path):
        """
        Directly reads Excel file with specific format assumptions:
        - Sheet 1: 'Data' contains TPS info with 4+ columns (nama, lat, long, volume)
        - Sheet 2: 'Matrix' contains distance matrix with names in ONLY first row and column
        
        Returns:
        - data_info: Dictionary with TPS info
        - distance_matrix: Numpy array with distances
        - debug_info: String with debug information
        """
        debug_info = ""
        
        try:
            # Read Excel file
            xls = pd.ExcelFile(excel_path)
            
            # Save all sheets info
            all_sheets = xls.sheet_names
            debug_info += f"Daftar sheet dalam file: {', '.join(all_sheets)}\n\n"
            
            # Verify sheet count
            if len(all_sheets) < 2:
                return None, None, "File Excel harus memiliki minimal 2 sheet (Data dan Matrix)"
            
            # Read data sheet (first sheet)
            data_sheet = all_sheets[0]  # Biasanya "Data"
            data_df = pd.read_excel(excel_path, sheet_name=data_sheet)
            
            debug_info += f"Sheet data: '{data_sheet}'\n"
            debug_info += f"Ukuran data: {data_df.shape[0]} baris × {data_df.shape[1]} kolom\n"
            debug_info += f"Kolom: {', '.join(str(col) for col in data_df.columns)}\n"
            
            # Verify data columns
            if len(data_df.columns) < 4:
                return None, None, debug_info + "Sheet Data harus memiliki minimal 4 kolom (nama, lat, long, volume)"
            
            # Extract TPS info
            tps_names = data_df.iloc[:, 0].values
            tps_volumes = data_df.iloc[:, 3].values
            
            # Extract location data if available
            lat_values = None
            long_values = None
            if len(data_df.columns) >= 3:
                lat_values = data_df.iloc[:, 1].values
                long_values = data_df.iloc[:, 2].values
            
            debug_info += f"Jumlah TPS: {len(tps_names)}\n"
            debug_info += f"Total volume: {sum(tps_volumes):.2f} m³\n\n"
            
            # Add sample data for debugging
            sample_indices = min(5, len(tps_names))
            debug_info += "Sample TPS (5 pertama):\n"
            for i in range(sample_indices):
                debug_info += f"- {tps_names[i]}: {tps_volumes[i]} m³\n"
            debug_info += "\n"
            
            # Prepare data info dictionary
            data_info = {
                'names': tps_names,
                'volumes': tps_volumes,
                'count': len(tps_names),
                'df': data_df,  # Store original DataFrame for display
                'has_coords': lat_values is not None and long_values is not None,
                'lat': lat_values,
                'long': long_values
            }
            
            # Now read distance matrix sheet
            matrix_sheet = all_sheets[1]  # Biasanya "Matrix"
            
            debug_info += f"Sheet matriks: '{matrix_sheet}'\n"
            
            try:
                # First try reading with standard parameters (header in first row)
                matrix_df = pd.read_excel(excel_path, sheet_name=matrix_sheet, index_col=0)
                read_method = "Menggunakan kolom pertama sebagai index"
            except:
                # If that fails, try with no headers
                matrix_df = pd.read_excel(excel_path, sheet_name=matrix_sheet, header=None)
                read_method = "Tanpa header (raw data)"
            
            debug_info += f"Metode pembacaan matriks: {read_method}\n"
            debug_info += f"Ukuran raw matriks: {matrix_df.shape[0]} baris × {matrix_df.shape[1]} kolom\n"
            
            # Store original matrix for display
            original_matrix_df = matrix_df.copy()
            
            # Convert to numpy array (will automatically skip index column/row if present)
            try:
                distance_matrix = matrix_df.values.astype(float)
                debug_info += "Berhasil mengkonversi matriks ke nilai numerik\n"
            except:
                debug_info += "Gagal mengkonversi matriks langsung, mencoba pendekatan lain...\n"
                
                # Try explicit conversion column by column
                for col in matrix_df.columns:
                    matrix_df[col] = pd.to_numeric(matrix_df[col], errors='coerce')
                distance_matrix = matrix_df.values.astype(float)
            
            debug_info += f"Ukuran array matriks: {distance_matrix.shape[0]}x{distance_matrix.shape[1]}\n"
            
            # Check if matrix size matches TPS count
            if distance_matrix.shape[0] != len(tps_names) or distance_matrix.shape[1] != len(tps_names):
                debug_info += f"PERINGATAN: Ukuran matriks ({distance_matrix.shape[0]}x{distance_matrix.shape[1]}) "
                debug_info += f"tidak sesuai dengan jumlah TPS ({len(tps_names)})\n"
                
                # Create a matrix of the correct size filled with large values
                full_matrix = np.ones((len(tps_names), len(tps_names))) * 999999
                
                # Fill diagonal with zeros (distance to self = 0)
                np.fill_diagonal(full_matrix, 0)
                
                # Copy available distances
                min_rows = min(distance_matrix.shape[0], len(tps_names))
                min_cols = min(distance_matrix.shape[1], len(tps_names))
                
                for i in range(min_rows):
                    for j in range(min_cols):
                        full_matrix[i, j] = distance_matrix[i, j]
                
                distance_matrix = full_matrix
                debug_info += f"Dibuat matriks {len(tps_names)}x{len(tps_names)} dengan nilai default = 999999\n"
            
            # Check for NaN values and replace with large values
            nan_count = np.isnan(distance_matrix).sum()
            if nan_count > 0:
                debug_info += f"Mengganti {nan_count} nilai NaN dengan 999999\n"
                distance_matrix = np.nan_to_num(distance_matrix, nan=999999)
            
            # Verify diagonal contains zeros (jarak TPS ke dirinya sendiri = 0)
            diag_sum = np.sum(np.diag(distance_matrix))
            if diag_sum > 0:
                debug_info += "PERINGATAN: Diagonal utama tidak semuanya 0. Memperbaiki...\n"
                np.fill_diagonal(distance_matrix, 0)
            
            debug_info += f"Matriks jarak final: {distance_matrix.shape[0]}x{distance_matrix.shape[1]}\n"
            debug_info += "Nilai diagonal utama (contoh 5 pertama): " + str(np.diag(distance_matrix)[:5]) + "\n"
            
            # Store original matrix for display
            data_info['original_matrix_df'] = original_matrix_df
            
            return data_info, distance_matrix, debug_info
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            error_msg = f"Error saat membaca file Excel: {str(e)}\n\nDetail error:\n{error_details}"
            return None, None, debug_info + "\n" + error_msg

    def display_dataframe(self, df, title, parent_frame):
        """Display a DataFrame in a scrollable frame with proper formatting"""
        # Create a label with the dataframe info
        info_label = ttk.Label(parent_frame, 
                              text=f"Data '{title}': {df.shape[0]} baris × {df.shape[1]} kolom")
        info_label.pack(fill="x", padx=5, pady=5, anchor="w")
        
        # Create a Text widget to display the DataFrame as text
        text_widget = tk.Text(parent_frame, wrap="none", height=20)
        text_widget.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Add horizontal and vertical scrollbars
        h_scrollbar = ttk.Scrollbar(parent_frame, orient="horizontal", command=text_widget.xview)
        h_scrollbar.pack(fill="x", side="bottom")
        
        v_scrollbar = ttk.Scrollbar(parent_frame, orient="vertical", command=text_widget.yview)
        v_scrollbar.pack(fill="y", side="right")
        
        text_widget.config(xscrollcommand=h_scrollbar.set, yscrollcommand=v_scrollbar.set)
        
        # Insert DataFrame as formatted text
        try:
            # Format the DataFrame as string with reasonable width
            df_str = df.to_string()
            text_widget.insert("1.0", df_str)
            
            # Add a notice of data size
            text_widget.insert("1.0", f"=== {title} ({df.shape[0]} baris × {df.shape[1]} kolom) ===\n\n")
            
        except Exception as e:
            error_msg = f"Error menampilkan dataframe: {str(e)}\n\n"
            error_msg += "Mencoba menampilkan head(10):\n\n"
            
            try:
                error_msg += str(df.head(10))
            except:
                error_msg += "Gagal menampilkan head(10)"
                
            text_widget.insert("1.0", error_msg)
    
    def display_matrix(self, matrix, tps_names, title, parent_frame):
        """Display a distance matrix with TPS names as labels"""
        # Create a label with the matrix info
        info_label = ttk.Label(parent_frame, 
                              text=f"{title}: {matrix.shape[0]}x{matrix.shape[1]} matriks")
        info_label.pack(fill="x", padx=5, pady=5, anchor="w")
        
        # Create a Text widget for the matrix
        text_widget = tk.Text(parent_frame, wrap="none", height=20)
        text_widget.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Add horizontal and vertical scrollbars
        h_scrollbar = ttk.Scrollbar(parent_frame, orient="horizontal", command=text_widget.xview)
        h_scrollbar.pack(fill="x", side="bottom")
        
        v_scrollbar = ttk.Scrollbar(parent_frame, orient="vertical", command=text_widget.yview)
        v_scrollbar.pack(fill="y", side="right")
        
        text_widget.config(xscrollcommand=h_scrollbar.set, yscrollcommand=v_scrollbar.set)
        
        try:
            # Create a DataFrame from the matrix with TPS names as labels
            if len(tps_names) >= matrix.shape[0]:
                row_labels = tps_names[:matrix.shape[0]]
                col_labels = tps_names[:matrix.shape[1]]
                matrix_df = pd.DataFrame(matrix, index=row_labels, columns=col_labels)
                
                # Format and display
                matrix_str = matrix_df.to_string()
                text_widget.insert("1.0", matrix_str)
            else:
                # Display raw matrix if names don't match
                matrix_str = pd.DataFrame(matrix).to_string()
                text_widget.insert("1.0", matrix_str)
                text_widget.insert("1.0", "Peringatan: Jumlah nama TPS tidak cukup untuk label matriks\n\n")
        
        except Exception as e:
            text_widget.insert("1.0", f"Error menampilkan matriks: {str(e)}\n\nMenampilkan raw data:\n\n")
            try:
                # Display first 10x10 portion of matrix
                preview = matrix[:min(10, matrix.shape[0]), :min(10, matrix.shape[1])]
                text_widget.insert("end", str(preview))
            except:
                text_widget.insert("end", "Tidak dapat menampilkan preview matriks")

    def display_results(self):
        """Display the clustering results in the UI with improved scrollable frames"""
        if not self.results:
            return
                
        # Clear previous results
        for widget in self.summary_tab.winfo_children():
            widget.destroy()
                
        for widget in self.viz_tab.winfo_children():
            widget.destroy()
                
        for widget in self.details_tab.winfo_children():
            widget.destroy()
        
        # SUMMARY TAB with scrollable frame
        summary_frame = self.create_scrollable_frame(self.summary_tab)
        
        ttk.Label(summary_frame, text=f"Algoritma: {self.results['algorithm']}", font=("Arial", 12, "bold")).pack(anchor="w", pady=5)
        ttk.Label(summary_frame, text=f"Total TPS: {self.results['total_tps']}", font=("Arial", 12)).pack(anchor="w", pady=5)
        ttk.Label(summary_frame, text=f"Total Volume: {self.results['total_volume']:.2f} m³", font=("Arial", 12)).pack(anchor="w", pady=5)
        ttk.Label(summary_frame, text=f"Jumlah Cluster: {self.results['num_clusters']}", font=("Arial", 12)).pack(anchor="w", pady=5)
        
        # Add multi-start info if available
        if 'multi_start_info' in self.results:
            ms_info = self.results['multi_start_info']
            ttk.Label(summary_frame, text=f"Seed Terbaik: {ms_info.get('best_seed', '-')}", 
                    font=("Arial", 12, "bold"), foreground="blue").pack(anchor="w", pady=5)
            ttk.Label(summary_frame, text=f"Total Run Multi-Start: {ms_info.get('num_runs', 0)}", 
                    font=("Arial", 12)).pack(anchor="w", pady=5)
            ttk.Label(summary_frame, text=f"Feasible: {'Ya' if ms_info.get('best_feasible', False) else 'Tidak'}", 
                    font=("Arial", 12)).pack(anchor="w", pady=5)
            ttk.Label(summary_frame, text=f"Total Waktu Multi-Start: {ms_info.get('total_process_time', 0):.2f} detik", 
                    font=("Arial", 12)).pack(anchor="w", pady=5)
        
        if 'metrics' in self.results:
            metrics = self.results['metrics']
            
            # Display different distance metrics depending on route optimization setting
            if self.results.get('optimize_routes', False):
                ttk.Label(summary_frame, text=f"Total Jarak Rute (Antar TPS): {metrics['total_route_distance']:.2f}", 
                        font=("Arial", 12)).pack(anchor="w", pady=5)
                
                # Add total distance including Garage-TPA if available
                if 'total_complete_route_distance' in metrics:
                    ttk.Label(summary_frame, text=f"Total Jarak Rute (Termasuk Garasi-TPA): {metrics['total_complete_route_distance']:.2f}", 
                            font=("Arial", 12, "bold"), foreground="blue").pack(anchor="w", pady=5)
                
                ttk.Label(summary_frame, text=f"Rata-rata Jarak Rute per Cluster: {metrics['avg_route_distance_per_cluster']:.2f}", 
                        font=("Arial", 12)).pack(anchor="w", pady=5)
            else:
                ttk.Label(summary_frame, text=f"Total Jarak: {metrics['total_distance']:.2f}", 
                        font=("Arial", 12)).pack(anchor="w", pady=5)
                ttk.Label(summary_frame, text=f"Jarak Rata-rata per Cluster: {metrics['avg_distance_per_cluster']:.2f}", 
                        font=("Arial", 12)).pack(anchor="w", pady=5)
        
        if 'execution_time' in self.results:
            ttk.Label(summary_frame, text=f"Waktu Eksekusi Algoritma: {self.results['execution_time']:.2f} detik", 
                    font=("Arial", 12)).pack(anchor="w", pady=5)
        
        # Add export button
        export_frame = ttk.Frame(summary_frame)
        export_frame.pack(pady=10)
        ttk.Button(export_frame, text="Export Hasil (Excel)", command=self.export_results).pack(side="left", padx=5)
        
        # -------------- VISUALIZATION TAB --------------
        # Create frame with scrollbar
        viz_frame = self.create_scrollable_frame(self.viz_tab)

        # Create larger figure to accommodate many clusters
        fig1 = plt.Figure(figsize=(16, 7), dpi=100)
        ax1 = fig1.add_subplot(111)

        # Use only numbers for cluster labels
        cluster_numbers = [str(c['cluster_number']) for c in self.results['clusters']]
        cluster_volumes = [c['total_volume'] for c in self.results['clusters']]
        cluster_counts = [c['num_tps'] for c in self.results['clusters']]

        # Main axis for volume
        bars = ax1.bar(cluster_numbers, cluster_volumes, color='skyblue', width=0.7)
        ax1.set_ylabel('Volume Total (m³)', color='blue')
        ax1.set_xlabel('Cluster')
        ax1.tick_params(axis='y', labelcolor='blue')

        # Secondary axis for TPS count
        ax2 = ax1.twinx()
        ax2.plot(cluster_numbers, cluster_counts, 'ro-', linewidth=2, markersize=8)
        ax2.set_ylabel('Jumlah TPS', color='red')
        ax2.tick_params(axis='y', labelcolor='red')

        # Rotate volume text to vertical and fix position
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax1.text(
                bar.get_x() + bar.get_width()/2.,
                height - 0.4,  # Position inside bar
                f'{height:.2f}',
                ha='center', va='top', color='blue',
                rotation=90,  # Vertical rotation
                fontsize=9    # Smaller font
            )

        # Add legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='skyblue', lw=4, label='Volume (m³)'),
            Line2D([0], [0], color='red', marker='o', lw=2, label='Jumlah TPS')
        ]
        ax1.legend(handles=legend_elements, loc='upper right')

        # Add title and fix layout
        fig1.suptitle(f'Visualisasi Cluster TPS ({self.results["algorithm"]})', fontsize=16)
        fig1.tight_layout(pad=2.0)  # Add padding

        # Add figure to canvas
        canvas1 = FigureCanvasTkAgg(fig1, master=viz_frame)
        canvas1.draw()
        canvas1.get_tk_widget().pack(fill="both", expand=True, padx=5, pady=5)

        # Figure 2: Pie chart 
        fig2 = plt.Figure(figsize=(10, 7), dpi=100)
        ax3 = fig2.add_subplot(111)

        # Pie chart volume distribution
        volume_percents = [c['total_volume']/self.results['total_volume']*100 for c in self.results['clusters']]
        patches, texts, autotexts = ax3.pie(
            volume_percents, 
            labels=cluster_numbers,
            autopct='%1.1f%%',
            startangle=90,
            shadow=True
        )
        ax3.axis('equal')  # Equal aspect ratio for circle

        # Fix label readability
        for text in texts:
            text.set_fontsize(9)
            
        for autotext in autotexts:
            autotext.set_fontsize(9)
            autotext.set_color('white')

        fig2.suptitle('Distribusi Volume TPS per Cluster', fontsize=16)
        fig2.tight_layout(pad=2.0)

        # Add to frame
        canvas2 = FigureCanvasTkAgg(fig2, master=viz_frame)
        canvas2.draw()
        canvas2.get_tk_widget().pack(fill="both", expand=True, padx=5, pady=5)

        # Add download button
        download_button = ttk.Button(
            viz_frame, 
            text="Download Visualisasi", 
            command=lambda: self.save_visualization([fig1, fig2], "viz_")
        )
        download_button.pack(pady=10)
        # -------------- END VISUALIZATION TAB --------------
        
        # Visualize routes if coordinates are available
        self.visualize_routes()
        
        # Create notebook for details tab (one tab per cluster)
        details_notebook = ttk.Notebook(self.details_tab)
        details_notebook.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Create a tab for each cluster
        for cluster_info in self.results['clusters']:
            cluster_num = cluster_info['cluster_number']
            cluster_tab = ttk.Frame(details_notebook)
            details_notebook.add(cluster_tab, text=f"Cluster {cluster_num}")
            
            # Create scrollable frame for this cluster tab
            cluster_scrollable_frame = self.create_scrollable_frame(cluster_tab)
            
            # Split into two frames - info on left, TPS table on right
            info_frame = ttk.Frame(cluster_scrollable_frame)
            info_frame.pack(side="left", fill="y", padx=10, pady=10, anchor="n")
            
            table_frame = ttk.Frame(cluster_scrollable_frame)
            table_frame.pack(side="right", fill="both", expand=True, padx=10, pady=10)
            
            # Cluster info in the left frame
            ttk.Label(info_frame, text=f"Cluster {cluster_num}", font=("Arial", 12, "bold")).pack(anchor="w", pady=5)
            ttk.Label(info_frame, text=f"Jumlah TPS: {cluster_info['num_tps']}", font=("Arial", 11)).pack(anchor="w")
            ttk.Label(info_frame, text=f"Total Volume: {cluster_info['total_volume']:.2f} m³", font=("Arial", 11)).pack(anchor="w")
            
            # Calculate min/max capacity bounds
            min_cap = self.min_capacity_var.get()
            max_cap = self.max_capacity_var.get()
            
            ttk.Label(info_frame, text=f"Batas Volume: {min_cap:.2f} - {max_cap:.2f} m³", font=("Arial", 11)).pack(anchor="w")
            
            # Check if cluster meets capacity constraints
            if cluster_info['total_volume'] < min_cap:
                ttk.Label(info_frame, text=f"⚠️ Volume di bawah kapasitas minimum ({min_cap:.2f} m³)!", 
                        font=("Arial", 11, "bold"), foreground="red").pack(anchor="w")
            elif cluster_info['total_volume'] > max_cap:
                ttk.Label(info_frame, text=f"⚠️ Volume melebihi kapasitas maksimum ({max_cap:.2f} m³)!", 
                        font=("Arial", 11, "bold"), foreground="red").pack(anchor="w")
            else:
                ttk.Label(info_frame, text=f"✓ Volume memenuhi batasan kapasitas", 
                        font=("Arial", 11), foreground="green").pack(anchor="w")
            
            # If we have route-specific metric, show it
            if self.results.get('optimize_routes', False) and 'metrics' in self.results:
                metrics = self.results['metrics']
                for cm in metrics['cluster_metrics']:
                    if cm['cluster_id'] == cluster_num:
                        ttk.Label(info_frame, text=f"Jarak Rute (Antar TPS): {cm['route_distance']:.2f}", 
                                font=("Arial", 11)).pack(anchor="w")
                        
                        # Add Garage-TPA distance info
                        if 'complete_route_distance' in cm:
                            ttk.Label(info_frame, text=f"Jarak Rute Lengkap (Garasi-TPA): {cm['complete_route_distance']:.2f}", 
                                    font=("Arial", 11, "bold"), foreground="blue").pack(anchor="w")
                            
                            if 'garage_to_first' in cm and cm['garage_to_first'] > 0:
                                ttk.Label(info_frame, text=f"  Garasi → TPS pertama: {cm['garage_to_first']:.2f}", 
                                        font=("Arial", 11)).pack(anchor="w")
                            
                            if 'last_to_tpa' in cm and cm['last_to_tpa'] > 0:
                                ttk.Label(info_frame, text=f"  TPS terakhir → TPA: {cm['last_to_tpa']:.2f}", 
                                        font=("Arial", 11)).pack(anchor="w")
                        break
            
            # Add download button for this cluster's data
            ttk.Button(
                info_frame, 
                text="Download Data Cluster Ini", 
                command=lambda c=cluster_num: self.export_single_cluster(c)
            ).pack(anchor="w", pady=10)
            
            # Now create the TPS table in the right frame with its own scrollbar
            table_container = ttk.Frame(table_frame)
            table_container.pack(fill="both", expand=True)
            
            # Prepare table with route info
            columns = ("No.", "Nama TPS", "Volume (m³)", "Jarak ke Berikutnya")
            tree = ttk.Treeview(table_container, columns=columns, show="headings")
            
            for col in columns:
                tree.heading(col, text=col)
                tree.column(col, anchor="center", width=100)
            
            # Shorter width for sequence number and slightly wider for TPS name
            tree.column("No.", width=50, anchor="center")
            tree.column("Nama TPS", width=200, anchor="w")
            
            # Add vertical scrollbar for table
            v_scrollbar = ttk.Scrollbar(table_container, orient="vertical", command=tree.yview)
            v_scrollbar.pack(side="right", fill="y")
            
            # Add horizontal scrollbar for table
            h_scrollbar = ttk.Scrollbar(table_container, orient="horizontal", command=tree.xview)
            h_scrollbar.pack(side="bottom", fill="x")
            
            # Configure tree to use scrollbars
            tree.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
            tree.pack(fill="both", expand=True)
            
            # Extract TPS data
            tps_indices = cluster_info['tps_indices']
            tps_names = cluster_info['tps_names']
            tps_volumes = cluster_info['tps_volumes']

            # Check if using fixed endpoints
            use_fixed_endpoints = self.use_fixed_endpoints_var.get()
            start_point_name = self.start_point_var.get() if use_fixed_endpoints else None
            end_point_name = self.end_point_var.get() if use_fixed_endpoints else None

            # Find indices of start and end points
            start_idx = None
            end_idx = None

            if use_fixed_endpoints and self.data_info and 'names' in self.data_info:
                names = self.data_info['names']
                
                # Find indices of start and end points with better matching
                for i, name in enumerate(names):
                    if isinstance(name, str) and (start_point_name.lower() in name.lower() or 
                                                name.lower() in start_point_name.lower()):
                        start_idx = i
                        break
                
                for i, name in enumerate(names):
                    if isinstance(name, str) and (end_point_name.lower() in name.lower() or 
                                                name.lower() in end_point_name.lower()):
                        end_idx = i
                        break
            
            # Insert TPS data in route order (already sorted in tps_indices)
            # Make sure we have data before trying to insert it
            if tps_indices and tps_names and tps_volumes:
                # Insert TPS data in route order
                for i, (idx, name, volume) in enumerate(zip(tps_indices, tps_names, tps_volumes)):
                    try:
                        # Calculate distance to next TPS
                        if i < len(tps_indices) - 1:
                            next_idx = tps_indices[i+1]
                            distance = self.distance_matrix[idx][next_idx]
                            distance_str = f"{distance:.2f}"
                        else:
                            distance_str = "-"
                        
                        tree.insert("", "end", values=(i+1, name, f"{volume:.2f}", distance_str))
                    except Exception as e:
                        print(f"Error inserting TPS data for cluster {cluster_num}, item {i}: {str(e)}")
            else:
                tree.insert("", "end", values=("N/A", "No data available", "N/A", "N/A"))
            
            # If using fixed endpoints, add rows for Garage and TPA
            if use_fixed_endpoints and 'metrics' in self.results:
                for cm in self.results['metrics']['cluster_metrics']:
                    if cm['cluster_id'] == cluster_num and 'garage_to_first' in cm and 'last_to_tpa' in cm:
                        # Add row for Garage to first TPS
                        tree.insert("", 0, values=("G", "Garasi", "-", f"{cm['garage_to_first']:.2f}"))
                        
                        # Add row for last TPS to TPA
                        tree.insert("", "end", values=("T", "TPA", "-", f"{cm['last_to_tpa']:.2f}"))
                        
                        # Add total distance row
                        tree.insert("", "end", values=("Total", "", "", f"{cm['complete_route_distance']:.2f}"))
                        break  

        self.display_centroids_tab()

    def visualize_multi_start_results(self, results):
        """
        Visualize results from multi-start optimization with improved scrollable frames
        
        Parameters:
        -----------
        results : list
            List of results from different runs
        """
        if not results:
            # Show message if no results
            ttk.Label(
                self.multi_start_tab, 
                text="Tidak ada hasil multi-start yang ditemukan.",
                font=("Arial", 12, "bold"),
                foreground="red"
            ).pack(expand=True, pady=20)
            return
        
        # Definisikan fungsi untuk menyimpan visualisasi sebagai PDF
        def save_as_pdf():
            """Save all multi-start visualizations as a single PDF file"""
            if not hasattr(self, 'multi_start_figures') or not self.multi_start_figures:
                messagebox.showinfo("Info", "Tidak ada gambar multi-start yang dapat disimpan")
                return
                    
            file_path = filedialog.asksaveasfilename(
                defaultextension=".pdf",
                filetypes=[("PDF files", "*.pdf")],
                title="Simpan Semua Visualisasi Multi-Start (PDF)"
            )
            
            if file_path:
                try:
                    from matplotlib.backends.backend_pdf import PdfPages
                    
                    with PdfPages(file_path) as pdf:
                        # Save individual figures
                        for fig in self.multi_start_figures:
                            pdf.savefig(fig, bbox_inches='tight')
                    
                    messagebox.showinfo("Sukses", f"Semua visualisasi multi-start berhasil disimpan ke {file_path}")
                    
                    # Ask if user wants to open the file
                    if messagebox.askyesno("Buka PDF", "Apakah Anda ingin membuka file PDF sekarang?"):
                        webbrowser.open('file://' + os.path.abspath(file_path))
                except Exception as e:
                    messagebox.showerror("Error", f"Gagal menyimpan file PDF: {str(e)}")
        
        # Definisikan fungsi untuk menyimpan visualisasi sebagai gambar terpisah
        def save_as_images():
            """Save all multi-start visualizations as separate image files"""
            if not hasattr(self, 'multi_start_figures') or not self.multi_start_figures:
                messagebox.showinfo("Info", "Tidak ada gambar multi-start yang dapat disimpan")
                return
                    
            # Ask for directory to save images
            directory = filedialog.askdirectory(
                title="Pilih Folder untuk Menyimpan Gambar Multi-Start"
            )
            
            if directory:
                try:
                    # Ask for image format
                    formats = {
                        "PNG (.png)": ".png",
                        "JPEG (.jpg)": ".jpg",
                        "PDF (.pdf)": ".pdf",
                        "SVG (.svg)": ".svg"
                    }
                    
                    format_dialog = tk.Toplevel(self.root)
                    format_dialog.title("Pilih Format Gambar")
                    format_dialog.geometry("300x200")
                    format_dialog.resizable(False, False)
                    format_dialog.transient(self.root)
                    format_dialog.grab_set()
                    
                    ttk.Label(format_dialog, text="Pilih format gambar:", padding=10).pack()
                    
                    selected_format = tk.StringVar()
                    for format_name, format_ext in formats.items():
                        ttk.Radiobutton(
                            format_dialog, 
                            text=format_name, 
                            value=format_ext, 
                            variable=selected_format
                        ).pack(anchor="w", padx=20, pady=5)
                    
                    # Default to PNG
                    selected_format.set(".png")
                    
                    def on_ok():
                        format_ext = selected_format.get()
                        format_dialog.destroy()
                        
                        try:
                            # Nama untuk gambar yang disimpan
                            chart_names = [
                                "scatter_plot",
                                "bar_chart_feasible",
                                "statistics_chart",
                                "overview"
                            ]
                            
                            # Save each figure
                            for i, fig in enumerate(self.multi_start_figures):
                                if i < len(chart_names):
                                    filename = f"multi_start_{chart_names[i]}{format_ext}"
                                else:
                                    filename = f"multi_start_chart_{i+1}{format_ext}"
                                
                                file_path = os.path.join(directory, filename)
                                fig.savefig(file_path, dpi=300, bbox_inches='tight')
                            
                            messagebox.showinfo("Sukses", f"Semua visualisasi multi-start berhasil disimpan ke folder {directory}")
                            
                            # Ask if user wants to open the directory
                            if messagebox.askyesno("Buka Folder", "Apakah Anda ingin membuka folder sekarang?"):
                                webbrowser.open('file://' + os.path.abspath(directory))
                        except Exception as e:
                            messagebox.showerror("Error", f"Gagal menyimpan gambar: {str(e)}")
                    
                    def on_cancel():
                        format_dialog.destroy()
                    
                    button_frame = ttk.Frame(format_dialog)
                    button_frame.pack(fill="x", pady=10)
                    
                    ttk.Button(button_frame, text="OK", command=on_ok).pack(side="right", padx=10)
                    ttk.Button(button_frame, text="Batal", command=on_cancel).pack(side="right", padx=10)
                    
                    # Center the dialog
                    format_dialog.update_idletasks()
                    width = format_dialog.winfo_width()
                    height = format_dialog.winfo_height()
                    x = (self.root.winfo_width() // 2) - (width // 2)
                    y = (self.root.winfo_height() // 2) - (height // 2)
                    format_dialog.geometry(f"+{x}+{y}")
                    
                    format_dialog.wait_window()
                    
                except Exception as e:
                    messagebox.showerror("Error", f"Gagal menyimpan gambar: {str(e)}")
        
        # Clear tab
        for widget in self.multi_start_tab.winfo_children():
            widget.destroy()
        
        # Inisialisasi list untuk menyimpan figure
        self.multi_start_figures = []
        
        # Create scrollable frame using the helper function
        scrollable_frame = self.create_scrollable_frame(self.multi_start_tab)
        
        # Title
        ttk.Label(
            scrollable_frame, 
            text=f"Hasil Multi-Start ({len(results)} run)", 
            font=("Arial", 14, "bold")
        ).pack(pady=5)
        
        # Best solution section
        best_frame = ttk.LabelFrame(scrollable_frame, text="Solusi Terbaik")
        best_frame.pack(fill="x", expand=True, padx=5, pady=5)
        
        # Find best solution
        feasible_results = [r for r in results if r.get('feasible', False)]
        if feasible_results:
            feasible_results.sort(key=lambda x: x.get('distance', float('inf')))
            best_result = feasible_results[0]
        else:
            # If no feasible solutions, find best by distance
            results.sort(key=lambda x: x.get('distance', float('inf')))
            best_result = results[0]
        
        # Display best solution info
        best_info_text = (
            f"Seed: {best_result.get('seed', '-')}\n"
            f"Jumlah Cluster: {best_result.get('num_clusters', 0)}\n"
            f"Jarak Total: {best_result.get('distance', float('inf')):.2f}\n"
            f"Feasible: {'Ya' if best_result.get('feasible', False) else 'Tidak'}\n"
            f"Waktu Eksekusi: {best_result.get('execution_time', 0):.2f} detik"
        )
        
        ttk.Label(best_frame, text=best_info_text, justify="left").pack(anchor="w", padx=10, pady=5)
        
        # Comparison table
        table_frame = ttk.LabelFrame(scrollable_frame, text="Perbandingan Hasil")
        table_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Create treeview container with horizontal scrollbar
        tree_container = ttk.Frame(table_frame)
        tree_container.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Create treeview for table
        columns = ("Run", "Seed", "Clusters", "Distance", "Feasible", "Time")
        tree = ttk.Treeview(tree_container, columns=columns, show="headings", height=10)
        
        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, width=80, anchor="center")
        
        # Adjust column widths
        tree.column("Run", width=40)
        tree.column("Seed", width=40)
        tree.column("Clusters", width=60)
        tree.column("Distance", width=100)
        tree.column("Feasible", width=60)
        tree.column("Time", width=60)
        
        # Add scrollbars to treeview
        tree_v_scrollbar = ttk.Scrollbar(tree_container, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=tree_v_scrollbar.set)
        tree_v_scrollbar.pack(side="right", fill="y")
        
        tree_h_scrollbar = ttk.Scrollbar(tree_container, orient="horizontal", command=tree.xview)
        tree.configure(xscrollcommand=tree_h_scrollbar.set)
        tree_h_scrollbar.pack(side="bottom", fill="x")
        
        tree.pack(fill="both", expand=True)
        
        # Add data to table
        for i, result in enumerate(results):
            feasible_text = "Ya" if result.get('feasible', False) else "Tidak"
            tree.insert(
                "", "end", 
                values=(
                    i+1, 
                    result.get('seed', '-'), 
                    result.get('num_clusters', 0),
                    f"{result.get('distance', float('inf')):.2f}",
                    feasible_text,
                    f"{result.get('execution_time', 0):.2f}s"
                )
            )
        
        # Visualize with scatter plot
        fig1 = plt.Figure(figsize=(10, 5), dpi=100)
        ax1 = fig1.add_subplot(111)
        
        # Extract data for visualization
        seeds = [r.get('seed', i) for i, r in enumerate(results)]
        distances = [r.get('distance', float('inf')) for r in results]
        num_clusters = [r.get('num_clusters', 0) for r in results]
        feasible = [r.get('feasible', False) for r in results]
        
        # Colors based on feasibility
        colors = ['green' if f else 'red' for f in feasible]
        
        # Create scatter plot
        scatter = ax1.scatter(num_clusters, distances, c=colors, alpha=0.7)
        
        # Add seed labels
        for i, seed in enumerate(seeds):
            ax1.annotate(str(seed), (num_clusters[i], distances[i]), fontsize=8)
        
        # Add legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='Feasible'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Tidak Feasible')
        ]
        ax1.legend(handles=legend_elements)
        
        # Labels and title
        ax1.set_xlabel('Jumlah Cluster')
        ax1.set_ylabel('Total Distance')
        ax1.set_title('Perbandingan Hasil Multi-Start')
        ax1.grid(True)
        
        # Adjust layout
        fig1.tight_layout()
        
        # Add to UI
        chart_frame1 = ttk.Frame(scrollable_frame)
        chart_frame1.pack(fill="both", expand=True, padx=5, pady=5)
        
        canvas1 = FigureCanvasTkAgg(fig1, master=chart_frame1)
        canvas1.draw()
        canvas1.get_tk_widget().pack(fill="both", expand=True)
        
        # Add to figures list
        self.multi_start_figures.append(fig1)
        
        # Bar chart for feasible results
        feasible_indices = [i for i, f in enumerate(feasible) if f]
        
        if feasible_indices:
            fig2 = plt.Figure(figsize=(10, 5), dpi=100)
            ax2 = fig2.add_subplot(111)
            
            # Filter for feasible solutions
            feasible_seeds = [seeds[i] for i in feasible_indices]
            feasible_distances = [distances[i] for i in feasible_indices]
            
            # Sort by distance
            sorted_indices = np.argsort(feasible_distances)
            sorted_seeds = [feasible_seeds[i] for i in sorted_indices]
            sorted_distances = [feasible_distances[i] for i in sorted_indices]
            
            # Create bar chart
            bars = ax2.bar(range(len(sorted_seeds)), sorted_distances, color='skyblue')
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax2.annotate(f'{height:.2f}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom')
            
            # Set x-ticks as seeds
            ax2.set_xticks(range(len(sorted_seeds)))
            ax2.set_xticklabels([f"Seed {s}" for s in sorted_seeds])
            
            # Add best solution marker
            best_idx = sorted_indices[0]
            ax2.axhline(y=sorted_distances[0], color='r', linestyle='--', 
                    label=f'Best: {sorted_distances[0]:.2f} (Seed {sorted_seeds[0]})')
            
            # Labels and title
            ax2.set_xlabel('Run (Seeds)')
            ax2.set_ylabel('Total Distance')
            ax2.set_title('Perbandingan Jarak untuk Solusi Feasible')
            ax2.legend()
            ax2.grid(True, axis='y')
            
            # Adjust layout
            fig2.tight_layout()
            
            # Add to UI
            chart_frame2 = ttk.Frame(scrollable_frame)
            chart_frame2.pack(fill="both", expand=True, padx=5, pady=5)
            
            canvas2 = FigureCanvasTkAgg(fig2, master=chart_frame2)
            canvas2.draw()
            canvas2.get_tk_widget().pack(fill="both", expand=True)
            
            # Add to figures list
            self.multi_start_figures.append(fig2)
        
        # Statistics
        stats_frame = ttk.LabelFrame(scrollable_frame, text="Statistik Multi-Start")
        stats_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Calculate statistics
        all_stats_text = (
            f"Total run: {len(results)}\n"
            f"Run feasible: {sum(feasible)}\n"
            f"Run tidak feasible: {len(results) - sum(feasible)}\n"
            f"Rentang jumlah cluster: {min(num_clusters)} - {max(num_clusters)}\n"
            f"Waktu total: {sum([r.get('execution_time', 0) for r in results]):.2f} detik\n"
            f"Waktu rata-rata per run: {sum([r.get('execution_time', 0) for r in results])/len(results):.2f} detik"
        )
        
        ttk.Label(stats_frame, text=all_stats_text, justify="left").pack(anchor="w", padx=10, pady=5)
        
        # Statistics for feasible results
        if feasible_indices:
            feasible_distances = [distances[i] for i in feasible_indices]
            
            feasible_stats_text = (
                f"\nStatistik untuk solusi feasible:\n"
                f"Jarak minimum: {min(feasible_distances):.2f}\n"
                f"Jarak maksimum: {max(feasible_distances):.2f}\n"
                f"Jarak rata-rata: {sum(feasible_distances)/len(feasible_distances):.2f}\n"
                f"Standar deviasi: {np.std(feasible_distances):.2f}\n"
                f"Koefisien variasi: {np.std(feasible_distances)/np.mean(feasible_distances)*100:.2f}%"
            )
            
            ttk.Label(stats_frame, text=feasible_stats_text, justify="left").pack(anchor="w", padx=10, pady=5)
            
            # Create statistics visualization
            fig3 = plt.Figure(figsize=(10, 6), dpi=100)
            ax3 = fig3.add_subplot(111)
            
            # Create boxplot of feasible distances
            ax3.boxplot(feasible_distances, vert=False, widths=0.7, patch_artist=True,
                    boxprops=dict(facecolor='lightblue'))
            
            # Add individual points (jitter)
            y_jitter = np.random.normal(1, 0.04, size=len(feasible_distances))
            ax3.scatter(feasible_distances, y_jitter, color='blue', alpha=0.5)
            
            # Add best, avg, worst markers
            ax3.axvline(x=min(feasible_distances), color='g', linestyle='-', linewidth=2, 
                    label=f'Min: {min(feasible_distances):.2f}')
            ax3.axvline(x=np.mean(feasible_distances), color='b', linestyle='--', linewidth=2,
                    label=f'Avg: {np.mean(feasible_distances):.2f}')
            ax3.axvline(x=max(feasible_distances), color='r', linestyle='-.', linewidth=2,
                    label=f'Max: {max(feasible_distances):.2f}')
            
            # Labels and title
            ax3.set_xlabel('Jarak Total')
            ax3.set_title('Distribusi Jarak untuk Solusi Feasible')
            ax3.legend()
            ax3.grid(True, axis='x')
            
            # Hide y-axis labels
            ax3.set_yticks([])
            
            # Adjust layout
            fig3.tight_layout()
            
            # Add to UI
            chart_frame3 = ttk.Frame(scrollable_frame)
            chart_frame3.pack(fill="both", expand=True, padx=5, pady=5)
            
            canvas3 = FigureCanvasTkAgg(fig3, master=chart_frame3)
            canvas3.draw()
            canvas3.get_tk_widget().pack(fill="both", expand=True)
            
            # Add to figures list
            self.multi_start_figures.append(fig3)
        
        # Create overview visualization with multiple subplots
        fig4 = plt.Figure(figsize=(12, 10), dpi=100)
        fig4.suptitle("Multi-Start Overview", fontsize=16, fontweight='bold')
        
        # Scatter plot in first subplot
        ax4_1 = fig4.add_subplot(221)
        ax4_1.scatter(num_clusters, distances, c=colors, alpha=0.7)
        ax4_1.set_xlabel('Jumlah Cluster')
        ax4_1.set_ylabel('Total Distance')
        ax4_1.set_title('Cluster vs Distance')
        ax4_1.grid(True)
        
        # Feasibility pie chart
        ax4_2 = fig4.add_subplot(222)
        feasible_count = sum(feasible)
        infeasible_count = len(feasible) - feasible_count
        ax4_2.pie([feasible_count, infeasible_count], 
                labels=['Feasible', 'Tidak Feasible'],
                colors=['green', 'red'],
                autopct='%1.1f%%',
                shadow=True)
        ax4_2.set_title('Proporsi Solusi Feasible')
        
        # Execution time vs distance
        ax4_3 = fig4.add_subplot(223)
        exec_times = [r.get('execution_time', 0) for r in results]
        ax4_3.scatter(exec_times, distances, c=colors, alpha=0.7)
        ax4_3.set_xlabel('Waktu Eksekusi (detik)')
        ax4_3.set_ylabel('Total Distance')
        ax4_3.set_title('Waktu Eksekusi vs Jarak')
        ax4_3.grid(True)
        
        # Histogram of distances
        ax4_4 = fig4.add_subplot(224)
        ax4_4.hist(distances, bins=1, color='skyblue', edgecolor='black')
        ax4_4.axvline(x=np.mean(distances), color='r', linestyle='--', linewidth=2,
                    label=f'Mean: {np.mean(distances):.2f}')
        ax4_4.set_xlabel('Total Distance')
        ax4_4.set_ylabel('Frekuensi')
        ax4_4.set_title('Distribusi Jarak')
        ax4_4.legend()
        ax4_4.grid(True)
        
        fig4.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for suptitle
        
        # Add to UI
        chart_frame4 = ttk.Frame(scrollable_frame)
        chart_frame4.pack(fill="both", expand=True, padx=5, pady=5)
        
        canvas4 = FigureCanvasTkAgg(fig4, master=chart_frame4)
        canvas4.draw()
        canvas4.get_tk_widget().pack(fill="both", expand=True)
        
        # Add to figures list
        self.multi_start_figures.append(fig4)
        
        # Add buttons
        button_frame = ttk.Frame(scrollable_frame)
        button_frame.pack(fill="x", padx=5, pady=10)
        
        ttk.Button(
            button_frame,
            text="Export Hasil Multi-Start ke Excel",
            command=self.export_multi_start_results
        ).pack(side="left", padx=5)
        
        # Add download buttons
        ttk.Button(
            button_frame,
            text="Download Semua Visualisasi (PDF)",
            command=save_as_pdf
        ).pack(side="left", padx=5)
        
        ttk.Button(
            button_frame,
            text="Download Semua Visualisasi (Gambar Terpisah)",
            command=save_as_images
        ).pack(side="left", padx=5)
                    
    def export_single_cluster(self, cluster_number):
        """Export data for a single cluster to Excel file"""
        if not self.results:
            messagebox.showerror("Error", "Tidak ada hasil untuk diekspor")
            return
        
        # Find the requested cluster
        cluster_info = None
        for cluster in self.results['clusters']:
            if cluster['cluster_number'] == cluster_number:
                cluster_info = cluster
                break
        
        if not cluster_info:
            messagebox.showerror("Error", f"Cluster {cluster_number} tidak ditemukan")
            return
        
        file_path = filedialog.asksaveasfilename(
            defaultextension=".xlsx",
            filetypes=[("Excel files", "*.xlsx")],
            title=f"Simpan Data Cluster {cluster_number}"
        )
        
        if not file_path:
            return
        
        try:
            # Create summary data
            summary_data = [{
                'Algoritma': self.results['algorithm'],
                'Total TPS': self.results['total_tps'],
                'Total Volume (m³)': self.results['total_volume'],
                'Jumlah Cluster': self.results['num_clusters']
            }]

            if 'metrics' in self.results:
                metrics = self.results['metrics']
                if self.results.get('optimize_routes', False):
                    summary_data[0]['Total Jarak Rute (Antar TPS)'] = metrics['total_route_distance']
                    
                    # Add complete route distance if available
                    if 'total_complete_route_distance' in metrics:
                        summary_data[0]['Total Jarak Rute (Termasuk Garasi-TPA)'] = metrics['total_complete_route_distance']
                    
                    summary_data[0]['Jarak Rute Rata-rata per Cluster'] = metrics['avg_route_distance_per_cluster']
                else:
                    summary_data[0]['Total Jarak'] = metrics['total_distance']
                    summary_data[0]['Jarak Rata-rata per Cluster'] = metrics['avg_distance_per_cluster']
            
            # Create detailed TPS data
            tps_data = []
            
            # Calculate route distances
            tps_indices = cluster_info['tps_indices']
            
            for i, (idx, name, volume) in enumerate(zip(
                    cluster_info['tps_indices'], 
                    cluster_info['tps_names'],
                    cluster_info['tps_volumes'])):
                
                tps_row = {
                    'Urutan': i+1,
                    'Index TPS': idx,
                    'Nama TPS': name,
                    'Volume (m³)': volume
                }
                
                # Add distance to next TPS if not the last one
                if i < len(tps_indices) - 1:
                    next_idx = tps_indices[i+1]
                    distance = self.distance_matrix[idx][next_idx]
                    tps_row['Jarak ke Berikutnya'] = distance
                
                # Add coordinates if available
                if self.data_info.get('has_coords', False):
                    tps_row['Latitude'] = self.data_info['lat'][idx]
                    tps_row['Longitude'] = self.data_info['long'][idx]
                
                tps_data.append(tps_row)
            
            # Create Excel writer
            with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
                pd.DataFrame(summary_data).to_excel(writer, sheet_name='Ringkasan', index=False)
                pd.DataFrame(tps_data).to_excel(writer, sheet_name='Detail TPS', index=False)
            
            messagebox.showinfo("Sukses", f"Data Cluster {cluster_number} berhasil disimpan ke {file_path}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Gagal menyimpan file: {str(e)}")
    
    def export_centroids(self):
        """Export centroid data to Excel file"""
        centroids = self.calculate_cluster_centroids()
        
        if not centroids:
            messagebox.showerror("Error", "Tidak ada data titik tengah untuk diekspor")
            return
        
        file_path = filedialog.asksaveasfilename(
            defaultextension=".xlsx",
            filetypes=[("Excel files", "*.xlsx")],
            title="Simpan Data Titik Tengah"
        )
        
        if not file_path:
            return
        
        try:
            # Create DataFrame for centroids
            centroids_df = pd.DataFrame(centroids)
            
            # Save to Excel
            with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
                centroids_df.to_excel(writer, sheet_name='Titik Tengah', index=False)
            
            messagebox.showinfo("Sukses", f"Data titik tengah berhasil disimpan ke {file_path}")
            
            # Ask if user wants to open the file
            if messagebox.askyesno("Buka File", "Apakah Anda ingin membuka file Excel sekarang?"):
                import os
                os.startfile(file_path)
                
        except Exception as e:
            messagebox.showerror("Error", f"Gagal menyimpan file: {str(e)}")

    def export_fleet_assignment(self):
        """Export fleet assignment data to Excel with balanced workload"""
        fleet_data = self.cluster_centroids_by_fleet()
        
        if not fleet_data:
            messagebox.showerror("Error", "Tidak ada data penugasan armada untuk diekspor")
            return
        
        file_path = filedialog.asksaveasfilename(
            defaultextension=".xlsx",
            filetypes=[("Excel files", "*.xlsx")],
            title="Simpan Penugasan Armada"
        )
        
        if not file_path:
            return
        
        try:
            # Create data for all fleets with day of operation
            all_data = []
            fleet_days = {}  # Track which day for each fleet
            
            for centroid in fleet_data['centroids']:
                fleet_id = centroid['fleet']
                
                # Initialize days counter for this fleet
                if fleet_id not in fleet_days:
                    fleet_days[fleet_id] = 1
                else:
                    fleet_days[fleet_id] += 1
                
                # Calculate which day of the week this is for the fleet
                day_num = fleet_days[fleet_id]
                day_names = ["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu"]
                day_name = day_names[day_num-1] if day_num <= len(day_names) else f"Hari {day_num}"
                
                all_data.append({
                    'Cluster': centroid['cluster'],
                    'Armada': centroid['fleet'],
                    'Longitude': centroid['long'],
                    'Latitude': centroid['lat'],
                    'Hari Operasi': day_name
                })
            
            # Create DataFrame
            all_df = pd.DataFrame(all_data)
            
            # Create Excel writer
            with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
                # Write all data to sheet
                all_df.to_excel(writer, sheet_name='Semua Armada', index=False)
                
                # Create sheet for each fleet
                for fleet_id, clusters in fleet_data['fleet_clusters'].items():
                    clusters_items = []  # Gunakan nama variabel yang berbeda
                    
                    for i, centroid in enumerate(clusters):
                        # Map to day of week
                        day_names = ["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu"]
                        day_name = day_names[i] if i < len(day_names) else f"Hari {i+1}"
                        
                        clusters_items.append({
                            'Cluster': centroid['cluster'],
                            'Longitude': centroid['long'],
                            'Latitude': centroid['lat'],
                            'Hari Operasi': day_name
                        })
                    
                    # Create DataFrame and write to sheet
                    fleet_df = pd.DataFrame(clusters_items)
                    fleet_df.to_excel(writer, sheet_name=f'Armada {fleet_id}', index=False)
                
                # Add summary sheet
                summary_data = [{
                    'Jumlah Armada': fleet_data['num_trucks'],
                    'Jumlah Cluster': len(fleet_data['centroids']),
                    'Tanggal Export': time.strftime('%d-%m-%Y %H:%M:%S'),
                    'Kapasitas Per Armada': '6 cluster/hari'
                }]
                
                # Fleet cluster count summary
                for fleet_id, clusters in fleet_data['fleet_clusters'].items():
                    day_usage = len(clusters)
                    day_remaining = 6 - day_usage
                    summary_data[0][f'Cluster Armada {fleet_id}'] = day_usage
                    summary_data[0][f'Sisa Hari Armada {fleet_id}'] = day_remaining if day_remaining >= 0 else "OVERLOAD"
                
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, sheet_name='Ringkasan', index=False)
                
                # Add detailed cluster info if available
                if self.results and 'clusters' in self.results:
                    detailed_data = []
                    
                    # Map fleet id to each cluster
                    cluster_to_fleet = {c['cluster']: c['fleet'] for c in fleet_data['centroids']}
                    
                    # Track day allocation
                    fleet_day_map = {}
                    
                    for cluster_info in self.results['clusters']:
                        cluster_num = cluster_info['cluster_number']
                        fleet_id = cluster_to_fleet.get(cluster_num, "N/A")
                        
                        # Determine day of operation
                        day_name = "N/A"
                        if fleet_id != "N/A":
                            if fleet_id not in fleet_day_map:
                                fleet_day_map[fleet_id] = 0
                                
                            fleet_day_map[fleet_id] += 1
                            day_num = fleet_day_map[fleet_id]
                            day_names = ["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu"]
                            day_name = day_names[day_num-1] if day_num <= len(day_names) else f"Hari {day_num}"
                        
                        detailed_data.append({
                            'Cluster': cluster_num,
                            'Armada': fleet_id,
                            'Hari Operasi': day_name,
                            'Jumlah TPS': cluster_info['num_tps'],
                            'Volume Total (m³)': cluster_info['total_volume'],
                            'TPS': ', '.join(cluster_info['tps_names'])
                        })
                    
                    # Create DataFrame
                    detailed_df = pd.DataFrame(detailed_data)
                    detailed_df.to_excel(writer, sheet_name='Detail Cluster', index=False)
            
            messagebox.showinfo("Sukses", f"Data penugasan armada berhasil disimpan ke {file_path}")
            
            # Ask if user wants to open the file
            if messagebox.askyesno("Buka File", "Apakah Anda ingin membuka file Excel sekarang?"):
                import os
                os.startfile(file_path)
                
        except Exception as e:
            messagebox.showerror("Error", f"Gagal menyimpan file: {str(e)}")

    def export_results(self):
        """Export results to Excel file with improved formatting"""
        if not self.results:
            messagebox.showerror("Error", "Tidak ada hasil untuk diekspor")
            return
            
        file_path = filedialog.asksaveasfilename(
            defaultextension=".xlsx",
            filetypes=[("Excel files", "*.xlsx")],
            title="Simpan Hasil Clustering"
        )
        
        if not file_path:
            return
            
        try:
            # Gunakan ExcelWriter dengan engine openpyxl
            from pandas import ExcelWriter
            
            with ExcelWriter(file_path, engine='openpyxl') as writer:
                # 1. Sheet Ringkasan
                summary_data = {
                    'Algoritma': [self.results['algorithm']],
                    'Total TPS': [self.results['total_tps']],
                    'Volume Total (m³)': [self.results['total_volume']],
                    'Jumlah Cluster': [self.results['num_clusters']],
                    'Waktu Eksekusi (detik)': [self.results['execution_time']]
                }
                
                metrics = self.results.get('metrics', {})
                summary_data['Total Jarak Intra-Cluster'] = [metrics.get('total_distance', 0)]
                summary_data['Total Jarak Rute (TPS-TPS)'] = [metrics.get('total_route_distance', 0)]
                
                if metrics.get('use_fixed_endpoints', False):
                    summary_data['Total Jarak Rute Lengkap (Garasi-TPS-TPA)'] = [metrics.get('total_complete_route_distance', 0)]
                
                # Tambahkan info multi-start
                if 'multi_start_info' in self.results:
                    ms_info = self.results['multi_start_info']
                    summary_data['Seed Terbaik'] = [ms_info.get('best_seed', '-')]
                    summary_data['Jumlah Run Multi-Start'] = [ms_info.get('num_runs', 0)]
                    summary_data['Solusi Feasible'] = ['Ya' if ms_info.get('best_feasible', False) else 'Tidak']
                    summary_data['Waktu Total Multi-Start (detik)'] = [ms_info.get('total_process_time', 0)]
                
                # Export ringkasan
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, sheet_name='Ringkasan', index=False)
                
                # 2. Sheet Ringkasan Cluster
                clusters_data = []
                for i, cluster_info in enumerate(self.results['clusters']):
                    cluster_metric = next(
                        (m for m in metrics.get('cluster_metrics', []) if m.get('cluster_id') == cluster_info['cluster_number']),
                        None
                    )
                    
                    cluster_row = {
                        'Cluster': cluster_info['cluster_number'],
                        'Jumlah TPS': cluster_info['num_tps'],
                        'Volume (m³)': cluster_info['total_volume'],
                        'Persentase Volume (%)': (cluster_info['total_volume'] / self.results['total_volume']) * 100
                    }
                    
                    if cluster_metric:
                        cluster_row['Avg Jarak Internal'] = cluster_metric.get('avg_distance', 0)
                        cluster_row['Jarak Rute'] = cluster_metric.get('route_distance', 0)
                        
                        if metrics.get('use_fixed_endpoints', False):
                            cluster_row['Jarak Garasi-TPS'] = cluster_metric.get('garage_to_first', 0)
                            cluster_row['Jarak TPS-TPA'] = cluster_metric.get('last_to_tpa', 0)
                            cluster_row['Jarak Total (G-TPS-T)'] = cluster_metric.get('complete_route_distance', 0)
                    
                    clusters_data.append(cluster_row)
                
                clusters_df = pd.DataFrame(clusters_data)
                clusters_df.to_excel(writer, sheet_name='Ringkasan Cluster', index=False)
                
                # 3. Sheet Detail Cluster (semua TPS)
                details_data = []
                for cluster_info in self.results['clusters']:
                    cluster_num = cluster_info['cluster_number']
                    
                    for i, (idx, name, volume) in enumerate(zip(
                            cluster_info['tps_indices'], 
                            cluster_info['tps_names'],
                            cluster_info['tps_volumes'])):
                        
                        tps_row = {
                            'Cluster': cluster_num,
                            'Urutan': i+1,
                            'Index TPS': idx,
                            'Nama TPS': name,
                            'Volume (m³)': volume
                        }
                        
                        # Jarak ke TPS berikutnya jika bukan yang terakhir
                        if i < len(cluster_info['tps_indices']) - 1:
                            next_idx = cluster_info['tps_indices'][i+1]
                            tps_row['Jarak ke Berikutnya'] = self.distance_matrix[idx][next_idx]
                        
                        # Tambahkan koordinat jika tersedia
                        if self.data_info.get('has_coords', False):
                            tps_row['Latitude'] = self.data_info['lat'][idx]
                            tps_row['Longitude'] = self.data_info['long'][idx]
                        
                        details_data.append(tps_row)
                
                details_df = pd.DataFrame(details_data)
                details_df.to_excel(writer, sheet_name='Detail Cluster', index=False)
                
                # 4. Sheet terpisah untuk setiap cluster
                for cluster_info in self.results['clusters']:
                    cluster_num = cluster_info['cluster_number']
                    sheet_name = f'Cluster {cluster_num}'
                    
                    # Pastikan nama sheet tidak lebih dari 31 karakter (batas Excel)
                    if len(sheet_name) > 31:
                        sheet_name = sheet_name[:31]
                    
                    # Data untuk cluster ini
                    cluster_data = []
                    for i, (idx, name, volume) in enumerate(zip(
                            cluster_info['tps_indices'], 
                            cluster_info['tps_names'],
                            cluster_info['tps_volumes'])):
                        
                        row = {
                            'Urutan': i+1,
                            'Nama TPS': name,
                            'Volume (m³)': volume,
                            'Index TPS': idx
                        }
                        
                        # Jarak ke TPS berikutnya
                        if i < len(cluster_info['tps_indices']) - 1:
                            next_idx = cluster_info['tps_indices'][i+1]
                            row['Jarak ke Berikutnya'] = self.distance_matrix[idx][next_idx]
                        
                        # Koordinat
                        if self.data_info.get('has_coords', False):
                            row['Latitude'] = self.data_info['lat'][idx]
                            row['Longitude'] = self.data_info['long'][idx]
                        
                        cluster_data.append(row)
                    
                    cluster_df = pd.DataFrame(cluster_data)
                    cluster_df.to_excel(writer, sheet_name=sheet_name, index=False)
                
                # 5. Sheet untuk hasil Multi-Start
                if hasattr(self, 'multi_start_results') and self.multi_start_results:
                    ms_data = []
                    for i, result in enumerate(self.multi_start_results):
                        ms_data.append({
                            'Run': i+1,
                            'Seed': result.get('seed', i),
                            'Jumlah Cluster': result.get('num_clusters', 0),
                            'Jarak Total': result.get('distance', float('inf')),
                            'Feasible': 'Ya' if result.get('feasible', False) else 'Tidak',
                            'Waktu Eksekusi (s)': result.get('execution_time', 0),
                        })
                    
                    ms_df = pd.DataFrame(ms_data)
                    ms_df.to_excel(writer, sheet_name='Multi-Start Results', index=False)
                    
                    # Tambahkan statistik multi-start
                    feasible_results = [r for r in self.multi_start_results if r.get('feasible', False)]
                    feasible_distances = [r.get('distance', float('inf')) for r in feasible_results] if feasible_results else []
                    
                    stats_data = [{
                        'Total Run': len(self.multi_start_results),
                        'Run Feasible': len(feasible_results),
                        'Run Tidak Feasible': len(self.multi_start_results) - len(feasible_results)
                    }]
                    
                    if feasible_distances:
                        stats_data[0].update({
                            'Jarak Minimum': min(feasible_distances),
                            'Jarak Maksimum': max(feasible_distances),
                            'Jarak Rata-rata': sum(feasible_distances)/len(feasible_distances)
                        })
                        
                        if len(feasible_distances) > 1:
                            stats_data[0].update({
                                'Standar Deviasi': np.std(feasible_distances),
                                'Koefisien Variasi (%)': np.std(feasible_distances)/np.mean(feasible_distances)*100
                            })
                    
                    stats_data[0].update({
                        'Waktu Total (s)': sum([r.get('execution_time', 0) for r in self.multi_start_results]),
                        'Waktu Rata-rata (s)': sum([r.get('execution_time', 0) for r in self.multi_start_results])/len(self.multi_start_results)
                    })
                    
                    stats_df = pd.DataFrame(stats_data)
                    stats_df.to_excel(writer, sheet_name='Statistik Multi-Start', index=False)
            
            messagebox.showinfo("Sukses", f"Hasil berhasil disimpan ke {file_path}")
            
            # Tanya jika ingin membuka file
            if messagebox.askyesno("Buka File", "Apakah Anda ingin membuka file Excel sekarang?"):
                import os
                os.startfile(file_path)
                
        except Exception as e:
            messagebox.showerror("Error", f"Gagal menyimpan file: {str(e)}")
    
    def visualize_routes(self):
        """Visualize the routes for each cluster if coordinate data is available"""
        # Reset route figures list
        self.route_figures = []
        
        # Check if using fixed endpoints
        use_fixed_endpoints = self.use_fixed_endpoints_var.get()
        start_point_name = self.start_point_var.get() if use_fixed_endpoints else None
        end_point_name = self.end_point_var.get() if use_fixed_endpoints else None
        
        # Find indices of start and end points
        start_idx = None
        end_idx = None
        
        if use_fixed_endpoints and self.data_info and 'names' in self.data_info:
            names = self.data_info['names']
            
            # Try to find start point (Garage)
            if start_point_name in names:
                # Exact match
                for i, name in enumerate(names):
                    if name == start_point_name:
                        start_idx = i
                        break
            else:
                # Partial match
                for i, name in enumerate(names):
                    if start_point_name.lower() in name.lower():
                        start_idx = i
                        break
            
            # Try to find end point (TPA)
            if end_point_name in names:
                # Exact match
                for i, name in enumerate(names):
                    if name == end_point_name:
                        end_idx = i
                        break
            else:
                # Partial match
                for i, name in enumerate(names):
                    if end_point_name.lower() in name.lower():
                        end_idx = i
                        break
        
        if not self.results or not self.data_info or not self.data_info.get('has_coords', False):
            ttk.Label(self.route_tab, text="Tidak dapat menampilkan rute: Tidak ada data koordinat").pack(padx=10, pady=10)
            return False
        
        # Clear route visualization tab
        for widget in self.route_tab.winfo_children():
            widget.destroy()

        # Create frame for download buttons
        download_buttons_frame = ttk.Frame(self.route_tab)
        download_buttons_frame.pack(fill="x", pady=10)

        # Create notebook for tabs
        route_notebook = ttk.Notebook(self.route_tab)
        route_notebook.pack(fill="both", expand=True, padx=5, pady=5)
        
        ttk.Button(
            download_buttons_frame, 
            text="Download Semua Rute (PDF Enhanced)",
            command=self.enhanced_save_all_routes
        ).pack(side="right", padx=10)
        
        ttk.Button(
            download_buttons_frame, 
            text="Download Semua Rute (Gambar Terpisah)", 
            command=self.save_all_routes_as_images
        ).pack(side="right", padx=10)
        
        # Create tab for each cluster
        for i, cluster_info in enumerate(self.results['clusters']):
            cluster_indices = cluster_info['tps_indices']
            cluster_tab = ttk.Frame(route_notebook)
            route_notebook.add(cluster_tab, text=f"Cluster {cluster_info['cluster_number']}")
            
            # Create scrollable frame for tab content
            tab_frame = ttk.Frame(cluster_tab)
            tab_frame.pack(fill="both", expand=True, padx=5, pady=5)
            
            # Create canvas with scrollbars
            canvas = tk.Canvas(tab_frame)
            
            v_scrollbar = ttk.Scrollbar(tab_frame, orient="vertical", command=canvas.yview)
            v_scrollbar.pack(side="right", fill="y")
            
            h_scrollbar = ttk.Scrollbar(tab_frame, orient="horizontal", command=canvas.xview)
            h_scrollbar.pack(side="bottom", fill="x")
            
            canvas.configure(xscrollcommand=h_scrollbar.set, yscrollcommand=v_scrollbar.set)
            canvas.pack(side="left", fill="both", expand=True)
            
            # Create frame in canvas for content
            content_frame = ttk.Frame(canvas)
            canvas_window = canvas.create_window((0, 0), window=content_frame, anchor="nw")
            
            # Event handler for resize
            def config_canvas(event, _canvas=canvas, _window=canvas_window):
                _canvas.configure(scrollregion=_canvas.bbox("all"))
                _canvas.itemconfig(_window, width=event.width)
            
            content_frame.bind("<Configure>", config_canvas)
            
            # Figure size based on number of TPS
            num_tps = len(cluster_indices)
            width = 14
            height = 10 + min(num_tps * 0.5, 10)
            
            # Create figure with 2 rows (map on top, table below)
            fig = plt.Figure(figsize=(width, height), dpi=100)
            self.route_figures.append(fig)
            
            # Layout with map on top and table below
            grid = fig.add_gridspec(2, 1, height_ratios=[4, 1])  # 4:1 ratio for map:table
            ax = fig.add_subplot(grid[0])  # Map in first row
            
            # Extract coordinates for this cluster
            lats = [self.data_info['lat'][idx] for idx in cluster_indices]
            longs = [self.data_info['long'][idx] for idx in cluster_indices]
            volumes = [self.data_info['volumes'][idx] for idx in cluster_indices]
            names = [self.data_info['names'][idx] for idx in cluster_indices]
            
            # Normalize volumes for scatter plot size
            if volumes:
                min_vol = min(volumes)
                max_vol = max(volumes)
                if max_vol > min_vol:
                    norm_volumes = [50 + 200 * (v - min_vol) / (max_vol - min_vol) for v in volumes]
                else:
                    norm_volumes = [100] * len(volumes)
            else:
                norm_volumes = [100] * len(volumes)
            
            # Collect ALL coordinates including Garage and TPA
            all_longs = longs.copy()
            all_lats = lats.copy()
            
            # Add Garage and TPA coordinates if using fixed endpoints
            if use_fixed_endpoints and start_idx is not None and end_idx is not None:
                garage_lat = self.data_info['lat'][start_idx]
                garage_long = self.data_info['long'][start_idx]
                all_longs.append(garage_long)
                all_lats.append(garage_lat)
                
                tpa_lat = self.data_info['lat'][end_idx]
                tpa_long = self.data_info['long'][end_idx]
                all_longs.append(tpa_long)
                all_lats.append(tpa_lat)
            
            # Set visualization bounds with padding
            if all_longs and all_lats:
                min_long, max_long = min(all_longs), max(all_longs)
                min_lat, max_lat = min(all_lats), max(all_lats)
                
                long_padding = 0.10 * (max_long - min_long) if max_long != min_long else 0.002
                lat_padding = 0.10 * (max_lat - min_lat) if max_lat != min_lat else 0.002
                
                ax.set_xlim(min_long - long_padding, max_long + long_padding)
                ax.set_ylim(min_lat - lat_padding, max_lat + lat_padding)
            
            # Plot TPS locations
            min_len = min(len(longs), len(lats), len(norm_volumes)) if longs and lats and norm_volumes else 0
            if min_len > 0:
                longs = longs[:min_len]
                lats = lats[:min_len]
                norm_volumes = norm_volumes[:min_len]
                
                scatter = ax.scatter(longs, lats, s=norm_volumes, c='blue', alpha=0.6, marker='o')
            
            # Plot route if more than one point
            if len(cluster_indices) > 1:
                for j in range(len(cluster_indices)-1):
                    ax.plot([longs[j], longs[j+1]], [lats[j], lats[j+1]], 'r-', linewidth=1.5)
            
            # Add Garage and TPA points if using fixed endpoints
            if use_fixed_endpoints and start_idx is not None and end_idx is not None:
                garage_lat = self.data_info['lat'][start_idx]
                garage_long = self.data_info['long'][start_idx]
                ax.plot(garage_long, garage_lat, 'bs', markersize=12, label='Garasi')  # Blue square
                
                tpa_lat = self.data_info['lat'][end_idx]
                tpa_long = self.data_info['long'][end_idx]
                ax.plot(tpa_long, tpa_lat, 'r^', markersize=12, label='TPA')  # Red triangle
                
                # Draw line from Garage to first TPS if there are any TPS
                if len(cluster_indices) > 0:
                    first_idx = 0
                    first_tps_long = longs[first_idx]
                    first_tps_lat = lats[first_idx]
                    ax.plot([garage_long, first_tps_long], [garage_lat, first_tps_lat], 'b-', linewidth=1.5)
                    
                    # Draw line from last TPS to TPA
                    last_idx = len(cluster_indices) - 1
                    last_tps_long = longs[last_idx]
                    last_tps_lat = lats[last_idx]
                    ax.plot([last_tps_long, tpa_long], [last_tps_lat, tpa_lat], 'r-', linewidth=1.5)
                
                # Highlight start and end TPS
                if len(longs) > 0 and len(lats) > 0:
                    ax.plot(longs[0], lats[0], 'go', markersize=10)  # Start point
                    ax.plot(longs[-1], lats[-1], 'ro', markersize=8)  # End point
                
                # Add direction arrows
                for j in range(len(cluster_indices)-1):
                    # Calculate middle position of segment for arrow
                    mid_x = (longs[j] + longs[j+1]) / 2
                    mid_y = (lats[j] + lats[j+1]) / 2
                    
                    # Calculate direction
                    dx = longs[j+1] - longs[j]
                    dy = lats[j+1] - lats[j]
                    
                    # Add arrow annotation
                    ax.annotate('', 
                            xy=(mid_x + dx/10, mid_y + dy/10),  # Arrow end point
                            xytext=(mid_x - dx/10, mid_y - dy/10),  # Arrow start point
                            arrowprops=dict(arrowstyle='->', color='red', linewidth=1.5),
                            annotation_clip=False)
                
                # Add sequence numbers to show route order
                for j, (x, y) in enumerate(zip(longs, lats)):
                    ax.text(x, y, f"{j+1}", fontsize=10, ha='center', va='center',
                        bbox=dict(boxstyle="circle", fc="white", ec="black", alpha=0.7))
            
            # TPS info in bottom right corner
            info_text = "Urutan TPS:\n"
            for j, (idx, name) in enumerate(zip(cluster_indices, names)):
                short_name = name
                if len(short_name) > 20:
                    short_name = short_name[:18] + "..."
                info_text += f"{j+1}. {short_name} ({idx})\n"
            
            props = dict(boxstyle='round', facecolor='white', alpha=0.8)
            ax.text(0.98, 0.02, info_text, 
                    transform=ax.transAxes,
                    horizontalalignment='right',
                    verticalalignment='bottom',
                    bbox=props,
                    fontsize=9)
            
            # Calculate route distance
            route_distance = 0
            for j in range(len(cluster_indices)-1):
                route_distance += self.distance_matrix[cluster_indices[j]][cluster_indices[j+1]]
            
            # Set title
            title_text = f"Cluster {cluster_info['cluster_number']}: {len(cluster_indices)} TPS, {cluster_info['total_volume']:.2f} m³, Jarak Rute: {route_distance:.2f}"
            ax.set_title(title_text, fontsize=12, wrap=True)
            
            # Axis labels
            ax.set_xlabel('Longitude')
            ax.set_ylabel('Latitude')
            
            # Add grid
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # Add legend if using Garage and TPA
            if use_fixed_endpoints and start_idx is not None and end_idx is not None:
                ax.legend(loc='best')
            
            # Create table below the map
            ax_table = fig.add_subplot(grid[1])
            ax_table.axis('off')
            
            # Prepare table data - combined Segment, TPS Name, Distance, Volume
            table_data = []
            
            # If there are TPS, process segments and TPS data
            if len(cluster_indices) > 0:
                # Process segments between TPS
                for j in range(len(cluster_indices)):
                    current_idx = cluster_indices[j]
                    
                    # Get TPS name and volume
                    tps_name = names[j] if j < len(names) else "?"
                    tps_volume = volumes[j] if j < len(volumes) else 0
                    
                    # For segment and distance
                    if j < len(cluster_indices) - 1:
                        # If not the last TPS, this is segment "j → j+1"
                        next_idx = cluster_indices[j+1]
                        segment = f"{j+1} → {j+2}"
                        distance = self.distance_matrix[current_idx][next_idx]
                    else:
                        # Last TPS to TPA
                        if use_fixed_endpoints and end_idx is not None:
                            segment = f"{j+1} → TPA"
                            distance = self.distance_matrix[current_idx][end_idx]
                        else:
                            # If no TPA, leave empty
                            segment = f"{j+1}"
                            distance = 0
                    
                    # Add data to table
                    table_data.append([
                        segment,
                        tps_name,
                        f"{distance:.2f}" if distance > 0 else "-",
                        f"{tps_volume:.2f}"
                    ])
                
                # Add total distance in last row if there are more than 1 TPS
                if len(cluster_indices) > 1:
                    table_data.append([
                        "Total",
                        "",
                        f"{route_distance:.2f}",
                        f"{cluster_info['total_volume']:.2f}"
                    ])
            
            # Create table
            if table_data:
                table = ax_table.table(
                    cellText=table_data,
                    colLabels=["Segmen", "Nama TPS", "Jarak", "Volume"],
                    loc='center',
                    cellLoc='center'
                )
                
                # Style table
                table.auto_set_font_size(False)
                table.set_fontsize(10)
                table.scale(1, 1.5)  # Row height
                
                # Set cell colors and properties
                for (row, col), cell in table.get_celld().items():
                    if row == 0:  # Header row
                        cell.set_text_props(fontweight='bold', color='white')
                        cell.set_facecolor('darkgreen')
                    elif row == len(table_data) and len(cluster_indices) > 1:  # Total row
                        cell.set_text_props(fontweight='bold')
                        cell.set_facecolor('lightgray')
                    else:
                        cell.set_facecolor('white')
                    
                    # Set column widths
                    if col == 0:  # Segment column
                        cell.set_width(0.15)
                    elif col == 1:  # TPS Name column
                        cell.set_width(0.55)  # More space for names
                    elif col == 2:  # Distance column
                        cell.set_width(0.15)
                    else:  # Volume column
                        cell.set_width(0.15)
                
                ax_table.set_title("Detail TPS", fontsize=12, fontweight='bold')
                
            # Layout padding
            fig.tight_layout(pad=2.5)
            
            # Create figure canvas and add to frame
            fig_canvas = FigureCanvasTkAgg(fig, master=content_frame)
            fig_canvas.draw()
            fig_canvas.get_tk_widget().pack(fill="both", expand=True, padx=5, pady=5)
            
            # Download button for this cluster
            download_frame = ttk.Frame(content_frame)
            download_frame.pack(fill="x", pady=10)
            
            ttk.Label(download_frame, text=f"Gambar Cluster {cluster_info['cluster_number']}").pack(side="left", padx=5)
            
            ttk.Button(
                download_frame, 
                text="Download Gambar Ini", 
                command=lambda fig=fig, c=cluster_info['cluster_number']: self.save_single_route_image(fig, f"rute_cluster_{c}")
            ).pack(side="right", padx=5)
        
        return True
    
    def visualize_fleet_assignments(self, parent_frame, fleet_data):
        """Create a visualization of fleet assignments with balanced workload"""
        # Create a figure
        fig = plt.Figure(figsize=(10, 8), dpi=100)
        ax = fig.add_subplot(111)
        
        # Define colors for fleets
        colors = ['red', 'blue', 'green', 'purple', 'orange', 'cyan', 
                'magenta', 'yellow', 'black', 'brown', 'pink', 'gray']
        markers = ['o', 's', '^', 'D', 'v', '>', '<', 'p', '*', 'h', 'H', '+']
        
        # Plot each fleet with different color
        for fleet_id, clusters in fleet_data['fleet_clusters'].items():
            # Get color and marker for this fleet
            color = colors[(fleet_id-1) % len(colors)]
            marker = markers[(fleet_id-1) % len(markers)]
            
            # Extract coordinates
            longs = [c['long'] for c in clusters]
            lats = [c['lat'] for c in clusters]
            cluster_nums = [c['cluster'] for c in clusters]
            
            # Plot centroids
            scatter = ax.scatter(longs, lats, c=color, marker=marker, s=100, 
                                label=f'Armada {fleet_id} ({len(clusters)}/6 hari)')
            
            # Add cluster numbers as labels
            for i, (x, y, num) in enumerate(zip(longs, lats, cluster_nums)):
                ax.annotate(str(num), (x, y), fontsize=8, ha='center', va='center',
                        backgroundcolor='white', bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))
        
        # Add legend
        ax.legend(loc='upper right')
        
        # Labels and title
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_title('Visualisasi Penugasan Armada Pengangkutan (Maks 6 Cluster/Armada)')
        ax.grid(True)
        
        # Add to UI
        canvas = FigureCanvasTkAgg(fig, master=parent_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)
        
        # Add download button
        ttk.Button(
            parent_frame,
            text="Download Visualisasi",
            command=lambda: self.save_visualization([fig], "fleet_assignment_")
        ).pack(pady=10)

    def visualize_centroids(self):
        """Create a visualization of cluster centroids"""
        centroids = self.calculate_cluster_centroids()
        
        if not centroids:
            messagebox.showerror("Error", "Tidak ada data titik tengah untuk visualisasi")
            return
        
        # Create popup window
        popup = tk.Toplevel(self.root)
        popup.title("Visualisasi Titik Tengah Cluster")
        popup.geometry("800x600")
        
        # Create figure
        fig = plt.Figure(figsize=(10, 8), dpi=100)
        ax = fig.add_subplot(111)
        
        # Extract coordinates
        longs = [c['long'] for c in centroids]
        lats = [c['lat'] for c in centroids]
        cluster_nums = [c['cluster'] for c in centroids]
        
        # Plot centroids
        scatter = ax.scatter(longs, lats, c='blue', s=100)
        
        # Add cluster numbers as labels
        for i, (x, y, num) in enumerate(zip(longs, lats, cluster_nums)):
            ax.annotate(str(num), (x, y), fontsize=9, ha='center', va='center',
                    backgroundcolor='white', bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))
        
        # Labels and title
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_title('Visualisasi Titik Tengah Cluster')
        ax.grid(True)
        
        # Add to UI
        canvas = FigureCanvasTkAgg(fig, master=popup)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)
        
        # Add download button
        ttk.Button(
            popup,
            text="Download Visualisasi",
            command=lambda: self.save_visualization([fig], "centroids_")
        ).pack(pady=10)

    def display_centroids_tab(self):
        """Display the centroid coordinates in a new tab"""
        # Clear the tab first
        for widget in self.centroid_tab.winfo_children():
            widget.destroy()
        
        # Calculate centroids
        centroids = self.calculate_cluster_centroids()
        
        if not centroids:
            ttk.Label(self.centroid_tab, text="Tidak dapat menampilkan titik tengah: Tidak ada data koordinat").pack(padx=10, pady=10)
            return
        
        # Create main container
        main_container = ttk.Frame(self.centroid_tab)
        main_container.pack(fill="both", expand=True)
        
        # Create control panel frame
        control_frame = ttk.Frame(main_container)
        control_frame.pack(fill="x", padx=10, pady=5)
        
        # Add title
        ttk.Label(
            control_frame, 
            text="Titik Tengah (Centroid) Setiap Cluster", 
            font=("Arial", 12, "bold")
        ).pack(side="left", pady=5)
        
        # Add fleet assignment controls
        fleet_frame = ttk.LabelFrame(control_frame, text="Penugasan Armada")
        fleet_frame.pack(side="right", padx=5, pady=5)
        
        ttk.Label(fleet_frame, text="Jumlah Armada:").grid(row=0, column=0, padx=5, pady=5)
        
        # Create spinbox for truck count
        truck_spinbox = ttk.Spinbox(
            fleet_frame,
            from_=1,
            to=20,
            textvariable=self.num_trucks_var,
            width=5
        )
        truck_spinbox.grid(row=0, column=1, padx=5, pady=5)
        
        # Add assign button
        ttk.Button(
            fleet_frame,
            text="Kelompokkan ke Armada",
            command=self.display_fleet_assignment
        ).grid(row=0, column=2, padx=5, pady=5)
        
        # Create content frame
        content_frame = ttk.Frame(main_container)
        content_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        # Create table frame
        table_frame = ttk.Frame(content_frame)
        table_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Create treeview for centroids table
        columns = ("Cluster", "Long", "Lat")
        tree = ttk.Treeview(table_frame, columns=columns, show="headings", height=15)
        
        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, anchor="center", width=100)
        
        # Add data to table
        for centroid in centroids:
            tree.insert("", "end", values=(
                centroid['cluster'],
                "{:.7f}".format(centroid['long']),
                "{:.7f}".format(centroid['lat'])
            ))
        
        # Add scrollbars
        v_scrollbar = ttk.Scrollbar(table_frame, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=v_scrollbar.set)
        v_scrollbar.pack(side="right", fill="y")
        
        h_scrollbar = ttk.Scrollbar(table_frame, orient="horizontal", command=tree.xview)
        tree.configure(xscrollcommand=h_scrollbar.set)
        h_scrollbar.pack(side="bottom", fill="x")
        
        tree.pack(fill="both", expand=True)
        
        # Add export button frame
        button_frame = ttk.Frame(content_frame)
        button_frame.pack(fill="x", pady=10)
        
        ttk.Button(
            button_frame,
            text="Download Titik Tengah (Excel)",
            command=self.export_centroids
        ).pack(side="left", padx=5)
        
        ttk.Button(
            button_frame,
            text="Visualisasi Titik Tengah",
            command=self.visualize_centroids
        ).pack(side="right", padx=5)


    def display_fleet_assignment(self):
        """Display the fleet assignment of clusters in the Centroid tab with balanced workload"""
        # Clear the centroid tab first
        for widget in self.centroid_tab.winfo_children():
            widget.destroy()
        
        # Get fleet assignments
        fleet_data = self.cluster_centroids_by_fleet()
        
        if not fleet_data:
            ttk.Label(self.centroid_tab, text="Tidak dapat menampilkan penugasan armada").pack(padx=10, pady=10)
            return
        
        # Create main container
        main_container = ttk.Frame(self.centroid_tab)
        main_container.pack(fill="both", expand=True)
        
        # Add title and info
        title_frame = ttk.Frame(main_container)
        title_frame.pack(fill="x", padx=10, pady=5)
        
        ttk.Label(
            title_frame, 
            text=f"Penugasan Armada Pengangkutan ({fleet_data['num_trucks']} Armada, Maks 6 Cluster/Armada)",
            font=("Arial", 14, "bold")
        ).pack(side="left", pady=5)
        
        # Add button for export
        export_button = ttk.Button(
            title_frame,
            text="Export Penugasan Armada (Excel)",
            command=self.export_fleet_assignment
        )
        export_button.pack(side="right", padx=5, pady=5)
        
        # Create notebook for fleet tabs
        fleet_notebook = ttk.Notebook(main_container)
        fleet_notebook.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Create "All Fleets" tab
        all_fleets_tab = ttk.Frame(fleet_notebook)
        fleet_notebook.add(all_fleets_tab, text="Semua Armada")
        
        # Create table for all fleets
        columns = ("Cluster", "Armada", "Long", "Lat", "Hari Operasi")
        all_tree = ttk.Treeview(all_fleets_tab, columns=columns, show="headings", height=20)
        
        for col in columns:
            all_tree.heading(col, text=col)
            all_tree.column(col, anchor="center", width=100)
        
        # Adjust column widths
        all_tree.column("Armada", width=70)
        all_tree.column("Long", width=120)
        all_tree.column("Lat", width=120)
        all_tree.column("Hari Operasi", width=80)
        
        # Add scrollbars
        v_scrollbar = ttk.Scrollbar(all_fleets_tab, orient="vertical", command=all_tree.yview)
        all_tree.configure(yscrollcommand=v_scrollbar.set)
        v_scrollbar.pack(side="right", fill="y")
        
        h_scrollbar = ttk.Scrollbar(all_fleets_tab, orient="horizontal", command=all_tree.xview)
        all_tree.configure(xscrollcommand=h_scrollbar.set)
        h_scrollbar.pack(side="bottom", fill="x")
        
        all_tree.pack(fill="both", expand=True)
        
        # Add data to table with day of operation
        # Track days for each fleet
        fleet_days = {}
        
        for centroid in fleet_data['centroids']:
            fleet_id = centroid['fleet']
            
            # Initialize days counter for this fleet
            if fleet_id not in fleet_days:
                fleet_days[fleet_id] = 1
            else:
                fleet_days[fleet_id] += 1
            
            # Calculate which day of the week this is for the fleet
            day_num = fleet_days[fleet_id]
            day_names = ["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu"]
            day_name = day_names[day_num-1] if day_num <= len(day_names) else f"Hari {day_num}"
            
            all_tree.insert("", "end", values=(
                centroid['cluster'],
                centroid['fleet'],
                "{:.7f}".format(centroid['long']),
                "{:.7f}".format(centroid['lat']),
                day_name
            ))
        
        # Create tab for each fleet
        for fleet_id, clusters in fleet_data['fleet_clusters'].items():
            fleet_tab = ttk.Frame(fleet_notebook)
            fleet_notebook.add(fleet_tab, text=f"Armada {fleet_id}")
            
            # Add info about this fleet
            ttk.Label(
                fleet_tab,
                text=f"Armada {fleet_id} melayani {len(clusters)}/6 cluster (hari)",
                font=("Arial", 12, "bold")
            ).pack(anchor="w", padx=10, pady=5)
            
            # Create table for this fleet
            fleet_columns = ("Cluster", "Long", "Lat", "Hari Operasi")
            fleet_tree = ttk.Treeview(fleet_tab, columns=fleet_columns, show="headings", height=15)
            
            for col in fleet_columns:
                fleet_tree.heading(col, text=col)
                fleet_tree.column(col, anchor="center", width=100)
            
            # Adjust column widths
            fleet_tree.column("Long", width=120)
            fleet_tree.column("Lat", width=120)
            fleet_tree.column("Hari Operasi", width=80)
            
            # Add scrollbars
            fleet_v_scrollbar = ttk.Scrollbar(fleet_tab, orient="vertical", command=fleet_tree.yview)
            fleet_tree.configure(yscrollcommand=fleet_v_scrollbar.set)
            fleet_v_scrollbar.pack(side="right", fill="y")
            
            fleet_h_scrollbar = ttk.Scrollbar(fleet_tab, orient="horizontal", command=fleet_tree.xview)
            fleet_tree.configure(xscrollcommand=fleet_h_scrollbar.set)
            fleet_h_scrollbar.pack(side="bottom", fill="x")
            
            fleet_tree.pack(fill="both", expand=True, padx=10, pady=5)
            
            # Add data to table with day of operation
            for i, centroid in enumerate(clusters):
                # Map to day of week
                day_names = ["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu"]
                day_name = day_names[i] if i < len(day_names) else f"Hari {i+1}"
                
                fleet_tree.insert("", "end", values=(
                    centroid['cluster'],
                    "{:.7f}".format(centroid['long']),
                    "{:.7f}".format(centroid['lat']),
                    day_name
                ))
        
        # Create visualization tab
        viz_tab = ttk.Frame(fleet_notebook)
        fleet_notebook.add(viz_tab, text="Visualisasi")
        
        # Create map visualization of fleet assignments
        self.visualize_fleet_assignments(viz_tab, fleet_data)

    def enhanced_save_all_routes(self):
        """Versi yang ditingkatkan untuk mengekspor semua rute ke PDF dengan pendekatan sederhana"""
        if not self.route_figures:
            messagebox.showinfo("Info", "Tidak ada gambar rute yang dapat disimpan")
            return
                
        file_path = filedialog.asksaveasfilename(
            defaultextension=".pdf",
            filetypes=[("PDF files", "*.pdf")],
            title="Simpan Semua Rute (PDF)"
        )
        
        if not file_path:
            return
        
        # Tampilkan progress dialog
        progress_dialog = tk.Toplevel(self.root)
        progress_dialog.title("Export PDF Progress")
        progress_dialog.geometry("400x150")
        progress_dialog.transient(self.root)
        progress_dialog.grab_set()
        
        # Center dialog
        progress_dialog.update_idletasks()
        x = (self.root.winfo_width() // 2) - (400 // 2)
        y = (self.root.winfo_height() // 2) - (150 // 2)
        progress_dialog.geometry(f"+{x}+{y}")
        
        # Dialog content
        ttk.Label(progress_dialog, text="Mengekspor semua rute ke PDF...", font=("Arial", 11, "bold")).pack(pady=(15, 10))
        
        progress_var = tk.DoubleVar(value=0)
        progress_bar = ttk.Progressbar(progress_dialog, variable=progress_var, maximum=100, length=350)
        progress_bar.pack(padx=20, pady=5)
        
        status_var = tk.StringVar(value="Menyiapkan...")
        status_label = ttk.Label(progress_dialog, textvariable=status_var)
        status_label.pack(pady=5)
        
        # Update progress function
        def update_progress(value, message):
            progress_var.set(value)
            status_var.set(message)
            progress_dialog.update_idletasks()
        
        # Run export in background thread
        def export_thread():
            try:
                from matplotlib.backends.backend_pdf import PdfPages
                import matplotlib.pyplot as plt
                from matplotlib.figure import Figure
                
                # A4 dimensions in mm and conversion to inches
                a4_width_mm = 210
                a4_height_mm = 297
                mm_to_inch = 0.0393701
                
                a4_width_inch = a4_width_mm * mm_to_inch
                a4_height_inch = a4_height_mm * mm_to_inch
                
                # 4cm margin in inches
                margin_cm = 4
                margin_inch = margin_cm / 2.54
                
                # Create PDF
                with PdfPages(file_path) as pdf:
                    # ===== HALAMAN 1: COVER =====
                    cover_fig = Figure(figsize=(a4_width_inch, a4_height_inch))
                    
                    # Set margins
                    cover_fig.subplots_adjust(
                        left=margin_inch/a4_width_inch,
                        right=1-(margin_inch/a4_width_inch),
                        top=1-(margin_inch/a4_height_inch),
                        bottom=margin_inch/a4_height_inch
                    )
                    
                    cover_ax = cover_fig.add_subplot(111)
                    cover_ax.axis('off')
                    
                    # Title - centered vertically and horizontally
                    cover_fig.text(0.5, 0.5, "Visualisasi Rute TPS Clustering", 
                                fontsize=20, fontweight='bold', ha='center', va='center')
                    
                    # Save cover page
                    pdf.savefig(cover_fig)
                    plt.close(cover_fig)
                    
                    update_progress(5, "Halaman cover ditambahkan...")
                    
                    # ===== HALAMAN 2: DAFTAR ISI =====
                    toc_fig = Figure(figsize=(a4_width_inch, a4_height_inch))
                    
                    # Set margins
                    toc_fig.subplots_adjust(
                        left=margin_inch/a4_width_inch,
                        right=1-(margin_inch/a4_width_inch),
                        top=1-(margin_inch/a4_height_inch),
                        bottom=margin_inch/a4_height_inch
                    )
                    
                    toc_ax = toc_fig.add_subplot(111)
                    toc_ax.axis('off')
                    
                    # Info cluster
                    if hasattr(self, 'results') and self.results:
                        clusters_info = f"Total Cluster: {self.results['num_clusters']}\n"
                        if 'metrics' in self.results:
                            metrics = self.results['metrics']
                            if 'total_route_distance' in metrics:
                                clusters_info += f"Total Jarak Rute: {metrics['total_route_distance']:.2f}\n"
                            if 'total_complete_route_distance' in metrics:
                                clusters_info += f"Total Jarak Rute (Garasi-TPA): {metrics['total_complete_route_distance']:.2f}\n"
                        
                        toc_fig.text(0.5, 0.8, clusters_info, ha='center', fontsize=12)
                    
                    # Add date
                    toc_fig.text(0.5, 0.7, f"Dibuat pada: {time.strftime('%d %B %Y, %H:%M')}", ha='center', fontsize=10)
                    
                    # Daftar isi title
                    toc_fig.text(0.5, 0.6, "Daftar Isi:", ha='center', fontsize=14, fontweight='bold')
                    
                    # Daftar isi content - with proper alignment from margin
                    toc_text = ""
                    for i, _ in enumerate(self.route_figures):
                        cluster_num = i+1
                        if hasattr(self, 'results') and 'clusters' in self.results and i < len(self.results['clusters']):
                            cluster_num = self.results['clusters'][i]['cluster_number']
                        toc_text += f"Cluster {cluster_num}: Halaman {i+3}\n"  # +3 karena cover + TOC + 1
                    
                    # Calculate left margin position for text alignment
                    left_margin_pos = margin_inch/a4_width_inch + 0.05
                    toc_fig.text(left_margin_pos, 0.55, toc_text, ha='left', va='top', fontsize=10)
                    
                    # Save TOC page
                    pdf.savefig(toc_fig)
                    plt.close(toc_fig)
                    
                    update_progress(10, "Halaman daftar isi ditambahkan...")
                    
                    # ===== HALAMAN RUTE: PENDEKATAN SEDERHANA =====
                    total_figures = len(self.route_figures)
                    
                    for i, orig_fig in enumerate(self.route_figures):
                        # Update progress
                        progress_pct = 10 + ((i + 1) / max(total_figures, 1)) * 90
                        update_progress(progress_pct, f"Memproses halaman {i+3}/{total_figures+2}...")
                        
                        # Get cluster info for page number
                        cluster_num = i+1
                        if hasattr(self, 'results') and 'clusters' in self.results and i < len(self.results['clusters']):
                            cluster_num = self.results['clusters'][i]['cluster_number']
                            
                        # Add page number to the original figure
                        orig_fig.text(0.5, 0.02, f"Cluster {cluster_num} - Hal {i+3}", ha='center', fontsize=9)
                        
                        # Simpan figure asli ke PDF dengan orientasi landscape
                        pdf.savefig(orig_fig, orientation='landscape')
                    
                    # Add metadata
                    d = pdf.infodict()
                    d['Title'] = 'Visualisasi Rute TPS Clustering'
                    d['Author'] = 'TPS Clustering App'
                    d['Subject'] = 'Route Visualization'
                    d['Keywords'] = 'clustering, route optimization'
                    d['CreationDate'] = time.strftime("%Y-%m-%d %H:%M:%S")
                
                # Complete
                update_progress(100, "Selesai!")
                
                # Close progress dialog after delay
                progress_dialog.after(500, progress_dialog.destroy)
                
                # Show success message
                messagebox.showinfo("Sukses", f"Semua gambar rute berhasil disimpan ke {file_path}")
                
                # Ask if user wants to open the file
                if messagebox.askyesno("Buka PDF", "Apakah Anda ingin membuka file PDF sekarang?"):
                    webbrowser.open('file://' + os.path.abspath(file_path))
                    
            except Exception as e:
                # Handle errors with detailed traceback
                import traceback
                error_details = traceback.format_exc()
                print(f"PDF Export Error: {str(e)}\n{error_details}")
                progress_dialog.destroy()
                messagebox.showerror("Error", f"Gagal menyimpan file PDF: {str(e)}")
        
        # Start the thread
        import threading
        export_thread = threading.Thread(target=export_thread)
        export_thread.daemon = True
        export_thread.start()


    def save_single_route_image(self, fig, base_filename):
        """Save a single route visualization as an image file"""
        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[
                ("PNG files", "*.png"), 
                ("JPEG files", "*.jpg"), 
                ("PDF files", "*.pdf"),
                ("SVG files", "*.svg")
            ],
            title="Simpan Gambar Rute",
            initialfile=base_filename
        )
        
        if file_path:
            try:
                fig.savefig(file_path, dpi=300, bbox_inches='tight')
                messagebox.showinfo("Sukses", f"Gambar berhasil disimpan ke {file_path}")
                
                # Ask if user wants to open the file
                if file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.pdf')):
                    if messagebox.askyesno("Buka Gambar", "Apakah Anda ingin membuka gambar sekarang?"):
                        webbrowser.open('file://' + os.path.abspath(file_path))
            except Exception as e:
                messagebox.showerror("Error", f"Gagal menyimpan gambar: {str(e)}")
    
    def save_all_routes(self):
        """Save all route visualizations as a single PDF file"""
        if not self.route_figures:
            messagebox.showinfo("Info", "Tidak ada gambar rute yang dapat disimpan")
            return
            
        file_path = filedialog.asksaveasfilename(
            defaultextension=".pdf",
            filetypes=[("PDF files", "*.pdf")],
            title="Simpan Semua Rute (PDF)"
        )
        
        if file_path:
            try:
                from matplotlib.backends.backend_pdf import PdfPages
                
                with PdfPages(file_path) as pdf:
                    for fig in self.route_figures:
                        pdf.savefig(fig, bbox_inches='tight')
                
                messagebox.showinfo("Sukses", f"Semua gambar rute berhasil disimpan ke {file_path}")
                
                # Ask if user wants to open the file
                if messagebox.askyesno("Buka PDF", "Apakah Anda ingin membuka file PDF sekarang?"):
                    webbrowser.open('file://' + os.path.abspath(file_path))
            except Exception as e:
                messagebox.showerror("Error", f"Gagal menyimpan file PDF: {str(e)}")
    
    def save_all_routes_as_images(self):
        """Save all route visualizations as separate image files"""
        if not self.route_figures:
            messagebox.showinfo("Info", "Tidak ada gambar rute yang dapat disimpan")
            return
            
        # Ask for directory to save images
        directory = filedialog.askdirectory(
            title="Pilih Folder untuk Menyimpan Gambar"
        )
        
        if directory:
            try:
                # Ask for image format
                formats = {
                    "PNG (.png)": ".png",
                    "JPEG (.jpg)": ".jpg",
                    "PDF (.pdf)": ".pdf",
                    "SVG (.svg)": ".svg"
                }
                
                format_dialog = tk.Toplevel(self.root)
                format_dialog.title("Pilih Format Gambar")
                format_dialog.geometry("300x200")
                format_dialog.resizable(False, False)
                format_dialog.transient(self.root)
                format_dialog.grab_set()
                
                ttk.Label(format_dialog, text="Pilih format gambar:", padding=10).pack()
                
                selected_format = tk.StringVar()
                for format_name, format_ext in formats.items():
                    ttk.Radiobutton(
                        format_dialog, 
                        text=format_name, 
                        value=format_ext, 
                        variable=selected_format
                    ).pack(anchor="w", padx=20, pady=5)
                
                # Default to PNG
                selected_format.set(".png")
                
                def on_ok():
                    format_ext = selected_format.get()
                    format_dialog.destroy()
                    
                    # Save each figure
                    for i, fig in enumerate(self.route_figures):
                        if i < len(self.results['clusters']):
                            cluster_num = self.results['clusters'][i]['cluster_number']
                            filename = f"rute_cluster_{cluster_num}{format_ext}"
                        else:
                            filename = f"rute_{i+1}{format_ext}"
                        
                        file_path = os.path.join(directory, filename)
                        fig.savefig(file_path, dpi=300, bbox_inches='tight')
                    
                    messagebox.showinfo("Sukses", f"Semua gambar rute berhasil disimpan ke folder {directory}")
                    
                    # Ask if user wants to open the directory
                    if messagebox.askyesno("Buka Folder", "Apakah Anda ingin membuka folder sekarang?"):
                        webbrowser.open('file://' + os.path.abspath(directory))
                
                def on_cancel():
                    format_dialog.destroy()
                
                button_frame = ttk.Frame(format_dialog)
                button_frame.pack(fill="x", pady=10)
                
                ttk.Button(button_frame, text="OK", command=on_ok).pack(side="right", padx=10)
                ttk.Button(button_frame, text="Batal", command=on_cancel).pack(side="right", padx=10)
                
                # Center the dialog
                format_dialog.update_idletasks()
                width = format_dialog.winfo_width()
                height = format_dialog.winfo_height()
                x = (self.root.winfo_width() // 2) - (width // 2)
                y = (self.root.winfo_height() // 2) - (height // 2)
                format_dialog.geometry(f"+{x}+{y}")
                
                format_dialog.wait_window()
                
            except Exception as e:
                messagebox.showerror("Error", f"Gagal menyimpan gambar: {str(e)}")
    
    def save_visualization(self, figures, prefix=""):
        """Save the visualization to a file"""
        if not figures:
            return
            
        file_path = filedialog.asksaveasfilename(
            defaultextension=".pdf",
            filetypes=[
                ("PDF File", "*.pdf"), 
                ("PNG Image", "*.png"), 
                ("JPEG Image", "*.jpg"),
                ("SVG Image", "*.svg")
            ],
            title="Simpan Visualisasi"
        )
        
        if file_path:
            try:
                # If multiple figures, save as a single PDF
                if len(figures) > 1 and file_path.lower().endswith('.pdf'):
                    from matplotlib.backends.backend_pdf import PdfPages
                    
                    with PdfPages(file_path) as pdf:
                        for fig in figures:
                            pdf.savefig(fig, bbox_inches='tight')
                    
                    messagebox.showinfo("Sukses", f"Visualisasi berhasil disimpan ke {file_path}")
                else:
                    # For single figure or non-PDF format, save the first figure only
                    figures[0].savefig(file_path, dpi=300, bbox_inches='tight')
                    messagebox.showinfo("Sukses", f"Visualisasi berhasil disimpan ke {file_path}")
                    
                    # If user wants to save multiple figures as individual files
                    if len(figures) > 1 and messagebox.askyesno(
                        "Simpan Semua?", 
                        f"Anda memiliki {len(figures)} visualisasi. Apakah Anda ingin menyimpan semua sebagai file terpisah?"
                    ):
                        # Ask for directory
                        directory = filedialog.askdirectory(
                            title="Pilih Folder untuk Menyimpan Semua Gambar"
                        )
                        
                        if directory:
                            # Extract extension
                            ext = file_path.rsplit('.', 1)[1].lower()
                            
                            # Save each figure
                            for i, fig in enumerate(figures):
                                fig_path = os.path.join(directory, f"{prefix}{i+1}.{ext}")
                                fig.savefig(fig_path, dpi=300, bbox_inches='tight')
                            
                            messagebox.showinfo("Sukses", f"Semua visualisasi berhasil disimpan ke folder {directory}")
                            
                            # Ask if user wants to open the directory
                            if messagebox.askyesno("Buka Folder", "Apakah Anda ingin membuka folder sekarang?"):
                                webbrowser.open('file://' + os.path.abspath(directory))
            except Exception as e:
                messagebox.showerror("Error", f"Gagal menyimpan visualisasi: {str(e)}")

# Main application entry point
if __name__ == "__main__":
    root = tk.Tk()
    app = TPSClusteringApp(root)
    root.mainloop()
