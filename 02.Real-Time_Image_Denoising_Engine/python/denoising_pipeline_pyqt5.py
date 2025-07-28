#!/usr/bin/env python3
"""
Real-Time Image Denoising Pipeline - Python Interface
====================================================

This module provides a high-level Python interface to the CUDA-accelerated
image denoising engine.
"""

import cv2
import numpy as np
import ctypes
import os
import sys
import threading
import time
from typing import Optional, Tuple, Dict, List, Callable
from dataclasses import dataclass
from enum import Enum
import queue
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DenoiseAlgorithm(Enum):
    """Enumeration of available denoising algorithms."""
    BILATERAL = 0
    NON_LOCAL_MEANS = 1
    GAUSSIAN = 2
    ADAPTIVE_BILATERAL = 3

@dataclass
class DenoiseParams:
    """Parameters for denoising algorithms."""
    sigma_color: float = 50.0
    sigma_space: float = 50.0
    h: float = 10.0  # NLM filtering strength
    template_window_size: int = 7
    search_window_size: int = 21
    gaussian_sigma: float = 1.0
    kernel_size: int = 5

@dataclass
class ProcessingStats:
    """Statistics for processing performance."""
    avg_fps: float = 0.0
    min_fps: float = 0.0
    max_fps: float = 0.0
    avg_latency_ms: float = 0.0
    frames_processed: int = 0
    frames_dropped: int = 0
    memory_usage_mb: int = 0

class DenoisingPipeline:
    """
    High-level interface for real-time image denoising.
    
    This class provides a Python wrapper around the CUDA denoising engine,
    enabling easy integration with Python applications and OpenCV.
    """
    
    def __init__(self, max_width: int = 1920, max_height: int = 1080):
        """
        Initialize the denoising pipeline.
        
        Args:
            max_width: Maximum image width to support
            max_height: Maximum image height to support
        """
        self.max_width = max_width
        self.max_height = max_height
        self.is_initialized = False
        self.processing_thread = None
        self.stop_processing = threading.Event()
        self.frame_queue = queue.Queue(maxsize=10)
        self.result_queue = queue.Queue(maxsize=10)
        self.stats = ProcessingStats()
        self.frame_callback: Optional[Callable] = None
        
        # Load the CUDA library (this would be the compiled C++ library)
        self._load_cuda_library()
        
        # Default parameters
        self.algorithm = DenoiseAlgorithm.BILATERAL
        self.params = DenoiseParams()
        
        logger.info(f"DenoisingPipeline initialized for {max_width}x{max_height}")
    
    def _load_cuda_library(self):
        """Load the compiled CUDA library."""
        # In a real implementation, this would load the compiled .so/.dll file
        # For this example, we'll simulate the interface
        logger.info("CUDA library loaded (simulated)")
        self.is_initialized = True
    
    def set_algorithm(self, algorithm: DenoiseAlgorithm):
        """Set the denoising algorithm to use."""
        self.algorithm = algorithm
        logger.info(f"Algorithm set to: {algorithm.name}")
    
    def set_parameters(self, params: DenoiseParams):
        """Set the denoising parameters."""
        self.params = params
        logger.info("Denoising parameters updated")
    
    def denoise_image(self, image: np.ndarray) -> np.ndarray:
        """
        Denoise a single image.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Denoised image as numpy array
        """
        if not self.is_initialized:
            raise RuntimeError("Pipeline not initialized")
        
        if image is None or image.size == 0:
            raise ValueError("Invalid input image")
        
        # Convert to float32 if necessary
        if image.dtype != np.float32:
            image = image.astype(np.float32) / 255.0
        
        # For demo purposes, we'll use OpenCV's built-in denoising
        # In the real implementation, this would call the CUDA kernels
        start_time = time.time()
        
        if self.algorithm == DenoiseAlgorithm.BILATERAL:
            # Convert to 8-bit for OpenCV bilateral filter
            img_8bit = (image * 255).astype(np.uint8)
            denoised = cv2.bilateralFilter(
                img_8bit, 
                self.params.kernel_size,
                self.params.sigma_color,
                self.params.sigma_space
            )
            result = denoised.astype(np.float32) / 255.0
            
        elif self.algorithm == DenoiseAlgorithm.NON_LOCAL_MEANS:
            img_8bit = (image * 255).astype(np.uint8)
            if len(img_8bit.shape) == 3:
                denoised = cv2.fastNlMeansDenoisingColored(
                    img_8bit,
                    None,
                    self.params.h,
                    self.params.h,
                    self.params.template_window_size,
                    self.params.search_window_size
                )
            else:
                denoised = cv2.fastNlMeansDenoising(
                    img_8bit,
                    None,
                    self.params.h,
                    self.params.template_window_size,
                    self.params.search_window_size
                )
            result = denoised.astype(np.float32) / 255.0
            
        elif self.algorithm == DenoiseAlgorithm.GAUSSIAN:
            kernel_size = self.params.kernel_size
            if kernel_size % 2 == 0:
                kernel_size += 1
            result = cv2.GaussianBlur(
                image,
                (kernel_size, kernel_size),
                self.params.gaussian_sigma
            )
            
        else:  # ADAPTIVE_BILATERAL
            # Simulate adaptive bilateral filtering
            img_8bit = (image * 255).astype(np.uint8)
            denoised = cv2.bilateralFilter(
                img_8bit,
                self.params.kernel_size,
                self.params.sigma_color * 1.5,  # Adaptive factor
                self.params.sigma_space
            )
            result = denoised.astype(np.float32) / 255.0
        
        processing_time = (time.time() - start_time) * 1000  # Convert to ms
        
        # Update statistics
        self.stats.frames_processed += 1
        if self.stats.frames_processed == 1:
            self.stats.avg_latency_ms = processing_time
        else:
            # Running average
            alpha = 0.1
            self.stats.avg_latency_ms = (1 - alpha) * self.stats.avg_latency_ms + alpha * processing_time
        
        self.stats.avg_fps = 1000.0 / self.stats.avg_latency_ms if self.stats.avg_latency_ms > 0 else 0
        
        return result
    
    def start_camera_processing(self, camera_id: int = 0) -> bool:
        """
        Start real-time camera processing.
        
        Args:
            camera_id: Camera device ID
            
        Returns:
            True if successful, False otherwise
        """
        if not self.is_initialized:
            logger.error("Pipeline not initialized")
            return False
        
        # Open camera
        self.camera = cv2.VideoCapture(camera_id)
        if not self.camera.isOpened():
            logger.error(f"Failed to open camera {camera_id}")
            return False
        
        # Set camera properties
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.max_width)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.max_height)
        self.camera.set(cv2.CAP_PROP_FPS, 30)
        
        # Start processing thread
        self.stop_processing.clear()
        self.processing_thread = threading.Thread(target=self._processing_worker)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        logger.info("Camera processing started")
        return True
    
    def stop_camera_processing(self):
        """Stop real-time camera processing."""
        if self.processing_thread:
            self.stop_processing.set()
            self.processing_thread.join(timeout=2.0)
            self.processing_thread = None
        
        if hasattr(self, 'camera'):
            self.camera.release()
            delattr(self, 'camera')
        
        logger.info("Camera processing stopped")
    
    def _processing_worker(self):
        """Worker thread for real-time processing."""
        logger.info("Processing worker started")
        
        while not self.stop_processing.is_set():
            try:
                # Capture frame
                ret, frame = self.camera.read()
                if not ret:
                    self.stats.frames_dropped += 1
                    continue
                
                # Add to processing queue
                try:
                    self.frame_queue.put_nowait(frame)
                except queue.Full:
                    # Drop oldest frame
                    try:
                        self.frame_queue.get_nowait()
                        self.frame_queue.put_nowait(frame)
                        self.stats.frames_dropped += 1
                    except queue.Empty:
                        pass
                
                # Process frames from queue
                try:
                    input_frame = self.frame_queue.get_nowait()
                    
                    # Denoise the frame
                    denoised_frame = self.denoise_image(input_frame)
                    
                    # Convert back to 8-bit for display
                    display_frame = (denoised_frame * 255).astype(np.uint8)
                    
                    # Add to result queue
                    try:
                        self.result_queue.put_nowait(display_frame)
                    except queue.Full:
                        # Drop oldest result
                        try:
                            self.result_queue.get_nowait()
                            self.result_queue.put_nowait(display_frame)
                        except queue.Empty:
                            pass
                    
                    # Call frame callback if set
                    if self.frame_callback:
                        self.frame_callback(display_frame, self.stats.avg_latency_ms)
                
                except queue.Empty:
                    time.sleep(0.001)  # Small delay to prevent busy waiting
                
            except Exception as e:
                logger.error(f"Error in processing worker: {e}")
                break
        
        logger.info("Processing worker stopped")
    
    def get_processed_frame(self, timeout_ms: int = 100) -> Optional[np.ndarray]:
        """
        Get the latest processed frame.
        
        Args:
            timeout_ms: Timeout in milliseconds
            
        Returns:
            Processed frame or None if timeout
        """
        try:
            timeout_sec = timeout_ms / 1000.0
            return self.result_queue.get(timeout=timeout_sec)
        except queue.Empty:
            return None
    
    def set_frame_callback(self, callback: Callable[[np.ndarray, float], None]):
        """
        Set a callback function to be called for each processed frame.
        
        Args:
            callback: Function that takes (frame, processing_time_ms) as arguments
        """
        self.frame_callback = callback
    
    def get_statistics(self) -> ProcessingStats:
        """Get current processing statistics."""
        return self.stats
    
    def reset_statistics(self):
        """Reset processing statistics."""
        self.stats = ProcessingStats()
    
    def benchmark_algorithms(self, test_image: np.ndarray, iterations: int = 100) -> Dict[str, Dict]:
        """
        Benchmark all available algorithms on a test image.
        
        Args:
            test_image: Test image for benchmarking
            iterations: Number of iterations for each algorithm
            
        Returns:
            Dictionary with benchmark results for each algorithm
        """
        results = {}
        
        original_algo = self.algorithm
        original_params = self.params
        
        algorithms = [
            (DenoiseAlgorithm.BILATERAL, "Bilateral"),
            (DenoiseAlgorithm.NON_LOCAL_MEANS, "Non-Local Means"),
            (DenoiseAlgorithm.GAUSSIAN, "Gaussian"),
            (DenoiseAlgorithm.ADAPTIVE_BILATERAL, "Adaptive Bilateral")
        ]
        
        for algo, name in algorithms:
            logger.info(f"Benchmarking {name}...")
            
            self.set_algorithm(algo)
            times = []
            
            # Warm up
            for _ in range(5):
                self.denoise_image(test_image)
            
            # Benchmark
            for i in range(iterations):
                start_time = time.time()
                result = self.denoise_image(test_image)
                end_time = time.time()
                
                processing_time = (end_time - start_time) * 1000  # ms
                times.append(processing_time)
                
                if i % 20 == 0:
                    logger.info(f"  Progress: {i}/{iterations}")
            
            # Calculate statistics
            avg_time = np.mean(times)
            min_time = np.min(times)
            max_time = np.max(times)
            std_time = np.std(times)
            
            results[name] = {
                'algorithm': algo,
                'avg_time_ms': avg_time,
                'min_time_ms': min_time,
                'max_time_ms': max_time,
                'std_time_ms': std_time,
                'avg_fps': 1000.0 / avg_time if avg_time > 0 else 0,
                'all_times': times
            }
        
        # Restore original settings
        self.set_algorithm(original_algo)
        self.set_parameters(original_params)
        
        return results
    
    def create_gui_application(self):
        """Create a modern GUI application for interactive denoising."""
        try:
            from PyQt5.QtWidgets import QApplication
            import sys
        except ImportError:
            logger.error("PyQt5 not available. Install with: pip install PyQt5")
            return None
        
        # Create QApplication if it doesn't exist
        app = QApplication.instance()
        if app is None:
            app = QApplication(sys.argv)
        
        return DenoisingGUI(self, app)

class DenoisingGUI:
    """Modern PyQt5 GUI application for interactive denoising."""
    
    def __init__(self, pipeline: DenoisingPipeline, app=None):
        try:
            from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                                       QGridLayout, QLabel, QPushButton, QComboBox, 
                                       QSlider, QGroupBox, QFileDialog, QMessageBox,
                                       QStatusBar, QSplitter, QProgressBar, QSpinBox,
                                       QDoubleSpinBox, QCheckBox, QTabWidget, QTextEdit,
                                       QScrollArea, QFrame, QSizePolicy)
            from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal, QSize
            from PyQt5.QtGui import QPixmap, QFont, QIcon, QPalette, QColor
        except ImportError:
            logger.error("PyQt5 dependencies not available.")
            raise
        
        self.pipeline = pipeline
        self.app = app
        self.current_image = None
        self.processed_image = None
        self.camera_timer = QTimer()
        
        # Create main window
        self.main_window = QMainWindow()
        self.main_window.setWindowTitle("Real-Time Image Denoising Engine")
        self.main_window.setGeometry(100, 100, 1200, 800)
        self.main_window.setMinimumSize(800, 600)
        
        # Set modern style
        self._setup_style()
        
        # Create widgets
        self._create_widgets()
        self._setup_layouts()
        self._setup_connections()
        self._setup_callbacks()
        
        # Setup update timer for live processing
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_live_display)
        
    def _setup_style(self):
        """Setup modern dark theme style."""
        try:
            from PyQt5.QtWidgets import QApplication
            from PyQt5.QtCore import Qt
            from PyQt5.QtGui import QPalette, QColor
            
            # Set dark theme
            palette = QPalette()
            palette.setColor(QPalette.Window, QColor(53, 53, 53))
            palette.setColor(QPalette.WindowText, QColor(255, 255, 255))
            palette.setColor(QPalette.Base, QColor(25, 25, 25))
            palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
            palette.setColor(QPalette.ToolTipBase, QColor(0, 0, 0))
            palette.setColor(QPalette.ToolTipText, QColor(255, 255, 255))
            palette.setColor(QPalette.Text, QColor(255, 255, 255))
            palette.setColor(QPalette.Button, QColor(53, 53, 53))
            palette.setColor(QPalette.ButtonText, QColor(255, 255, 255))
            palette.setColor(QPalette.BrightText, QColor(255, 0, 0))
            palette.setColor(QPalette.Link, QColor(42, 130, 218))
            palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
            palette.setColor(QPalette.HighlightedText, QColor(0, 0, 0))
            
            self.app.setPalette(palette)
            
            # Set custom stylesheet
            style = """
            QMainWindow {
                background-color: #2b2b2b;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #555;
                border-radius: 8px;
                margin-top: 1ex;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
            QPushButton {
                background-color: #404040;
                border: 1px solid #666;
                border-radius: 6px;
                padding: 6px 12px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #4a4a4a;
                border-color: #888;
            }
            QPushButton:pressed {
                background-color: #2a2a2a;
            }
            QSlider::groove:horizontal {
                border: 1px solid #999;
                height: 8px;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #B1B1B1, stop:1 #c4c4c4);
                margin: 2px 0;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #b4b4b4, stop:1 #8f8f8f);
                border: 1px solid #5c5c5c;
                width: 18px;
                margin: -2px 0;
                border-radius: 9px;
            }
            QComboBox {
                border: 1px solid #666;
                border-radius: 4px;
                padding: 4px 8px;
                background-color: #404040;
            }
            QComboBox::drop-down {
                border: none;
            }
            QComboBox::down-arrow {
                image: url(down_arrow.png);
            }
            QLabel {
                background-color: transparent;
            }
            QStatusBar {
                background-color: #333;
                border-top: 1px solid #666;
            }
            """
            self.main_window.setStyleSheet(style)
            
        except Exception as e:
            logger.warning(f"Could not apply custom style: {e}")
    
    def _create_widgets(self):
        """Create all GUI widgets."""
        from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, 
                                   QLabel, QPushButton, QComboBox, QSlider, QGroupBox, 
                                   QStatusBar, QSplitter, QProgressBar, QSpinBox,
                                   QDoubleSpinBox, QCheckBox, QTabWidget, QTextEdit,
                                   QScrollArea, QFrame)
        from PyQt5.QtCore import Qt
        from PyQt5.QtGui import QFont
        
        # Central widget
        self.central_widget = QWidget()
        self.main_window.setCentralWidget(self.central_widget)
        
        # Control Panel
        self.control_group = QGroupBox("Control Panel")
        
        # File operations
        self.load_btn = QPushButton("📁 Load Image")
        self.save_btn = QPushButton("💾 Save Result")
        self.camera_btn = QPushButton("📷 Start Camera")
        self.stop_camera_btn = QPushButton("⏹ Stop Camera")
        self.benchmark_btn = QPushButton("⚡ Benchmark")
        
        # Algorithm selection
        self.algo_label = QLabel("Algorithm:")
        self.algo_combo = QComboBox()
        self.algo_combo.addItems(["BILATERAL", "NON_LOCAL_MEANS", "GAUSSIAN", "ADAPTIVE_BILATERAL"])
        
        # Parameters group
        self.params_group = QGroupBox("Algorithm Parameters")
        
        # Bilateral filter parameters
        self.sigma_color_label = QLabel("Sigma Color:")
        self.sigma_color_slider = QSlider(Qt.Horizontal)
        self.sigma_color_slider.setRange(1, 100)
        self.sigma_color_slider.setValue(50)
        self.sigma_color_value = QLabel("50")
        
        self.sigma_space_label = QLabel("Sigma Space:")
        self.sigma_space_slider = QSlider(Qt.Horizontal)
        self.sigma_space_slider.setRange(1, 100)
        self.sigma_space_slider.setValue(50)
        self.sigma_space_value = QLabel("50")
        
        # NLM parameters
        self.h_label = QLabel("H Parameter:")
        self.h_slider = QSlider(Qt.Horizontal)
        self.h_slider.setRange(1, 50)
        self.h_slider.setValue(10)
        self.h_value = QLabel("10")
        
        self.template_label = QLabel("Template Size:")
        self.template_spin = QSpinBox()
        self.template_spin.setRange(3, 15)
        self.template_spin.setValue(7)
        self.template_spin.setSingleStep(2)
        
        self.search_label = QLabel("Search Size:")
        self.search_spin = QSpinBox()
        self.search_spin.setRange(7, 35)
        self.search_spin.setValue(21)
        self.search_spin.setSingleStep(2)
        
        # Gaussian parameters
        self.gaussian_sigma_label = QLabel("Gaussian Sigma:")
        self.gaussian_sigma_slider = QSlider(Qt.Horizontal)
        self.gaussian_sigma_slider.setRange(1, 50)
        self.gaussian_sigma_slider.setValue(10)
        self.gaussian_sigma_value = QLabel("1.0")
        
        self.kernel_size_label = QLabel("Kernel Size:")
        self.kernel_size_spin = QSpinBox()
        self.kernel_size_spin.setRange(3, 15)
        self.kernel_size_spin.setValue(5)
        self.kernel_size_spin.setSingleStep(2)
        
        # Real-time processing options
        self.realtime_group = QGroupBox("Real-time Options")
        self.auto_process_check = QCheckBox("Auto Process")
        self.auto_process_check.setChecked(True)
        self.show_fps_check = QCheckBox("Show FPS")
        self.show_fps_check.setChecked(True)
        
        # Image display area
        self.image_splitter = QSplitter(Qt.Horizontal)
        
        # Original image
        self.orig_group = QGroupBox("Original Image")
        self.orig_scroll = QScrollArea()
        self.orig_label = QLabel()
        self.orig_label.setAlignment(Qt.AlignCenter)
        self.orig_label.setText("No image loaded")
        self.orig_label.setMinimumSize(400, 300)
        self.orig_label.setStyleSheet("border: 1px solid #666; background-color: #1a1a1a;")
        self.orig_scroll.setWidget(self.orig_label)
        self.orig_scroll.setWidgetResizable(True)
        
        # Processed image
        self.proc_group = QGroupBox("Denoised Image")
        self.proc_scroll = QScrollArea()
        self.proc_label = QLabel()
        self.proc_label.setAlignment(Qt.AlignCenter)
        self.proc_label.setText("No processed image")
        self.proc_label.setMinimumSize(400, 300)
        self.proc_label.setStyleSheet("border: 1px solid #666; background-color: #1a1a1a;")
        self.proc_scroll.setWidget(self.proc_label)
        self.proc_scroll.setWidgetResizable(True)
        
        # Performance display
        self.perf_group = QGroupBox("Performance")
        self.fps_label = QLabel("FPS: --")
        self.latency_label = QLabel("Latency: --")
        self.memory_label = QLabel("Memory: --")
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        
        # Log display
        self.log_group = QGroupBox("Log")
        self.log_text = QTextEdit()
        self.log_text.setMaximumHeight(100)
        self.log_text.setReadOnly(True)
        
        # Status bar
        self.status_bar = QStatusBar()
        self.main_window.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")
    
    def _setup_layouts(self):
        """Setup widget layouts."""
        from PyQt5.QtWidgets import QVBoxLayout, QHBoxLayout, QGridLayout
        from PyQt5.QtCore import Qt
        
        # Main layout
        main_layout = QHBoxLayout(self.central_widget)
        
        # Left panel (controls)
        left_panel = QVBoxLayout()
        left_panel.setSpacing(10)
        
        # Control group layout
        control_layout = QVBoxLayout(self.control_group)
        
        # File operations layout
        file_layout = QHBoxLayout()
        file_layout.addWidget(self.load_btn)
        file_layout.addWidget(self.save_btn)
        file_layout.addWidget(self.camera_btn)
        file_layout.addWidget(self.stop_camera_btn)
        file_layout.addStretch()
        control_layout.addLayout(file_layout)
        
        # Algorithm selection layout
        algo_layout = QHBoxLayout()
        algo_layout.addWidget(self.algo_label)
        algo_layout.addWidget(self.algo_combo)
        algo_layout.addWidget(self.benchmark_btn)
        algo_layout.addStretch()
        control_layout.addLayout(algo_layout)
        
        # Parameters group layout
        params_layout = QGridLayout(self.params_group)
        
        # Bilateral parameters
        params_layout.addWidget(self.sigma_color_label, 0, 0)
        params_layout.addWidget(self.sigma_color_slider, 0, 1)
        params_layout.addWidget(self.sigma_color_value, 0, 2)
        
        params_layout.addWidget(self.sigma_space_label, 1, 0)
        params_layout.addWidget(self.sigma_space_slider, 1, 1)
        params_layout.addWidget(self.sigma_space_value, 1, 2)
        
        # NLM parameters
        params_layout.addWidget(self.h_label, 2, 0)
        params_layout.addWidget(self.h_slider, 2, 1)
        params_layout.addWidget(self.h_value, 2, 2)
        
        params_layout.addWidget(self.template_label, 3, 0)
        params_layout.addWidget(self.template_spin, 3, 1)
        
        params_layout.addWidget(self.search_label, 4, 0)
        params_layout.addWidget(self.search_spin, 4, 1)
        
        # Gaussian parameters
        params_layout.addWidget(self.gaussian_sigma_label, 5, 0)
        params_layout.addWidget(self.gaussian_sigma_slider, 5, 1)
        params_layout.addWidget(self.gaussian_sigma_value, 5, 2)
        
        params_layout.addWidget(self.kernel_size_label, 6, 0)
        params_layout.addWidget(self.kernel_size_spin, 6, 1)
        
        # Real-time options layout
        realtime_layout = QVBoxLayout(self.realtime_group)
        realtime_layout.addWidget(self.auto_process_check)
        realtime_layout.addWidget(self.show_fps_check)
        
        # Performance layout
        perf_layout = QVBoxLayout(self.perf_group)
        perf_layout.addWidget(self.fps_label)
        perf_layout.addWidget(self.latency_label)
        perf_layout.addWidget(self.memory_label)
        perf_layout.addWidget(self.progress_bar)
        
        # Log layout
        log_layout = QVBoxLayout(self.log_group)
        log_layout.addWidget(self.log_text)
        
        # Assemble left panel
        left_panel.addWidget(self.control_group)
        left_panel.addWidget(self.params_group)
        left_panel.addWidget(self.realtime_group)
        left_panel.addWidget(self.perf_group)
        left_panel.addWidget(self.log_group)
        left_panel.addStretch()
        
        # Right panel (images)
        right_panel = QVBoxLayout()
        
        # Setup image groups
        orig_layout = QVBoxLayout(self.orig_group)
        orig_layout.addWidget(self.orig_scroll)
        
        proc_layout = QVBoxLayout(self.proc_group)
        proc_layout.addWidget(self.proc_scroll)
        
        # Add to splitter
        self.image_splitter.addWidget(self.orig_group)
        self.image_splitter.addWidget(self.proc_group)
        self.image_splitter.setSizes([400, 400])
        
        right_panel.addWidget(self.image_splitter)
        
        # Create left widget
        left_widget = QWidget()
        left_widget.setLayout(left_panel)
        left_widget.setMaximumWidth(350)
        left_widget.setMinimumWidth(300)
        
        # Create right widget
        right_widget = QWidget()
        right_widget.setLayout(right_panel)
        
        # Add to main layout
        main_layout.addWidget(left_widget)
        main_layout.addWidget(right_widget, 1)
    
    def _setup_connections(self):
        """Setup signal-slot connections."""
        # File operations
        self.load_btn.clicked.connect(self.load_image)
        self.save_btn.clicked.connect(self.save_image)
        self.camera_btn.clicked.connect(self.start_camera)
        self.stop_camera_btn.clicked.connect(self.stop_camera)
        self.benchmark_btn.clicked.connect(self.run_benchmark)
        
        # Algorithm selection
        self.algo_combo.currentTextChanged.connect(self.on_algorithm_change)
        
        # Parameter changes
        self.sigma_color_slider.valueChanged.connect(self.on_sigma_color_change)
        self.sigma_space_slider.valueChanged.connect(self.on_sigma_space_change)
        self.h_slider.valueChanged.connect(self.on_h_change)
        self.template_spin.valueChanged.connect(self.on_param_change)
        self.search_spin.valueChanged.connect(self.on_param_change)
        self.gaussian_sigma_slider.valueChanged.connect(self.on_gaussian_sigma_change)
        self.kernel_size_spin.valueChanged.connect(self.on_param_change)
        
        # Real-time options
        self.auto_process_check.toggled.connect(self.on_auto_process_toggle)
    
    def _setup_callbacks(self):
        """Setup pipeline callbacks."""
        self.pipeline.set_frame_callback(self.on_frame_processed)
    
    def load_image(self):
        """Load an image file."""
        from PyQt5.QtWidgets import QFileDialog, QMessageBox
        
        filename, _ = QFileDialog.getOpenFileName(
            self.main_window,
            "Select Image",
            "",
            "Image files (*.jpg *.jpeg *.png *.bmp *.tiff);;All files (*)"
        )
        
        if filename:
            self.current_image = cv2.imread(filename)
            if self.current_image is not None:
                self.display_original_image()
                if self.auto_process_check.isChecked():
                    self.process_current_image()
                self.status_bar.showMessage(f"Loaded: {os.path.basename(filename)}")
                self.log_message(f"Image loaded: {filename}")
            else:
                QMessageBox.critical(self.main_window, "Error", "Failed to load image")
    
    def save_image(self):
        """Save the processed image."""
        from PyQt5.QtWidgets import QFileDialog, QMessageBox
        
        if self.processed_image is not None:
            filename, _ = QFileDialog.getSaveFileName(
                self.main_window,
                "Save Processed Image",
                "",
                "JPEG files (*.jpg);;PNG files (*.png);;All files (*)"
            )
            
            if filename:
                cv2.imwrite(filename, self.processed_image)
                self.status_bar.showMessage(f"Saved: {os.path.basename(filename)}")
                self.log_message(f"Image saved: {filename}")
        else:
            QMessageBox.warning(self.main_window, "Warning", "No processed image to save")
    
    def start_camera(self):
        """Start camera processing."""
        from PyQt5.QtWidgets import QMessageBox
        
        if self.pipeline.start_camera_processing():
            self.status_bar.showMessage("Camera processing started")
            self.camera_btn.setEnabled(False)
            self.stop_camera_btn.setEnabled(True)
            self.update_timer.start(33)  # ~30 FPS
            self.log_message("Camera processing started")
        else:
            QMessageBox.critical(self.main_window, "Error", "Failed to start camera")
    
    def stop_camera(self):
        """Stop camera processing."""
        self.pipeline.stop_camera_processing()
        self.update_timer.stop()
        self.camera_btn.setEnabled(True)
        self.stop_camera_btn.setEnabled(False)
        self.status_bar.showMessage("Camera processing stopped")
        self.log_message("Camera processing stopped")
    
    def run_benchmark(self):
        """Run algorithm benchmark."""
        from PyQt5.QtWidgets import QMessageBox, QProgressDialog
        from PyQt5.QtCore import Qt
        
        if self.current_image is None:
            # Create test image
            test_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
            self.log_message("Created test image for benchmark")
        else:
            test_image = self.current_image
        
        # Show progress dialog
        progress = QProgressDialog("Running benchmark...", "Cancel", 0, 100, self.main_window)
        progress.setWindowModality(Qt.WindowModal)
        progress.show()
        
        try:
            # Run benchmark
            self.log_message("Starting benchmark...")
            results = self.pipeline.benchmark_algorithms(test_image, iterations=50)
            
            # Display results
            result_text = "=== Benchmark Results ===\n\n"
            for name, result in results.items():
                result_text += f"{name}:\n"
                result_text += f"  Average Time: {result['avg_time_ms']:.2f}ms\n"
                result_text += f"  Min Time: {result['min_time_ms']:.2f}ms\n"
                result_text += f"  Max Time: {result['max_time_ms']:.2f}ms\n"
                result_text += f"  Average FPS: {result['avg_fps']:.1f}\n\n"
            
            QMessageBox.information(self.main_window, "Benchmark Results", result_text)
            self.log_message("Benchmark completed")
            
        except Exception as e:
            QMessageBox.critical(self.main_window, "Error", f"Benchmark failed: {e}")
        finally:
            progress.close()
    
    def on_algorithm_change(self, algo_name):
        """Handle algorithm selection change."""
        algorithm = DenoiseAlgorithm[algo_name]
        self.pipeline.set_algorithm(algorithm)
        self.log_message(f"Algorithm changed to: {algo_name}")
        
        if self.current_image is not None and self.auto_process_check.isChecked():
            self.process_current_image()
    
    def on_sigma_color_change(self, value):
        """Handle sigma color parameter change."""
        self.sigma_color_value.setText(str(value))
        self.on_param_change()
    
    def on_sigma_space_change(self, value):
        """Handle sigma space parameter change."""
        self.sigma_space_value.setText(str(value))
        self.on_param_change()
    
    def on_h_change(self, value):
        """Handle H parameter change."""
        self.h_value.setText(str(value))
        self.on_param_change()
    
    def on_gaussian_sigma_change(self, value):
        """Handle Gaussian sigma parameter change."""
        sigma_val = value / 10.0  # Scale to 0.1-5.0 range
        self.gaussian_sigma_value.setText(f"{sigma_val:.1f}")
        self.on_param_change()
    
    def on_param_change(self):
        """Handle parameter changes."""
        params = DenoiseParams(
            sigma_color=self.sigma_color_slider.value(),
            sigma_space=self.sigma_space_slider.value(),
            h=self.h_slider.value(),
            template_window_size=self.template_spin.value(),
            search_window_size=self.search_spin.value(),
            gaussian_sigma=self.gaussian_sigma_slider.value() / 10.0,
            kernel_size=self.kernel_size_spin.value()
        )
        self.pipeline.set_parameters(params)
        
        if self.current_image is not None and self.auto_process_check.isChecked():
            self.process_current_image()
    
    def on_auto_process_toggle(self, checked):
        """Handle auto process toggle."""
        if checked and self.current_image is not None:
            self.process_current_image()
    
    def process_current_image(self):
        """Process the current image."""
        if self.current_image is not None:
            try:
                self.progress_bar.setVisible(True)
                self.progress_bar.setRange(0, 0)  # Indeterminate progress
                
                processed = self.pipeline.denoise_image(self.current_image)
                self.processed_image = (processed * 255).astype(np.uint8)
                self.display_processed_image()
                
                stats = self.pipeline.get_statistics()
                self.status_bar.showMessage(f"Processed in {stats.avg_latency_ms:.1f}ms")
                self.update_performance_display(stats)
                
                self.progress_bar.setVisible(False)
                
            except Exception as e:
                from PyQt5.QtWidgets import QMessageBox
                QMessageBox.critical(self.main_window, "Error", f"Processing failed: {e}")
                self.progress_bar.setVisible(False)
    
    def display_original_image(self):
        """Display the original image."""
        if self.current_image is not None:
            self._display_image(self.current_image, self.orig_label)
    
    def display_processed_image(self):
        """Display the processed image."""
        if self.processed_image is not None:
            self._display_image(self.processed_image, self.proc_label)
    
    def _display_image(self, image, label):
        """Display an image in a label widget."""
        from PyQt5.QtGui import QPixmap, QImage
        from PyQt5.QtCore import Qt
        
        # Get label size
        label_size = label.size()
        max_width = label_size.width() - 20
        max_height = label_size.height() - 20
        
        # Resize image to fit in the label
        height, width = image.shape[:2]
        
        if max_width > 0 and max_height > 0:
            # Calculate scale to fit in label
            scale_w = max_width / width
            scale_h = max_height / height
            scale = min(scale_w, scale_h, 1.0)  # Don't upscale
            
            if scale < 1.0:
                new_width = int(width * scale)
                new_height = int(height * scale)
                image = cv2.resize(image, (new_width, new_height))
        
        # Convert BGR to RGB for Qt
        if len(image.shape) == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
        
        # Convert to QImage
        height, width = image_rgb.shape[:2]
        if len(image_rgb.shape) == 3:
            bytes_per_line = 3 * width
            q_image = QImage(image_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)
        else:
            bytes_per_line = width
            q_image = QImage(image_rgb.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
        
        # Convert to QPixmap and set to label
        pixmap = QPixmap.fromImage(q_image)
        label.setPixmap(pixmap)
        label.setScaledContents(False)
        label.setAlignment(Qt.AlignCenter)
    
    def update_live_display(self):
        """Update display with live camera feed."""
        frame = self.pipeline.get_processed_frame(timeout_ms=30)
        if frame is not None:
            self.processed_image = frame
            self.display_processed_image()
            
            # Update performance display
            stats = self.pipeline.get_statistics()
            self.update_performance_display(stats)
    
    def update_performance_display(self, stats):
        """Update performance indicators."""
        if self.show_fps_check.isChecked():
            self.fps_label.setText(f"FPS: {stats.avg_fps:.1f}")
            self.latency_label.setText(f"Latency: {stats.avg_latency_ms:.1f}ms")
            self.memory_label.setText(f"Memory: {stats.memory_usage_mb}MB")
    
    def on_frame_processed(self, frame, processing_time):
        """Callback for processed frames from camera."""
        # This will be called from the processing thread
        # We need to update the GUI in the main thread
        pass  # Live updates are handled by update_live_display()
    
    def log_message(self, message):
        """Add a message to the log display."""
        import datetime
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        formatted_message = f"[{timestamp}] {message}"
        self.log_text.append(formatted_message)
        
        # Auto-scroll to bottom
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
        
        # Limit log size
        if self.log_text.document().blockCount() > 100:
            cursor = self.log_text.textCursor()
            cursor.movePosition(cursor.Start)
            cursor.select(cursor.BlockUnderCursor)
            cursor.removeSelectedText()
    
    def closeEvent(self, event):
        """Handle window close event."""
        if self.pipeline.processing_thread:
            self.stop_camera()
        event.accept()
    
    def run(self):
        """Run the GUI application."""
        self.main_window.show()
        return self.app.exec_()

def main():
    """Main function for standalone execution."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Real-Time Image Denoising Pipeline")
    parser.add_argument("--input", "-i", help="Input image path")
    parser.add_argument("--output", "-o", help="Output image path")
    parser.add_argument("--algorithm", "-a", choices=["bilateral", "nlm", "gaussian", "adaptive"],
                       default="bilateral", help="Denoising algorithm")
    parser.add_argument("--camera", "-c", action="store_true", help="Use camera input")
    parser.add_argument("--gui", "-g", action="store_true", help="Launch GUI application")
    parser.add_argument("--benchmark", "-b", action="store_true", help="Run benchmark")
    
    # Algorithm parameters
    parser.add_argument("--sigma-color", type=float, default=50.0, help="Bilateral filter color sigma")
    parser.add_argument("--sigma-space", type=float, default=50.0, help="Bilateral filter space sigma")
    parser.add_argument("--h-param", type=float, default=10.0, help="NLM filtering strength")
    parser.add_argument("--template-size", type=int, default=7, help="NLM template window size")
    parser.add_argument("--search-size", type=int, default=21, help="NLM search window size")
    parser.add_argument("--gaussian-sigma", type=float, default=1.0, help="Gaussian sigma")
    parser.add_argument("--kernel-size", type=int, default=5, help="Kernel size")
    
    args = parser.parse_args()
    
    # Create pipeline
    pipeline = DenoisingPipeline()
    
    # Set algorithm
    algorithm_map = {
        "bilateral": DenoiseAlgorithm.BILATERAL,
        "nlm": DenoiseAlgorithm.NON_LOCAL_MEANS,
        "gaussian": DenoiseAlgorithm.GAUSSIAN,
        "adaptive": DenoiseAlgorithm.ADAPTIVE_BILATERAL
    }
    pipeline.set_algorithm(algorithm_map[args.algorithm])
    
    # Set parameters
    params = DenoiseParams(
        sigma_color=args.sigma_color,
        sigma_space=args.sigma_space,
        h=args.h_param,
        template_window_size=args.template_size,
        search_window_size=args.search_size,
        gaussian_sigma=args.gaussian_sigma,
        kernel_size=args.kernel_size
    )
    pipeline.set_parameters(params)
    
    try:
        if args.gui:
            # Launch GUI
            gui = pipeline.create_gui_application()
            if gui:
                return gui.run()
            else:
                logger.error("Failed to create GUI application")
                return 1
        
        elif args.benchmark:
            # Run benchmark
            logger.info("Creating test image for benchmark...")
            test_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
            
            results = pipeline.benchmark_algorithms(test_image, iterations=50)
            
            print("\n=== Benchmark Results ===")
            print(f"{'Algorithm':<20} {'Avg Time':<12} {'Min Time':<12} {'Max Time':<12} {'Avg FPS':<10}")
            print("-" * 70)
            
            for name, result in results.items():
                print(f"{name:<20} {result['avg_time_ms']:<12.2f} {result['min_time_ms']:<12.2f} "
                      f"{result['max_time_ms']:<12.2f} {result['avg_fps']:<10.1f}")
        
        elif args.camera:
            # Real-time camera processing
            logger.info("Starting real-time camera processing...")
            logger.info("Press 'q' to quit, 's' to save frame")
            
            if not pipeline.start_camera_processing():
                logger.error("Failed to start camera processing")
                return 1
            
            cv2.namedWindow("Real-time Denoising", cv2.WINDOW_AUTOSIZE)
            
            try:
                while True:
                    frame = pipeline.get_processed_frame(timeout_ms=100)
                    if frame is not None:
                        cv2.imshow("Real-time Denoising", frame)
                    
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q') or key == 27:  # 'q' or ESC
                        break
                    elif key == ord('s') and frame is not None:
                        # Save current frame
                        timestamp = int(time.time())
                        filename = f"denoised_frame_{timestamp}.jpg"
                        cv2.imwrite(filename, frame)
                        logger.info(f"Frame saved as: {filename}")
            
            finally:
                pipeline.stop_camera_processing()
                cv2.destroyAllWindows()
        
        elif args.input:
            # Single image processing
            logger.info(f"Processing image: {args.input}")
            
            image = cv2.imread(args.input)
            if image is None:
                logger.error(f"Failed to load image: {args.input}")
                return 1
            
            start_time = time.time()
            result = pipeline.denoise_image(image)
            processing_time = (time.time() - start_time) * 1000
            
            # Convert back to 8-bit
            result_8bit = (result * 255).astype(np.uint8)
            
            # Save result
            output_path = args.output
            if not output_path:
                base, ext = os.path.splitext(args.input)
                output_path = f"{base}_denoised{ext}"
            
            cv2.imwrite(output_path, result_8bit)
            
            logger.info(f"Processing completed in {processing_time:.2f}ms")
            logger.info(f"Result saved to: {output_path}")
            
            # Show statistics
            stats = pipeline.get_statistics()
            logger.info(f"Average FPS: {stats.avg_fps:.1f}")
        
        else:
            # No mode specified
            print("No operation specified. Use --help for usage information.")
            print("Quick examples:")
            print("  python denoising_pipeline.py --input image.jpg")
            print("  python denoising_pipeline.py --camera")
            print("  python denoising_pipeline.py --gui")
            print("  python denoising_pipeline.py --benchmark")
    
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        return 0
    except Exception as e:
        logger.error(f"Error: {e}")
        return 1
    finally:
        # Cleanup
        if pipeline.processing_thread:
            pipeline.stop_camera_processing()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

def main():
    """Main function for standalone execution."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Real-Time Image Denoising Pipeline")
    parser.add_argument("--input", "-i", help="Input image path")
    parser.add_argument("--output", "-o", help="Output image path")
    parser.add_argument("--algorithm", "-a", choices=["bilateral", "nlm", "gaussian", "adaptive"],
                       default="bilateral", help="Denoising algorithm")
    parser.add_argument("--camera", "-c", action="store_true", help="Use camera input")
    parser.add_argument("--gui", "-g", action="store_true", help="Launch GUI application")
    parser.add_argument("--benchmark", "-b", action="store_true", help="Run benchmark")
    
    # Algorithm parameters
    parser.add_argument("--sigma-color", type=float, default=50.0, help="Bilateral filter color sigma")
    parser.add_argument("--sigma-space", type=float, default=50.0, help="Bilateral filter space sigma")
    parser.add_argument("--h-param", type=float, default=10.0, help="NLM filtering strength")
    parser.add_argument("--template-size", type=int, default=7, help="NLM template window size")
    parser.add_argument("--search-size", type=int, default=21, help="NLM search window size")
    parser.add_argument("--gaussian-sigma", type=float, default=1.0, help="Gaussian sigma")
    parser.add_argument("--kernel-size", type=int, default=5, help="Kernel size")
    
    args = parser.parse_args()
    
    # Create pipeline
    pipeline = DenoisingPipeline()
    
    # Set algorithm
    algorithm_map = {
        "bilateral": DenoiseAlgorithm.BILATERAL,
        "nlm": DenoiseAlgorithm.NON_LOCAL_MEANS,
        "gaussian": DenoiseAlgorithm.GAUSSIAN,
        "adaptive": DenoiseAlgorithm.ADAPTIVE_BILATERAL
    }
    pipeline.set_algorithm(algorithm_map[args.algorithm])
    
    # Set parameters
    params = DenoiseParams(
        sigma_color=args.sigma_color,
        sigma_space=args.sigma_space,
        h=args.h_param,
        template_window_size=args.template_size,
        search_window_size=args.search_size,
        gaussian_sigma=args.gaussian_sigma,
        kernel_size=args.kernel_size
    )
    pipeline.set_parameters(params)
    
    try:
        if args.gui:
            # Launch GUI
            gui = pipeline.create_gui_application()
            if gui:
                gui.run()
            else:
                logger.error("Failed to create GUI application")
        
        elif args.benchmark:
            # Run benchmark
            logger.info("Creating test image for benchmark...")
            test_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
            
            results = pipeline.benchmark_algorithms(test_image, iterations=50)
            
            print("\n=== Benchmark Results ===")
            print(f"{'Algorithm':<20} {'Avg Time':<12} {'Min Time':<12} {'Max Time':<12} {'Avg FPS':<10}")
            print("-" * 70)
            
            for name, result in results.items():
                print(f"{name:<20} {result['avg_time_ms']:<12.2f} {result['min_time_ms']:<12.2f} "
                      f"{result['max_time_ms']:<12.2f} {result['avg_fps']:<10.1f}")
        
        elif args.camera:
            # Real-time camera processing
            logger.info("Starting real-time camera processing...")
            logger.info("Press 'q' to quit, 's' to save frame")
            
            if not pipeline.start_camera_processing():
                logger.error("Failed to start camera processing")
                return
            
            cv2.namedWindow("Real-time Denoising", cv2.WINDOW_AUTOSIZE)
            
            try:
                while True:
                    frame = pipeline.get_processed_frame(timeout_ms=100)
                    if frame is not None:
                        cv2.imshow("Real-time Denoising", frame)
                    
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q') or key == 27:  # 'q' or ESC
                        break
                    elif key == ord('s') and frame is not None:
                        # Save current frame
                        timestamp = int(time.time())
                        filename = f"denoised_frame_{timestamp}.jpg"
                        cv2.imwrite(filename, frame)
                        logger.info(f"Frame saved as: {filename}")
            
            finally:
                pipeline.stop_camera_processing()
                cv2.destroyAllWindows()
        
        elif args.input:
            # Single image processing
            logger.info(f"Processing image: {args.input}")
            
            image = cv2.imread(args.input)
            if image is None:
                logger.error(f"Failed to load image: {args.input}")
                return
            
            start_time = time.time()
            result = pipeline.denoise_image(image)
            processing_time = (time.time() - start_time) * 1000
            
            # Convert back to 8-bit
            result_8bit = (result * 255).astype(np.uint8)
            
            # Save result
            output_path = args.output
            if not output_path:
                base, ext = os.path.splitext(args.input)
                output_path = f"{base}_denoised{ext}"
            
            cv2.imwrite(output_path, result_8bit)
            
            logger.info(f"Processing completed in {processing_time:.2f}ms")
            logger.info(f"Result saved to: {output_path}")
            
            # Show statistics
            stats = pipeline.get_statistics()
            logger.info(f"Average FPS: {stats.avg_fps:.1f}")
        
        else:
            # No mode specified
            print("No operation specified. Use --help for usage information.")
            print("Quick examples:")
            print("  python denoising_pipeline.py --input image.jpg")
            print("  python denoising_pipeline.py --camera")
            print("  python denoising_pipeline.py --gui")
            print("  python denoising_pipeline.py --benchmark")
    
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Error: {e}")
    finally:
        # Cleanup
        if pipeline.processing_thread:
            pipeline.stop_camera_processing()

if __name__ == "__main__":
    main()