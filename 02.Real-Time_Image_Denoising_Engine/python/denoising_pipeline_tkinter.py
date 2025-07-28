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
        """Create a simple GUI application for interactive denoising."""
        try:
            import tkinter as tk
            from tkinter import ttk, filedialog, messagebox
            from PIL import Image, ImageTk
        except ImportError:
            logger.error("GUI dependencies not available. Install tkinter and Pillow.")
            return None
        
        return DenoisingGUI(self)

class DenoisingGUI:
    """Simple GUI application for interactive denoising."""
    
    def __init__(self, pipeline: DenoisingPipeline):
        self.pipeline = pipeline
        self.root = tk.Tk()
        self.root.title("Real-Time Image Denoising")
        self.root.geometry("800x600")
        
        self.current_image = None
        self.processed_image = None
        
        self._create_widgets()
        self._setup_callbacks()
    
    def _create_widgets(self):
        """Create GUI widgets."""
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Control panel
        control_frame = ttk.LabelFrame(main_frame, text="Controls")
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        # File operations
        file_frame = ttk.Frame(control_frame)
        file_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(file_frame, text="Load Image", command=self.load_image).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(file_frame, text="Save Result", command=self.save_image).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(file_frame, text="Start Camera", command=self.start_camera).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(file_frame, text="Stop Camera", command=self.stop_camera).pack(side=tk.LEFT)
        
        # Algorithm selection
        algo_frame = ttk.Frame(control_frame)
        algo_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(algo_frame, text="Algorithm:").pack(side=tk.LEFT)
        self.algo_var = tk.StringVar(value="BILATERAL")
        algo_combo = ttk.Combobox(algo_frame, textvariable=self.algo_var, 
                                 values=["BILATERAL", "NON_LOCAL_MEANS", "GAUSSIAN", "ADAPTIVE_BILATERAL"])
        algo_combo.pack(side=tk.LEFT, padx=(5, 0))
        algo_combo.bind('<<ComboboxSelected>>', self.on_algorithm_change)
        
        # Parameters
        params_frame = ttk.LabelFrame(control_frame, text="Parameters")
        params_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Create parameter controls (simplified)
        self.sigma_color_var = tk.DoubleVar(value=50.0)
        self.sigma_space_var = tk.DoubleVar(value=50.0)
        self.h_var = tk.DoubleVar(value=10.0)
        
        ttk.Label(params_frame, text="Sigma Color:").grid(row=0, column=0, sticky=tk.W)
        ttk.Scale(params_frame, from_=1, to=100, variable=self.sigma_color_var, 
                 orient=tk.HORIZONTAL, command=self.on_param_change).grid(row=0, column=1, sticky=tk.EW)
        
        ttk.Label(params_frame, text="Sigma Space:").grid(row=1, column=0, sticky=tk.W)
        ttk.Scale(params_frame, from_=1, to=100, variable=self.sigma_space_var,
                 orient=tk.HORIZONTAL, command=self.on_param_change).grid(row=1, column=1, sticky=tk.EW)
        
        params_frame.columnconfigure(1, weight=1)
        
        # Image display
        image_frame = ttk.Frame(main_frame)
        image_frame.pack(fill=tk.BOTH, expand=True)
        
        # Original image
        orig_frame = ttk.LabelFrame(image_frame, text="Original")
        orig_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        self.orig_label = ttk.Label(orig_frame)
        self.orig_label.pack(fill=tk.BOTH, expand=True)
        
        # Processed image
        proc_frame = ttk.LabelFrame(image_frame, text="Denoised")
        proc_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        self.proc_label = ttk.Label(proc_frame)
        self.proc_label.pack(fill=tk.BOTH, expand=True)
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.pack(fill=tk.X, pady=(10, 0))
    
    def _setup_callbacks(self):
        """Setup event callbacks."""
        self.pipeline.set_frame_callback(self.on_frame_processed)
    
    def load_image(self):
        """Load an image file."""
        filename = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff")]
        )
        
        if filename:
            self.current_image = cv2.imread(filename)
            if self.current_image is not None:
                self.display_original_image()
                self.process_current_image()
                self.status_var.set(f"Loaded: {os.path.basename(filename)}")
            else:
                messagebox.showerror("Error", "Failed to load image")
    
    def save_image(self):
        """Save the processed image."""
        if self.processed_image is not None:
            filename = filedialog.asksaveasfilename(
                title="Save Processed Image",
                defaultextension=".jpg",
                filetypes=[("JPEG files", "*.jpg"), ("PNG files", "*.png")]
            )
            
            if filename:
                cv2.imwrite(filename, self.processed_image)
                self.status_var.set(f"Saved: {os.path.basename(filename)}")
        else:
            messagebox.showwarning("Warning", "No processed image to save")
    
    def start_camera(self):
        """Start camera processing."""
        if self.pipeline.start_camera_processing():
            self.status_var.set("Camera processing started")
        else:
            messagebox.showerror("Error", "Failed to start camera")
    
    def stop_camera(self):
        """Stop camera processing."""
        self.pipeline.stop_camera_processing()
        self.status_var.set("Camera processing stopped")
    
    def on_algorithm_change(self, event=None):
        """Handle algorithm selection change."""
        algo_name = self.algo_var.get()
        algorithm = DenoiseAlgorithm[algo_name]
        self.pipeline.set_algorithm(algorithm)
        
        if self.current_image is not None:
            self.process_current_image()
    
    def on_param_change(self, value=None):
        """Handle parameter changes."""
        params = DenoiseParams(
            sigma_color=self.sigma_color_var.get(),
            sigma_space=self.sigma_space_var.get(),
            h=self.h_var.get()
        )
        self.pipeline.set_parameters(params)
        
        if self.current_image is not None:
            self.process_current_image()
    
    def process_current_image(self):
        """Process the current image."""
        if self.current_image is not None:
            try:
                processed = self.pipeline.denoise_image(self.current_image)
                self.processed_image = (processed * 255).astype(np.uint8)
                self.display_processed_image()
                
                stats = self.pipeline.get_statistics()
                self.status_var.set(f"Processed in {stats.avg_latency_ms:.1f}ms")
            except Exception as e:
                messagebox.showerror("Error", f"Processing failed: {e}")
    
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
        # Resize image to fit in the label
        height, width = image.shape[:2]
        max_size = 300
        
        if max(height, width) > max_size:
            scale = max_size / max(height, width)
            new_width = int(width * scale)
            new_height = int(height * scale)
            image = cv2.resize(image, (new_width, new_height))
        
        # Convert BGR to RGB for PIL
        if len(image.shape) == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
        
        # Convert to PIL Image and then to PhotoImage
        pil_image = Image.fromarray(image_rgb)
        photo = ImageTk.PhotoImage(pil_image)
        
        label.configure(image=photo)
        label.image = photo  # Keep a reference
    
    def on_frame_processed(self, frame, processing_time):
        """Callback for processed frames from camera."""
        # Update the processed image display
        self.processed_image = frame
        self.display_processed_image()
        
        # Update status
        stats = self.pipeline.get_statistics()
        self.status_var.set(f"Live: {stats.avg_fps:.1f} FPS, {processing_time:.1f}ms")
    
    def run(self):
        """Run the GUI application."""
        self.root.mainloop()

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