"""
UNIFIED OHS + FACE RECOGNITION SYSTEM
Menggabungkan deteksi safety equipment dengan face recognition
Author: BJBS Safety System
"""

import cv2
import numpy as np
from ultralytics import YOLO
import onnxruntime as ort
from sklearn.metrics.pairwise import cosine_similarity
import os
import pickle
import time
from datetime import datetime
from collections import defaultdict, deque
import threading
from queue import Queue
import torch

# ==================== CONFIGURATION ====================
class Config:
    # Model paths
    OHS_MODEL = "models/ohs_yolo.pt"        # YOLO untuk person, helmet, vest
    FACE_REC_MODEL = "models/glintr100.onnx"  # Face recognition
    
    # Database
    AUTHORIZED_DIR = "bjbs-authorized"
    EMBEDDINGS_CACHE = "authorized_embeddings.pkl"
    
    # Icons
    HELMET_ICON = "icons/helmet.png"
    VEST_ICON = "icons/vest.png"
    
    # Detection thresholds
    OHS_CONF = 0.4
    FACE_CONF = 0.3
    FACE_SIMILARITY = 0.35  # Threshold untuk face recognition
    
    # Display
    ICON_SIZE = 48
    SHOW_FPS = True
    SAVE_VIOLATIONS = True
    VIOLATION_DIR = "violations"
    
    # Performance
    USE_THREADING = True
    QUEUE_SIZE = 2  # Frame buffer size
    TARGET_FPS = 30

# ==================== FACE RECOGNIZER MODULE ====================
# ==================== IMPROVED FACE RECOGNIZER ====================
# ==================== IMPROVED FACE RECOGNIZER ====================
class FaceRecognizer:
    """Handle face recognition dengan support multiple photos per person"""
    
    def __init__(self, model_path, authorized_dir, cache_file):
        self.model_path = model_path
        self.authorized_dir = authorized_dir
        self.cache_file = cache_file
        self.session = None
        self.authorized_embeddings = {}  # Format: {person_name: [embedding1, embedding2, ...]}
        
        self.init_model()
        self.load_database()
    
    def init_model(self):
        """Initialize ONNX face recognition model"""
        if not os.path.exists(self.model_path):
            print(f"⚠️  Face recognition model not found: {self.model_path}")
            return
        
        providers = []
        if 'CUDAExecutionProvider' in ort.get_available_providers():
            providers.append(('CUDAExecutionProvider', {
                'device_id': 0,
                'arena_extend_strategy': 'kNextPowerOfTwo',
                'gpu_mem_limit': 4 * 1024 * 1024 * 1024,
                'cudnn_conv_algo_search': 'EXHAUSTIVE',
            }))
        providers.append('CPUExecutionProvider')
        
        self.session = ort.InferenceSession(self.model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        print(f"✅ Face recognition model loaded: {self.session.get_providers()[0]}")
    
    def parse_person_name(self, filename):
        """
        Parse person name from filename
        Format: NamaOrang_XX.jpg -> NamaOrang
        Contoh: 
          - Dwi_Ahmad_01.jpg -> Dwi_Ahmad
          - Dwi_Ahmad_02.jpg -> Dwi_Ahmad
          - John_Doe.jpg -> John_Doe
        """
        name_without_ext = os.path.splitext(filename)[0]
        
        # Split by underscore
        parts = name_without_ext.split('_')
        
        # Check if last part is numeric (photo ID)
        if len(parts) > 1 and parts[-1].isdigit():
            # Remove photo ID, join the rest
            person_name = '_'.join(parts[:-1])
        else:
            # No photo ID, use whole name
            person_name = name_without_ext
        
        return person_name
    
    def load_database(self):
        """Load authorized persons database with multiple photos support"""
        if not os.path.exists(self.authorized_dir):
            os.makedirs(self.authorized_dir)
            print(f"📁 Created {self.authorized_dir}/ - Add authorized person images here")
            print(f"💡 Tip: Use format 'Name_01.jpg', 'Name_02.jpg' for multiple photos")
            return
        
        image_files = [f for f in os.listdir(self.authorized_dir) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if not image_files:
            print(f"⚠️  No images in {self.authorized_dir}/")
            return
        
        # Group files by person
        person_files = defaultdict(list)
        for filename in image_files:
            person_name = self.parse_person_name(filename)
            person_files[person_name].append(filename)
        
        # Try load from cache
        need_recompute = True
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'rb') as f:
                    cached = pickle.load(f)
                
                # Check if cache is still valid
                if set(cached.keys()) == set(person_files.keys()):
                    # Check if number of embeddings per person matches
                    cache_valid = all(
                        len(cached[name]) == len(person_files[name])
                        for name in cached.keys()
                    )
                    if cache_valid:
                        self.authorized_embeddings = cached
                        total_photos = sum(len(embs) for embs in cached.values())
                        print(f"✅ Loaded {len(cached)} persons ({total_photos} photos) from cache")
                        need_recompute = False
            except Exception as e:
                print(f"Cache load failed: {e}")
        
        if need_recompute:
            self.compute_embeddings(person_files)
    
    def compute_embeddings(self, person_files):
        """Compute face embeddings for all authorized persons"""
        total_files = sum(len(files) for files in person_files.values())
        print(f"🔄 Computing embeddings for {len(person_files)} persons ({total_files} photos)...")
        
        for person_name, filenames in person_files.items():
            embeddings = []
            
            for filename in filenames:
                img_path = os.path.join(self.authorized_dir, filename)
                img = cv2.imread(img_path)
                if img is None:
                    continue
                
                embedding = self.get_embedding(img)
                if embedding is not None:
                    embeddings.append(embedding)
                    print(f"  ✓ {person_name} ({filename})")
            
            if embeddings:
                self.authorized_embeddings[person_name] = embeddings
        
        # Save cache
        if self.authorized_embeddings:
            total_photos = sum(len(embs) for embs in self.authorized_embeddings.values())
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.authorized_embeddings, f)
            print(f"💾 Saved {len(self.authorized_embeddings)} persons ({total_photos} photos)")
    
    def preprocess_face(self, face_img):
        """Preprocess face for recognition model (112x112, normalized)"""
        try:
            if face_img.size == 0:
                return None
            face_resized = cv2.resize(face_img, (112, 112))
            face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
            face_normalized = (face_rgb.astype(np.float32) - 127.5) / 127.5
            face_input = np.transpose(face_normalized, (2, 0, 1))
            face_input = np.expand_dims(face_input, axis=0)
            return face_input
        except:
            return None
    
    def get_embedding(self, face_img):
        """Extract face embedding from face image"""
        if self.session is None:
            return None
        
        face_input = self.preprocess_face(face_img)
        if face_input is None:
            return None
        
        try:
            embedding = self.session.run([self.output_name], {self.input_name: face_input})[0]
            embedding = embedding / np.linalg.norm(embedding)
            return embedding.flatten()
        except:
            return None
    
    def recognize(self, face_img, threshold=0.35):
        """
        Recognize face against database
        Match against ALL embeddings of each person, use highest similarity
        """
        if not self.authorized_embeddings:
            return None, 0.0
        
        embedding = self.get_embedding(face_img)
        if embedding is None:
            return None, 0.0
        
        best_match = None
        best_similarity = 0.0
        
        # For each person in database
        for person_name, person_embeddings in self.authorized_embeddings.items():
            # Compare with ALL photos of this person
            for auth_embedding in person_embeddings:
                similarity = cosine_similarity(
                    embedding.reshape(1, -1), 
                    auth_embedding.reshape(1, -1)
                )[0][0]
                
                # Keep track of best match across all persons
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = person_name
        
        # Return match if above threshold (with underscore replaced by space for display)
        if best_similarity >= threshold:
            display_name = best_match.replace('_', ' ')
            return display_name, best_similarity
        else:
            return None, best_similarity
        

# ==================== OHS DETECTOR MODULE ====================
class OHSDetector:
    """Handle OHS equipment detection (person, helmet, vest)"""
    
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        
        # Force GPU if available
        if torch.cuda.is_available():
            self.model.to('cuda:0')
            print(f"✅ OHS detector on GPU")
        
        # Get class IDs
        names = self.model.names
        inv = {v.lower(): k for k, v in names.items()}
        self.CID_PERSON = inv.get("person")
        self.CID_HELMET = inv.get("helmet")
        self.CID_VEST = inv.get("vest")
        self.CID_NOHELMET = inv.get("no-helmet")
        self.CID_NOVEST = inv.get("no-vest")
    
    def detect(self, frame, conf_threshold=0.7):
        """Detect persons and OHS equipment"""
        results = self.model.predict(frame, conf=conf_threshold, verbose=False)[0]
        
        persons, helmets, vests = [], [], []
        
        if results.boxes is not None and len(results.boxes) > 0:
            boxes = results.boxes.xyxy.cpu().numpy()
            clses = results.boxes.cls.cpu().numpy().astype(int)
            
            for (x1, y1, x2, y2), c in zip(boxes, clses):
                box = (int(x1), int(y1), int(x2), int(y2))
                if c == self.CID_PERSON:
                    persons.append(box)
                elif c == self.CID_HELMET:
                    helmets.append(box)
                elif c == self.CID_VEST:
                    vests.append(box)
        
        return persons, helmets, vests
    
    @staticmethod
    def center_in(box_person, box_item):
        """Check if item center is inside person box"""
        x1p, y1p, x2p, y2p = box_person
        x1i, y1i, x2i, y2i = box_item
        cx = (x1i + x2i) / 2
        cy = (y1i + y2i) / 2
        return (x1p <= cx <= x2p) and (y1p <= cy <= y2p)

# ==================== SMOOTH TRACKER MODULE ====================
class PersonTracker:
    """
    Track persons across frames dengan smooth bounding box
    Mengatasi jitter/shaking pada deteksi
    """
    
    def __init__(self, smoothing_factor=0.3, iou_threshold=0.3, max_lost_frames=15):
        """
        Args:
            smoothing_factor: 0-1, makin kecil makin smooth (tapi lebih delay)
            iou_threshold: minimum IOU untuk matching detection dengan track
            max_lost_frames: max frame hilang sebelum track dihapus
        """
        self.tracks = {}  # {track_id: track_info}
        self.next_track_id = 0
        self.smoothing_factor = smoothing_factor
        self.iou_threshold = iou_threshold
        self.max_lost_frames = max_lost_frames
    
    @staticmethod
    def calculate_iou(box1, box2):
        """Calculate Intersection over Union"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Intersection area
        xi1 = max(x1_1, x1_2)
        yi1 = max(y1_1, y1_2)
        xi2 = min(x2_1, x2_2)
        yi2 = min(y2_1, y2_2)
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        
        # Union area
        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0
    
    @staticmethod
    def smooth_box(old_box, new_box, alpha):
        """Exponential Moving Average untuk smooth coordinates"""
        return tuple(
            int(alpha * new + (1 - alpha) * old)
            for old, new in zip(old_box, new_box)
        )
    
    def update(self, detections):
        """
        Update tracks dengan deteksi baru
        
        Args:
            detections: List of dict with keys: 'bbox', 'name', 'has_helmet', 'has_vest', etc.
        
        Returns:
            List of tracked results dengan smooth bounding boxes
        """
        # Match detections dengan existing tracks
        matched_tracks = set()
        matched_detections = set()
        
        # Untuk setiap detection, cari track terdekat
        for det_idx, detection in enumerate(detections):
            det_box = detection['bbox']
            best_iou = 0
            best_track_id = None
            
            # Cari track dengan IOU tertinggi
            for track_id, track in self.tracks.items():
                iou = self.calculate_iou(det_box, track['smooth_bbox'])
                if iou > best_iou and iou >= self.iou_threshold:
                    best_iou = iou
                    best_track_id = track_id
            
            # Jika ada match
            if best_track_id is not None:
                track = self.tracks[best_track_id]
                
                # Smooth bounding box dengan EMA
                track['smooth_bbox'] = self.smooth_box(
                    track['smooth_bbox'], 
                    det_box, 
                    self.smoothing_factor
                )
                
                # Update info lainnya (langsung tanpa smoothing)
                track['name'] = detection['name']
                track['has_helmet'] = detection['has_helmet']
                track['has_vest'] = detection['has_vest']
                track['is_compliant'] = detection['is_compliant']
                track['is_authorized'] = detection['is_authorized']
                track['similarity'] = detection.get('similarity', 0.0)
                track['lost_frames'] = 0  # Reset lost counter
                
                matched_tracks.add(best_track_id)
                matched_detections.add(det_idx)
        
        # Buat track baru untuk deteksi yang tidak match
        for det_idx, detection in enumerate(detections):
            if det_idx not in matched_detections:
                track_id = self.next_track_id
                self.next_track_id += 1
                
                self.tracks[track_id] = {
                    'track_id': track_id,
                    'smooth_bbox': detection['bbox'],  # Initial bbox
                    'name': detection['name'],
                    'has_helmet': detection['has_helmet'],
                    'has_vest': detection['has_vest'],
                    'is_compliant': detection['is_compliant'],
                    'is_authorized': detection['is_authorized'],
                    'similarity': detection.get('similarity', 0.0),
                    'lost_frames': 0
                }
                matched_tracks.add(track_id)
        
        # Update lost frames untuk track yang tidak match
        tracks_to_remove = []
        for track_id, track in self.tracks.items():
            if track_id not in matched_tracks:
                track['lost_frames'] += 1
                
                # Hapus track jika sudah lama hilang
                if track['lost_frames'] > self.max_lost_frames:
                    tracks_to_remove.append(track_id)
        
        # Hapus track yang hilang
        for track_id in tracks_to_remove:
            del self.tracks[track_id]
        
        # Return tracked results dengan smooth bbox
        return [
            {
                'bbox': track['smooth_bbox'],
                'name': track['name'],
                'has_helmet': track['has_helmet'],
                'has_vest': track['has_vest'],
                'is_compliant': track['is_compliant'],
                'is_authorized': track['is_authorized'],
                'similarity': track['similarity'],
                'track_id': track['track_id']
            }
            for track in self.tracks.values()
        ]
    
    def reset(self):
        """Reset semua tracks"""
        self.tracks = {}
        self.next_track_id = 0

# ==================== UNIFIED SYSTEM ====================
class UnifiedSafetySystem:
    """
    Unified OHS + Face Recognition System dengan Smooth Tracking
    """
    
    def __init__(self, config=Config):
        self.cfg = config
        
        print("="*60)
        print("🔧 INITIALIZING UNIFIED SAFETY RECOGNITION SYSTEM")
        print("="*60)
        
        # Initialize modules
        self.ohs_detector = OHSDetector(config.OHS_MODEL)
        self.face_recognizer = FaceRecognizer(
            config.FACE_REC_MODEL,
            config.AUTHORIZED_DIR,
            config.EMBEDDINGS_CACHE
        )
        
        # Initialize tracker untuk smooth bbox
        self.tracker = PersonTracker(
            smoothing_factor=0.3,      # 0.3 = smooth, 0.7 = responsive
            iou_threshold=0.3,         # Threshold untuk matching
            max_lost_frames=15         # Max frame sebelum track hilang
        )
        
        # Load icons
        self.helmet_icon = self.load_icon(config.HELMET_ICON, config.ICON_SIZE)
        self.vest_icon = self.load_icon(config.VEST_ICON, config.ICON_SIZE)
        
        # Threading setup
        if config.USE_THREADING:
            self.frame_queue = Queue(maxsize=config.QUEUE_SIZE)
            self.result_queue = Queue(maxsize=config.QUEUE_SIZE)
            self.processing = True
        
        # Violation tracking
        if config.SAVE_VIOLATIONS:
            os.makedirs(config.VIOLATION_DIR, exist_ok=True)
        
        # FPS tracking
        self.fps_tracker = deque(maxlen=30)
        
        print("="*60)
        print("✅ SYSTEM READY (with Smooth Tracking)")
        print("="*60)
    
    def load_icon(self, path, size):
        """Load and resize icon with alpha channel"""
        if not os.path.exists(path):
            print(f"⚠️  Icon not found: {path}")
            return None
        icon = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if icon is None:
            return None
        h, w = icon.shape[:2]
        scale = size / w
        icon = cv2.resize(icon, (size, max(1, int(h*scale))))
        return icon
    
    def paste_icon(self, dst, icon, x, y):
        """Alpha blend icon onto frame"""
        if icon is None:
            return
        ih, iw = icon.shape[:2]
        H, W = dst.shape[:2]
        if x >= W or y >= H:
            return
        
        x2, y2 = min(W, x+iw), min(H, y+ih)
        roi = dst[y:y2, x:x2]
        icon_crop = icon[0:(y2-y), 0:(x2-x)]
        
        if icon_crop.shape[2] == 4:
            rgb = icon_crop[:, :, :3]
            alpha = icon_crop[:, :, 3:4] / 255.0
            roi[:] = (alpha * rgb + (1 - alpha) * roi).astype(np.uint8)
    
    def extract_face_from_person(self, frame, person_box):
        """Extract face region from person detection box"""
        x1, y1, x2, y2 = person_box
        
        # Focus on upper 40% of person box (likely head region)
        h_person = y2 - y1
        head_y2 = y1 + int(h_person * 0.4)
        
        # Add padding
        padding = 15
        H, W = frame.shape[:2]
        y1_crop = max(0, y1 - padding)
        y2_crop = min(H, head_y2 + padding)
        x1_crop = max(0, x1 - padding)
        x2_crop = min(W, x2 + padding)
        
        return frame[y1_crop:y2_crop, x1_crop:x2_crop]
    
    def process_frame(self, frame):
        """
        Main processing pipeline dengan smooth tracking:
        1. Detect persons + OHS equipment
        2. For each person: extract face and recognize
        3. Update tracker untuk smooth bounding box
        """
        start_time = time.time()
        
        # Stage 1: Detect persons and OHS equipment
        persons, helmets, vests = self.ohs_detector.detect(frame, self.cfg.OHS_CONF)
        
        # Stage 2: Process each person (raw detection)
        raw_results = []
        for person_box in persons:
            x1, y1, x2, y2 = person_box
            
            # Check OHS compliance
            has_helmet = any(self.ohs_detector.center_in(person_box, h) for h in helmets)
            has_vest = any(self.ohs_detector.center_in(person_box, v) for v in vests)
            
            # Face recognition
            face_region = self.extract_face_from_person(frame, person_box)
            person_name, similarity = self.face_recognizer.recognize(
                face_region, 
                self.cfg.FACE_SIMILARITY
            )
            
            # Determine status
            is_compliant = has_helmet and has_vest
            is_authorized = person_name is not None
            
            raw_results.append({
                'bbox': person_box,
                'name': person_name if is_authorized else "Unknown",
                'similarity': similarity,
                'has_helmet': has_helmet,
                'has_vest': has_vest,
                'is_compliant': is_compliant,
                'is_authorized': is_authorized
            })
        
        # Stage 3: Update tracker untuk smooth results
        tracked_results = self.tracker.update(raw_results)
        
        # Track FPS
        elapsed = time.time() - start_time
        self.fps_tracker.append(elapsed)
        
        return tracked_results, len(helmets) > 0, len(vests) > 0
    
    def draw_results(self, frame, results, any_helmet, any_vest):
        """Draw all detections and results on frame"""
        
        # Draw each person
        for res in results:
            x1, y1, x2, y2 = res['bbox']
            name = res['name']
            is_compliant = res['is_compliant']
            has_helmet = res['has_helmet']
            has_vest = res['has_vest']
            similarity = res['similarity']
            
            # Color coding
            if is_compliant:
                color = (0, 255, 0)  # Green - full compliance
                status = "SAFE"
            elif has_helmet or has_vest:
                color = (0, 165, 255)  # Orange - partial compliance
                status = "INCOMPLETE"
            else:
                color = (0, 0, 255)  # Red - no compliance
                status = "VIOLATION"
            
            # Draw person box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            
            # Prepare label
            if name == "Unknown":
                label = f"{status} | Unknown Person"
            else:
                label = f"{status} | {name.upper()}"
            
            equipment = f"{'Helmet' if has_helmet else 'NO Helmet'} | {'Vest' if has_vest else 'NO Vest'}"
            
            # Draw labels with background
            y_offset = y1 - 40
            for text in [label, equipment]:
                (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame, (x1, y_offset-th-5), (x1+tw+10, y_offset+5), color, -1)
                cv2.putText(frame, text, (x1+5, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                y_offset += th + 10
            
            # Save violation
            if not is_compliant and self.cfg.SAVE_VIOLATIONS:
                self.save_violation(frame, res)
        
        # Draw global status icons (top-left)
        ix, iy = 10, 10
        if any_helmet and self.helmet_icon is not None:
            self.paste_icon(frame, self.helmet_icon, ix, iy)
            cv2.putText(frame, "Helmet Detected", (ix + self.cfg.ICON_SIZE + 8, iy + 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            iy += self.helmet_icon.shape[0] + 10
        
        if any_vest and self.vest_icon is not None:
            self.paste_icon(frame, self.vest_icon, ix, iy)
            cv2.putText(frame, "Vest Detected", (ix + self.cfg.ICON_SIZE + 8, iy + 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Draw FPS and stats
        if self.cfg.SHOW_FPS and self.fps_tracker:
            avg_time = sum(self.fps_tracker) / len(self.fps_tracker)
            fps = 1.0 / avg_time if avg_time > 0 else 0
            stats = f"FPS: {fps:.1f} | Persons: {len(results)} | DB: {len(self.face_recognizer.authorized_embeddings)}"
            cv2.putText(frame, stats, (10, frame.shape[0] - 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return frame
    
    def save_violation(self, frame, result):
        """Save frame when violation detected"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = result['name'].replace(" ", "_")
        helmet = "NoHelmet" if not result['has_helmet'] else "Helmet"
        vest = "NoVest" if not result['has_vest'] else "Vest"
        
        filename = f"{self.cfg.VIOLATION_DIR}/violation_{name}_{helmet}_{vest}_{timestamp}.jpg"
        cv2.imwrite(filename, frame)
    
    def processing_worker(self):
        """Background thread for frame processing"""
        while self.processing:
            if not self.frame_queue.empty():
                frame = self.frame_queue.get()
                results, any_helmet, any_vest = self.process_frame(frame)
                self.result_queue.put((results, any_helmet, any_vest))
    
    def run_camera(self, source=0):
        """Run live camera with unified detection"""
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print("❌ Cannot open camera")
            return
        
        print("\n🎥 Starting camera...")
        print("📋 Press 'q' to quit")
        print("📋 Press 's' to save current frame")
        print("-" * 60)
        
        # Start processing thread
        if self.cfg.USE_THREADING:
            worker = threading.Thread(target=self.processing_worker, daemon=True)
            worker.start()
        
        frame_count = 0
        last_results = ([], False, False)
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                if self.cfg.USE_THREADING:
                    # Non-blocking: put frame if queue not full
                    if not self.frame_queue.full():
                        self.frame_queue.put(frame.copy())
                    
                    # Get latest result if available
                    if not self.result_queue.empty():
                        last_results = self.result_queue.get()
                    
                    results, any_helmet, any_vest = last_results
                else:
                    # Blocking: process directly
                    results, any_helmet, any_vest = self.process_frame(frame)
                
                # Draw results
                annotated = self.draw_results(frame, results, any_helmet, any_vest)
                
                # Display
                cv2.imshow("BJBS Unified Safety System", annotated)
                
                # Print violations to console
                for res in results:
                    if not res['is_compliant']:
                        name = res['name']
                        issues = []
                        if not res['has_helmet']:
                            issues.append("NO HELMET")
                        if not res['has_vest']:
                            issues.append("NO VEST")
                        print(f"⚠️  VIOLATION: {name} - {', '.join(issues)}")
                
                # Handle key press
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    save_name = f"snapshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                    cv2.imwrite(save_name, annotated)
                    print(f"💾 Saved: {save_name}")
        
        except KeyboardInterrupt:
            print("\n🛑 Interrupted by user")
        
        finally:
            self.processing = False
            cap.release()
            cv2.destroyAllWindows()
            print("✅ Camera stopped")

# ==================== MAIN ====================
def main():
    # Create system
    system = UnifiedSafetySystem()
    
    # Run camera
    system.run_camera(source=0)  # 0 = webcam, or "video.mp4"

if __name__ == "__main__":
    main()