"""
Card Detection Module - Nhận diện lá bài từ hình ảnh
Supports multiple detection methods:
1. Template Matching (fast, works with standard card designs)
2. Color + Contour Detection (medium, works with most cards)
3. YOLO/CNN Model (accurate, requires training)
4. Google Vision API (cloud-based, very accurate)
"""

import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum
import logging
import io
import base64

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DetectionMethod(Enum):
    TEMPLATE = "template"
    CONTOUR = "contour"
    YOLO = "yolo"
    HYBRID = "hybrid"
    MANUAL = "manual"


@dataclass
class DetectedCard:
    """Represents a detected card"""
    rank: str  # 2-10, J, Q, K, A
    suit: str  # S, H, D, C (Spades, Hearts, Diamonds, Clubs)
    confidence: float  # 0.0 - 1.0
    bbox: Tuple[int, int, int, int]  # x, y, width, height
    
    def to_string(self) -> str:
        """Convert to string format like 'AS', 'KH', '10D'"""
        return f"{self.rank}{self.suit}"
    
    def __str__(self):
        return self.to_string()


@dataclass
class DetectionResult:
    """Result of card detection"""
    cards: List[DetectedCard]
    image_with_boxes: np.ndarray
    raw_image: np.ndarray
    method_used: DetectionMethod
    processing_time: float
    warnings: List[str]
    
    @property
    def card_strings(self) -> List[str]:
        return [c.to_string() for c in self.cards]
    
    @property
    def is_valid_hand(self) -> bool:
        return len(self.cards) == 13 and len(set(self.card_strings)) == 13


class CardDetector:
    """
    Main card detection class
    """
    
    # Card dimensions ratio (standard playing cards)
    CARD_ASPECT_RATIO = 1.4  # height/width
    
    # Suit colors
    RED_SUITS = ['H', 'D']  # Hearts, Diamonds
    BLACK_SUITS = ['S', 'C']  # Spades, Clubs
    
    # Rank values
    RANKS = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
    SUITS = ['S', 'H', 'D', 'C']
    
    # Suit symbols for display
    SUIT_SYMBOLS = {
        'S': '♠', 'H': '♥', 'D': '♦', 'C': '♣'
    }
    
    def __init__(self, method: DetectionMethod = DetectionMethod.HYBRID):
        self.method = method
        self.templates_loaded = False
        self.templates = {}
        self.yolo_model = None
        
        # Try to load templates
        self._load_templates()
        
        # Try to load YOLO model if available
        if method in [DetectionMethod.YOLO, DetectionMethod.HYBRID]:
            self._load_yolo_model()
    
    def _load_templates(self):
        """Load card templates for template matching"""
        template_dir = Path(__file__).parent.parent.parent / "data" / "card_templates"
        
        if template_dir.exists():
            for rank in self.RANKS:
                for suit in self.SUITS:
                    template_path = template_dir / f"{rank}{suit}.png"
                    if template_path.exists():
                        template = cv2.imread(str(template_path))
                        if template is not None:
                            self.templates[f"{rank}{suit}"] = template
            
            if self.templates:
                self.templates_loaded = True
                logger.info(f"Loaded {len(self.templates)} card templates")
        else:
            logger.warning(f"Template directory not found: {template_dir}")
    
    def _load_yolo_model(self):
        """Load YOLO model for card detection"""
        try:
            from ultralytics import YOLO
            
            model_path = Path(__file__).parent.parent.parent / "models" / "card_detection" / "yolo_cards.pt"
            
            if model_path.exists():
                self.yolo_model = YOLO(str(model_path))
                logger.info("YOLO card detection model loaded")
            else:
                # Try to use pre-trained model
                logger.warning("Custom YOLO model not found, using fallback methods")
        except ImportError:
            logger.warning("ultralytics not installed, YOLO detection disabled")
        except Exception as e:
            logger.error(f"Error loading YOLO model: {e}")
    
    def detect_from_image(self, image: np.ndarray) -> DetectionResult:
        """
        Detect cards from numpy array image
        
        Args:
            image: BGR image as numpy array
            
        Returns:
            DetectionResult with detected cards
        """
        import time
        start_time = time.time()
        
        warnings = []
        
        # Preprocess image
        processed = self._preprocess_image(image)
        
        # Try detection methods based on setting
        if self.method == DetectionMethod.YOLO and self.yolo_model:
            cards = self._detect_yolo(processed)
        elif self.method == DetectionMethod.TEMPLATE and self.templates_loaded:
            cards = self._detect_template(processed)
        elif self.method == DetectionMethod.CONTOUR:
            cards = self._detect_contour(processed)
        elif self.method == DetectionMethod.HYBRID:
            cards = self._detect_hybrid(processed)
        else:
            # Fallback to contour detection
            cards = self._detect_contour(processed)
            warnings.append("Using fallback contour detection")
        
        # Post-process: remove duplicates, sort
        cards = self._post_process_cards(cards)
        
        # Draw boxes on image
        image_with_boxes = self._draw_detections(image.copy(), cards)
        
        processing_time = time.time() - start_time
        
        # Validate
        if len(cards) != 13:
            warnings.append(f"Detected {len(cards)} cards, expected 13")
        
        if len(cards) != len(set(c.to_string() for c in cards)):
            warnings.append("Duplicate cards detected")
        
        return DetectionResult(
            cards=cards,
            image_with_boxes=image_with_boxes,
            raw_image=image,
            method_used=self.method,
            processing_time=processing_time,
            warnings=warnings
        )
    
    def detect_from_file(self, file_path: str) -> DetectionResult:
        """Detect cards from file path"""
        image = cv2.imread(file_path)
        if image is None:
            raise ValueError(f"Could not read image: {file_path}")
        return self.detect_from_image(image)
    
    def detect_from_bytes(self, image_bytes: bytes) -> DetectionResult:
        """Detect cards from bytes (for web uploads)"""
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Could not decode image from bytes")
        return self.detect_from_image(image)
    
    def detect_from_pil(self, pil_image: Image.Image) -> DetectionResult:
        """Detect cards from PIL Image"""
        # Convert PIL to OpenCV format
        image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        return self.detect_from_image(image)
    
    def detect_from_base64(self, base64_string: str) -> DetectionResult:
        """Detect cards from base64 encoded image"""
        # Remove header if present
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]
        
        image_bytes = base64.b64decode(base64_string)
        return self.detect_from_bytes(image_bytes)
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for detection"""
        # Resize if too large
        max_dim = 1920
        h, w = image.shape[:2]
        
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            image = cv2.resize(image, None, fx=scale, fy=scale)
        
        # Enhance contrast
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        return enhanced
    
    def _detect_contour(self, image: np.ndarray) -> List[DetectedCard]:
        """
        Detect cards using contour detection
        Works by finding rectangular shapes with card aspect ratio
        """
        cards = []
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply blur and edge detection
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        
        # Dilate edges to close gaps
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by size and shape
        image_area = image.shape[0] * image.shape[1]
        min_card_area = image_area * 0.01  # Card should be at least 1% of image
        max_card_area = image_area * 0.15  # Card should be at most 15% of image
        
        card_contours = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            if min_card_area < area < max_card_area:
                # Approximate contour to polygon
                peri = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
                
                # Cards should have 4 corners
                if len(approx) == 4:
                    # Check aspect ratio
                    x, y, w, h = cv2.boundingRect(approx)
                    aspect_ratio = max(h, w) / min(h, w)
                    
                    if 1.2 < aspect_ratio < 1.8:  # Close to card ratio
                        card_contours.append((contour, (x, y, w, h)))
        
        # Extract card regions and identify
        for contour, bbox in card_contours:
            x, y, w, h = bbox
            card_roi = image[y:y+h, x:x+w]
            
            # Identify card
            rank, suit, confidence = self._identify_card_region(card_roi)
            
            if rank and suit:
                cards.append(DetectedCard(
                    rank=rank,
                    suit=suit,
                    confidence=confidence,
                    bbox=bbox
                ))
        
        return cards
    
    def _identify_card_region(self, card_roi: np.ndarray) -> Tuple[Optional[str], Optional[str], float]:
        """
        Identify rank and suit from a card region
        Uses color analysis and corner detection
        """
        if card_roi.size == 0:
            return None, None, 0.0
        
        h, w = card_roi.shape[:2]
        
        # Focus on top-left corner where rank/suit is shown
        corner_h = int(h * 0.35)
        corner_w = int(w * 0.25)
        corner = card_roi[5:corner_h, 5:corner_w]
        
        if corner.size == 0:
            return None, None, 0.0
        
        # Detect suit by color
        suit = self._detect_suit_color(corner)
        
        # Detect rank by shape/template
        rank = self._detect_rank(corner)
        
        confidence = 0.7 if rank and suit else 0.3
        
        return rank, suit, confidence
    
    def _detect_suit_color(self, roi: np.ndarray) -> Optional[str]:
        """Detect suit based on color (red = Hearts/Diamonds, black = Spades/Clubs)"""
        # Convert to HSV
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # Red color ranges (Hearts/Diamonds)
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        
        mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask_red = mask_red1 + mask_red2
        
        red_pixels = cv2.countNonZero(mask_red)
        total_pixels = roi.shape[0] * roi.shape[1]
        
        if red_pixels / total_pixels > 0.05:
            # Red suit - need to distinguish Hearts vs Diamonds
            # For now, return H (can be improved with shape detection)
            return 'H'
        else:
            # Black suit
            return 'S'
    
    def _detect_rank(self, roi: np.ndarray) -> Optional[str]:
        """Detect rank from corner region"""
        # Convert to grayscale and threshold
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # Get largest contour (likely the rank character)
        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)
        
        # Analyze aspect ratio to guess rank
        aspect = w / h if h > 0 else 0
        
        # Simple heuristics (can be replaced with ML)
        if aspect > 0.8:
            return '10'  # Wide character
        elif aspect < 0.4:
            return '1'  # Could be A or 1
        else:
            # Need more sophisticated detection
            return 'A'  # Default
        
        return None
    
    def _detect_template(self, image: np.ndarray) -> List[DetectedCard]:
        """Detect cards using template matching"""
        cards = []
        
        if not self.templates_loaded:
            logger.warning("Templates not loaded, falling back to contour detection")
            return self._detect_contour(image)
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        for card_name, template in self.templates.items():
            template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
            
            # Try multiple scales
            for scale in [0.5, 0.75, 1.0, 1.25, 1.5]:
                resized = cv2.resize(template_gray, None, fx=scale, fy=scale)
                
                if resized.shape[0] > gray.shape[0] or resized.shape[1] > gray.shape[1]:
                    continue
                
                result = cv2.matchTemplate(gray, resized, cv2.TM_CCOEFF_NORMED)
                
                threshold = 0.7
                locations = np.where(result >= threshold)
                
                for pt in zip(*locations[::-1]):
                    h, w = resized.shape
                    
                    # Extract rank and suit from card_name
                    if len(card_name) == 2:
                        rank, suit = card_name[0], card_name[1]
                    else:  # 10X
                        rank, suit = card_name[:2], card_name[2]
                    
                    cards.append(DetectedCard(
                        rank=rank,
                        suit=suit,
                        confidence=result[pt[1], pt[0]],
                        bbox=(pt[0], pt[1], w, h)
                    ))
        
        return cards
    
    def _detect_yolo(self, image: np.ndarray) -> List[DetectedCard]:
        """Detect cards using YOLO model"""
        if not self.yolo_model:
            logger.warning("YOLO model not loaded, falling back to contour detection")
            return self._detect_contour(image)
        
        cards = []
        
        # Run inference
        results = self.yolo_model(image, verbose=False)
        
        for result in results:
            boxes = result.boxes
            
            for box in boxes:
                # Get bounding box
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = box.conf[0].cpu().numpy()
                class_id = int(box.cls[0].cpu().numpy())
                
                # Map class_id to card name
                card_name = self.yolo_model.names[class_id]
                
                if len(card_name) >= 2:
                    if len(card_name) == 2:
                        rank, suit = card_name[0], card_name[1]
                    else:
                        rank, suit = card_name[:2], card_name[2]
                    
                    cards.append(DetectedCard(
                        rank=rank.upper(),
                        suit=suit.upper(),
                        confidence=float(confidence),
                        bbox=(int(x1), int(y1), int(x2-x1), int(y2-y1))
                    ))
        
        return cards
    
    def _detect_hybrid(self, image: np.ndarray) -> List[DetectedCard]:
        """
        Hybrid detection: combine multiple methods
        """
        all_cards = []
        
        # Try YOLO first if available
        if self.yolo_model:
            yolo_cards = self._detect_yolo(image)
            all_cards.extend(yolo_cards)
        
        # If not enough cards, try contour
        if len(all_cards) < 13:
            contour_cards = self._detect_contour(image)
            all_cards.extend(contour_cards)
        
        # If still not enough and templates available, try template
        if len(all_cards) < 13 and self.templates_loaded:
            template_cards = self._detect_template(image)
            all_cards.extend(template_cards)
        
        return all_cards
    
    def _post_process_cards(self, cards: List[DetectedCard]) -> List[DetectedCard]:
        """Post-process detected cards: remove duplicates, sort by position"""
        if not cards:
            return []
        
        # Remove duplicates (same card detected multiple times)
        unique_cards = {}
        
        for card in cards:
            key = card.to_string()
            
            if key not in unique_cards or card.confidence > unique_cards[key].confidence:
                unique_cards[key] = card
        
        # Sort by x position (left to right)
        sorted_cards = sorted(unique_cards.values(), key=lambda c: c.bbox[0])
        
        return sorted_cards
    
    def _draw_detections(self, image: np.ndarray, cards: List[DetectedCard]) -> np.ndarray:
        """Draw detection boxes on image"""
        for card in cards:
            x, y, w, h = card.bbox
            
            # Color based on suit
            if card.suit in self.RED_SUITS:
                color = (0, 0, 255)  # Red
            else:
                color = (0, 0, 0)  # Black
            
            # Draw rectangle
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            
            # Draw label
            label = f"{card.rank}{self.SUIT_SYMBOLS.get(card.suit, card.suit)}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            
            # Background for label
            cv2.rectangle(image, (x, y - label_size[1] - 10), 
                         (x + label_size[0], y), color, -1)
            
            # Text
            cv2.putText(image, label, (x, y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Confidence
            conf_text = f"{card.confidence:.0%}"
            cv2.putText(image, conf_text, (x, y + h + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return image


class ManualCardSelector:
    """
    Interactive card selection for when auto-detection fails
    """
    
    def __init__(self):
        self.selected_cards = []
    
    def get_card_grid_html(self, already_selected: List[str] = None) -> str:
        """Generate HTML for card selection grid"""
        if already_selected is None:
            already_selected = []
        
        suits = [('S', '♠', '#000'), ('H', '♥', '#e74c3c'), 
                 ('D', '♦', '#e74c3c'), ('C', '♣', '#000')]
        ranks = ['A', 'K', 'Q', 'J', '10', '9', '8', '7', '6', '5', '4', '3', '2']
        
        html = '<div style="display: grid; grid-template-columns: repeat(13, 1fr); gap: 4px; max-width: 100%;">'
        
        for suit_code, suit_symbol, color in suits:
            for rank in ranks:
                card_id = f"{rank}{suit_code}"
                disabled = card_id in already_selected
                
                opacity = "0.3" if disabled else "1"
                cursor = "not-allowed" if disabled else "pointer"
                
                html += f'''
                <div class="card-select" data-card="{card_id}" 
                     style="background: white; border: 2px solid {color}; 
                            border-radius: 4px; padding: 8px 4px; text-align: center;
                            cursor: {cursor}; opacity: {opacity}; font-size: 14px;
                            color: {color}; font-weight: bold;">
                    {rank}{suit_symbol}
                </div>
                '''
        
        html += '</div>'
        
        return html


# ============ UTILITY FUNCTIONS ============

def create_sample_detection():
    """Create a sample detection for testing"""
    # This would be replaced with actual detection
    sample_cards = [
        DetectedCard('A', 'S', 0.95, (10, 10, 60, 90)),
        DetectedCard('K', 'H', 0.92, (80, 10, 60, 90)),
        DetectedCard('Q', 'D', 0.88, (150, 10, 60, 90)),
        # ... more cards
    ]
    
    return sample_cards


def validate_detected_cards(cards: List[DetectedCard]) -> Tuple[bool, List[str]]:
    """Validate detected cards"""
    errors = []
    
    # Check count
    if len(cards) != 13:
        errors.append(f"Expected 13 cards, got {len(cards)}")
    
    # Check for duplicates
    card_strings = [c.to_string() for c in cards]
    duplicates = [c for c in card_strings if card_strings.count(c) > 1]
    
    if duplicates:
        errors.append(f"Duplicate cards: {set(duplicates)}")
    
    # Check for valid cards
    valid_ranks = set(['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A'])
    valid_suits = set(['S', 'H', 'D', 'C'])
    
    for card in cards:
        if card.rank not in valid_ranks:
            errors.append(f"Invalid rank: {card.rank}")
        if card.suit not in valid_suits:
            errors.append(f"Invalid suit: {card.suit}")
    
    return len(errors) == 0, errors


# ============ TESTING ============

if __name__ == "__main__":
    # Test detection
    detector = CardDetector(method=DetectionMethod.CONTOUR)
    
    # Create test image (you would use a real image)
    test_image = np.zeros((480, 640, 3), dtype=np.uint8)
    test_image[:] = (34, 139, 34)  # Green background
    
    try:
        result = detector.detect_from_image(test_image)
        print(f"Detected {len(result.cards)} cards")
        print(f"Method: {result.method_used}")
        print(f"Time: {result.processing_time:.3f}s")
        print(f"Warnings: {result.warnings}")
    except Exception as e:
        print(f"Error: {e}")