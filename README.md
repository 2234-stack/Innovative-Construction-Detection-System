# Innovative-Construction-Detection-System
Construction monitoring system - predicting problems before they happen and catching invisible defects in real-time.

\\

"""
============
Storage Inventory Comparison System - PHOTO VERSION
============
Detects missing and added objects between two photos of storage areas
Sends clear messages about exactly what changed and how many
"""

import numpy as np
import cv2
import os
from collections import defaultdict
import time

# Use the existing COCO_CLASSES from your code
COCO_CLASSES = ("person", "bicycle", "car", "motorcycle", "airplane", "bus",
        "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign",
        "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
        "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
        "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", 
        "kite", "baseball bat", "baseball glove", "skateboard",
        "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
        "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
        "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
        "couch", "potted plant", "bed", "dining table", "toilet", "tv",
        "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
        "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
        "scissors", "teddy bear", "hair drier", "toothbrush",)

class StorageInventoryAnalyzer:
    """
    Advanced storage inventory comparison system for PHOTOS
    Compares two photos and detects missing/added objects with clear messages
    """
    
    def __init__(self):
        self.inventory_original = defaultdict(list)  # Objects in original photo
        self.inventory_current = defaultdict(list)   # Objects in current photo
        self.missing_objects = []
        self.added_objects = []
        
        # Tracking parameters
        self.iou_threshold = 0.5  # IoU threshold for object matching
        
    def calculate_iou(self, box1, box2):
        """Calculate Intersection over Union between two boxes"""
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        # Calculate coordinates
        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)
        
        # Calculate area of intersection
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        
        # Calculate union area
        box1_area = w1 * h1
        box2_area = w2 * h2
        union_area = box1_area + box2_area - inter_area
        
        # Avoid division by zero
        iou = inter_area / union_area if union_area > 0 else 0
        return iou
    
    def analyze_photo_inventory(self, image_path, inventory_dict, display_name="Analysis"):
        """Analyze a single photo and build inventory of objects"""
        print(f"🔍 Analyzing photo: {image_path}")
        
        # Load image
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"❌ Error: Could not load image {image_path}")
            return []
        
        # Get detections using your YOLOv8 code
        detections = self.get_detections(frame)
        
        # Convert to our object format
        objects = []
        for i, detection in enumerate(detections):
            obj = {
                'bbox': detection['bbox'],
                'class_id': detection['class_id'],
                'class_name': detection['class'],
                'score': detection['score'],
                'object_id': i
            }
            objects.append(obj)
            
            # Add to inventory dictionary
            inventory_dict[detection['class']].append(obj)
        
        # Visualize detections
        display_frame = self.visualize_detections(frame, objects, display_name)
        cv2.imshow(display_name, display_frame)
        cv2.waitKey(1000)  # Show for 1 second
        cv2.destroyAllWindows()
        
        print(f"✅ Found {len(objects)} objects in {image_path}")
        return objects
    
    def compare_photos(self, original_photo, current_photo):
        """Compare two photos and detect missing/added objects"""
        print("🔍 COMPARING PHOTOS...")
        
        # Analyze both photos
        original_objects = self.analyze_photo_inventory(original_photo, self.inventory_original, "Original Photo")
        current_objects = self.analyze_photo_inventory(current_photo, self.inventory_current, "Current Photo")
        
        # Reset previous results
        self.missing_objects = []
        self.added_objects = []
        
        # Find missing objects (in original but not in current)
        for orig_obj in original_objects:
            found_match = False
            for curr_obj in current_objects:
                if (orig_obj['class_name'] == curr_obj['class_name'] and 
                    self.calculate_iou(orig_obj['bbox'], curr_obj['bbox']) > self.iou_threshold):
                    found_match = True
                    break
            
            if not found_match:
                self.missing_objects.append(orig_obj)
        
        # Find added objects (in current but not in original)
        for curr_obj in current_objects:
            found_match = False
            for orig_obj in original_objects:
                if (curr_obj['class_name'] == orig_obj['class_name'] and 
                    self.calculate_iou(curr_obj['bbox'], orig_obj['bbox']) > self.iou_threshold):
                    found_match = True
                    break
            
            if not found_match:
                self.added_objects.append(curr_obj)
        
        return self.missing_objects, self.added_objects
    
    def send_inventory_alerts(self):
        """Send clear messages about what's missing and added"""
        print("\n" + "🚨" * 20)
        print("🚨 INVENTORY CHANGE ALERT")
        print("🚨" * 20)
        
        # Count objects by type
        missing_by_type = defaultdict(int)
        added_by_type = defaultdict(int)
        
        for obj in self.missing_objects:
            missing_by_type[obj['class_name']] += 1
            
        for obj in self.added_objects:
            added_by_type[obj['class_name']] += 1
        
        # Send missing objects alert
        if self.missing_objects:
            print(f"\n❌ MISSING ITEMS ALERT:")
            print(f"   Total missing: {len(self.missing_objects)} objects")
            
            for obj_type, count in missing_by_type.items():
                print(f"   └── {count} {obj_type}(s) are missing")
                
            # List specific missing items
            print(f"\n   📋 Missing details:")
            for i, obj in enumerate(self.missing_objects[:5]):  # Show first 5
                bbox = obj['bbox']
                print(f"      {i+1}. {obj['class_name']} (Position: [{bbox[0]}, {bbox[1]}])")
            
            if len(self.missing_objects) > 5:
                print(f"      ... and {len(self.missing_objects) - 5} more items")
                
        else:
            print(f"\n✅ NO ITEMS MISSING - Everything is accounted for!")
        
        # Send added objects alert  
        if self.added_objects:
            print(f"\n✅ NEW ITEMS ALERT:")
            print(f"   Total added: {len(self.added_objects)} objects")
            
            for obj_type, count in added_by_type.items():
                print(f"   └── {count} {obj_type}(s) were added")
                
            # List specific added items
            print(f"\n   📋 Added details:")
            for i, obj in enumerate(self.added_objects[:5]):  # Show first 5
                bbox = obj['bbox']
                print(f"      {i+1}. {obj['class_name']} (Position: [{bbox[0]}, {bbox[1]}])")
            
            if len(self.added_objects) > 5:
                print(f"      ... and {len(self.added_objects) - 5} more items")
                
        else:
            print(f"\n📝 NO NEW ITEMS ADDED - Inventory unchanged")
        
        # Summary message
        total_changes = len(self.missing_objects) + len(self.added_objects)
        if total_changes > 0:
            print(f"\n📊 SUMMARY: {total_changes} changes detected in storage inventory")
            print(f"   └── {len(self.missing_objects)} items missing")
            print(f"   └── {len(self.added_objects)} items added")
        else:
            print(f"\n📊 SUMMARY: No changes detected - inventory is identical")
    
    def create_comparison_photo(self, original_photo_path, current_photo_path, output_path="inventory_comparison.jpg"):
        """Create a side-by-side comparison photo with annotations"""
        print(f"\n🎨 Creating comparison photo...")
        
        # Load both photos
        original_img = cv2.imread(original_photo_path)
        current_img = cv2.imread(current_photo_path)
        
        if original_img is None or current_img is None:
            print("❌ Error: Could not load one or both photos")
            return
        
        # Resize to same height for comparison
        height = min(original_img.shape[0], current_img.shape[0])
        original_img = cv2.resize(original_img, (int(original_img.shape[1] * height / original_img.shape[0]), height))
        current_img = cv2.resize(current_img, (int(current_img.shape[1] * height / current_img.shape[0]), height))
        
        # Draw missing objects on original photo (RED)
        for obj in self.missing_objects:
            bbox = obj['bbox']
            # Scale bbox if needed
            scale_x = original_img.shape[1] / 640  # Assuming original detection was on 640x640
            scale_y = original_img.shape[0] / 640
            scaled_bbox = [int(bbox[0] * scale_x), int(bbox[1] * scale_y), 
                          int(bbox[2] * scale_x), int(bbox[3] * scale_y)]
            
            cv2.rectangle(original_img, 
                         (scaled_bbox[0], scaled_bbox[1]), 
                         (scaled_bbox[0] + scaled_bbox[2], scaled_bbox[1] + scaled_bbox[3]), 
                         (0, 0, 255), 3)
            cv2.putText(original_img, f"MISSING: {obj['class_name']}", 
                       (scaled_bbox[0], scaled_bbox[1] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Draw added objects on current photo (GREEN)
        for obj in self.added_objects:
            bbox = obj['bbox']
            # Scale bbox if needed
            scale_x = current_img.shape[1] / 640
            scale_y = current_img.shape[0] / 640
            scaled_bbox = [int(bbox[0] * scale_x), int(bbox[1] * scale_y), 
                          int(bbox[2] * scale_x), int(bbox[3] * scale_y)]
            
            cv2.rectangle(current_img, 
                         (scaled_bbox[0], scaled_bbox[1]), 
                         (scaled_bbox[0] + scaled_bbox[2], scaled_bbox[1] + scaled_bbox[3]), 
                         (0, 255, 0), 3)
            cv2.putText(current_img, f"ADDED: {obj['class_name']}", 
                       (scaled_bbox[0], scaled_bbox[1] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Combine side by side
        combined = np.hstack([original_img, current_img])
        
        # Add headers
        header_height = 50
        header = np.zeros((header_height, combined.shape[1], 3), dtype=np.uint8)
        
        cv2.putText(header, "ORIGINAL INVENTORY (Missing in RED)", 
                   (50, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(header, "CURRENT INVENTORY (Added in GREEN)", 
                   (original_img.shape[1] + 50, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        combined_with_header = np.vstack([header, combined])
        
        # Add summary footer
        footer_height = 40
        footer = np.zeros((footer_height, combined.shape[1], 3), dtype=np.uint8)
        summary_text = f"Missing: {len(self.missing_objects)} items | Added: {len(self.added_objects)} items"
        cv2.putText(footer, summary_text, 
                   (combined.shape[1]//2 - 200, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        final_image = np.vstack([combined_with_header, footer])
        
        # Save the comparison image
        cv2.imwrite(output_path, final_image)
        print(f"✅ Comparison photo saved: {output_path}")
        
        # Display the result
        cv2.imshow('Inventory Comparison', final_image)
        cv2.waitKey(3000)  # Show for 3 seconds
        cv2.destroyAllWindows()
    
    def get_detections(self, frame):
        """PLACEHOLDER - Replace with your actual YOLOv8 detection"""
        # This is where you integrate with your YOLOv8 code
        # For now, return empty list - replace with actual detection
        
        # Example of how to integrate:
        # yolo = YoloV8()
        # blob = yolo.preprocess(frame)
        # output = model.inference(blob)  # Your MemryX inference
        # detections = yolo.postprocess(output)
        
        # For testing, let's create some dummy detections
        detections = []
        height, width = frame.shape[:2]
        
        # Add some test detections (remove this in production)
        test_objects = [
            {'bbox': [100, 100, 80, 80], 'class_id': 39, 'class': 'bottle', 'score': 0.9},
            {'bbox': [300, 200, 100, 120], 'class_id': 56, 'class': 'chair', 'score': 0.8},
            {'bbox': [500, 150, 90, 90], 'class_id': 67, 'class': 'cell phone', 'score': 0.7},
        ]
        
        return test_objects  # Replace with actual detections

    def visualize_detections(self, frame, objects, title="Detections"):
        """Visualize detections on frame"""
        display = frame.copy()
        
        for obj in objects:
            bbox = obj['bbox']
            cv2.rectangle(display, (bbox[0], bbox[1]), 
                         (bbox[0] + bbox[2], bbox[1] + bbox[3]), (0, 255, 0), 2)
            cv2.putText(display, f"{obj['class_name']} ({obj['score']:.2f})", 
                       (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        cv2.putText(display, f"Objects: {len(objects)}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return display

# Your existing YoloV8 class (keep it exactly as is)
class YoloV8:
    """
    A helper class to run YOLOv8 pre- and post-proccessing.
    """
    def __init__(self, stream_img_size=None, model_type='tflite'):
        self.name = 'YoloV8'
        self.input_size = (640,640,3) 
        self.input_width = 640
        self.input_height = 640
        self.confidence_thres = 0.4
        self.iou_thres = 0.6
        self.model_type = model_type

        self.stream_mode = False
        if stream_img_size:
            # Pre-calculate ratio/pad values for preprocessing
            self.preprocess(np.zeros(stream_img_size))
            self.stream_mode = True

    def preprocess(self, img):
        self.original_imgage = img
        [self.img_height, self.img_width, _] = self.original_imgage.shape
        
        self.length = max((self.img_height, self.img_width))
        self.image = np.zeros((self.length, self.length, 3), np.uint8)
        self.image[0:self.img_height, 0:self.img_width] = self.original_imgage

        scale = self.length / 640
        blob = cv2.dnn.blobFromImage(self.image, scalefactor=1 / 255, size=(640, 640), swapRB=True)

        if self.model_type == 'tflite':
            blob = blob.transpose(0, 2, 3, 1)

        return blob

    def postprocess(self, output):
        outputs = np.transpose(np.squeeze(output[0]))
        x_factor = self.length / self.input_width
        y_factor = self.length / self.input_height

        boxes = outputs[:, :4]
        class_scores = outputs[:, 4:]

        max_scores = np.max(class_scores, axis=1)
        class_ids = np.argmax(class_scores, axis=1)

        valid_indices = np.where(max_scores >= self.confidence_thres)[0]
        if len(valid_indices) == 0:
            return []

        valid_boxes = boxes[valid_indices]
        valid_class_ids = class_ids[valid_indices]
        valid_scores = max_scores[valid_indices]

        valid_boxes[:, 0] = (valid_boxes[:, 0] - valid_boxes[:, 2] / 2) * x_factor
        valid_boxes[:, 1] = (valid_boxes[:, 1] - valid_boxes[:, 3] / 2) * y_factor
        valid_boxes[:, 2] = valid_boxes[:, 2] * x_factor
        valid_boxes[:, 3] = valid_boxes[:, 3] * y_factor

        detections = [{
            'bbox': valid_boxes[i].astype(int).tolist(),
            'class_id': int(valid_class_ids[i]),
            'class': COCO_CLASSES[int(valid_class_ids[i])],
            'score': valid_scores[i]
        } for i in range(len(valid_indices))]

        if len(detections) > 0:
            boxes_for_nms = [d['bbox'] for d in detections]
            scores_for_nms = [d['score'] for d in detections]

            indices = cv2.dnn.NMSBoxes(boxes_for_nms, scores_for_nms, self.confidence_thres, self.iou_thres)

            if len(indices) > 0:
                if isinstance(indices[0], list) or isinstance(indices[0], np.ndarray):
                    indices = [i[0] for i in indices]
                final_detections = [detections[i] for i in indices]
            else:
                final_detections = []
        else:
            final_detections = []

        return final_detections

# Main execution
def main():
    print("🚀 NEOM Storage Inventory Comparison - PHOTO SYSTEM")
    print("=" * 60)
    
    # Initialize analyzer
    analyzer = StorageInventoryAnalyzer()
    
    # Define your photo paths
    original_photo = "storage_original.jpg"  # Replace with your original photo path
    current_photo = "storage_current.jpg"    # Replace with your current photo path
    
    # Compare the two photos
    missing, added = analyzer.compare_photos(original_photo, current_photo)
    
    # Send clear alerts about what changed
    analyzer.send_inventory_alerts()
    
    # Create comparison photo
    analyzer.create_comparison_photo(original_photo, current_photo, "inventory_comparison.jpg")
    
    print("\n🎉 PHOTO ANALYSIS COMPLETE!")
    print("📸 Check 'inventory_comparison.jpg' for visual results")

if __name__ == "__main__":
    main()
```
