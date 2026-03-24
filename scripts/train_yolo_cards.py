"""
Train YOLOv8 để nhận diện bài
"""

import os
from pathlib import Path

def main():
    print("🃏 YOLO Card Detector Training")
    print("=" * 50)
    
    # Check ultralytics
    try:
        from ultralytics import YOLO
        print("✅ ultralytics installed")
    except ImportError:
        print("📦 Installing ultralytics...")
        os.system("pip install ultralytics")
        from ultralytics import YOLO
    
    print("\n📋 Options:")
    print("1. Download dataset from Roboflow (Recommended)")
    print("2. Use local dataset")
    print("3. Skip (use Gemini only)")
    
    choice = input("\nSelect (1/2/3): ").strip()
    
    if choice == "1":
        download_and_train()
    elif choice == "2":
        train_local()
    else:
        print("⏭️  Skipping YOLO training. Using Gemini only.")

def download_and_train():
    """Download dataset và train"""
    
    print("\n📥 Downloading dataset from Roboflow...")
    print("\n📝 Steps:")
    print("1. Go to: https://universe.roboflow.com/augmented-startups/playing-cards-ow27d")
    print("2. Click 'Download Dataset'")
    print("3. Select format: YOLOv8")
    print("4. Copy download code")
    print("5. Paste API key when prompted")
    
    api_key = input("\nRoboflow API key (or press Enter to skip): ").strip()
    
    if not api_key:
        print("ℹ️  Manual download:")
        print("   1. Download from Roboflow")
        print("   2. Extract to: data/card_dataset/")
        print("   3. Run option 2")
        return
    
    # Download with roboflow
    try:
        from roboflow import Roboflow
        
        rf = Roboflow(api_key=api_key)
        project = rf.workspace("augmented-startups").project("playing-cards-ow27d")
        dataset = project.version(4).download("yolov8", location="data/card_dataset")
        
        print(f"✅ Dataset downloaded to: data/card_dataset/")
        
        # Train
        train_with_dataset("data/card_dataset/data.yaml")
        
    except ImportError:
        print("❌ roboflow not installed")
        print("   Install: pip install roboflow")
    except Exception as e:
        print(f"❌ Download failed: {e}")

def train_local():
    """Train với dataset local"""
    
    data_yaml = input("Path to data.yaml (default: data/card_dataset/data.yaml): ").strip()
    if not data_yaml:
        data_yaml = "data/card_dataset/data.yaml"
    
    if not Path(data_yaml).exists():
        print(f"❌ File not found: {data_yaml}")
        print("\nCreate dataset structure:")
        print("data/card_dataset/")
        print("├── data.yaml")
        print("├── train/")
        print("│   ├── images/")
        print("│   └── labels/")
        print("└── valid/")
        print("    ├── images/")
        print("    └── labels/")
        return
    
    train_with_dataset(data_yaml)

def train_with_dataset(data_yaml: str):
    """Train model"""
    from ultralytics import YOLO
    
    print(f"\n🎯 Training with: {data_yaml}")
    
    epochs = input("Epochs (default: 50): ").strip()
    epochs = int(epochs) if epochs else 50
    
    imgsz = input("Image size (default: 640): ").strip()
    imgsz = int(imgsz) if imgsz else 640
    
    print(f"\n🚀 Starting training...")
    print(f"   Epochs: {epochs}")
    print(f"   Image size: {imgsz}")
    print(f"   Model: YOLOv8n (nano - fastest)")
    
    # Load pretrained model
    model = YOLO("yolov8n.pt")
    
    # Train
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=16,
        project="models/card_detector",
        name="train",
        exist_ok=True,
        verbose=True,
        device="auto",  # Auto-detect GPU/CPU
    )
    
    # Copy best model
    best_path = Path("models/card_detector/train/weights/best.pt")
    final_path = Path("models/card_detector/best.pt")
    
    if best_path.exists():
        import shutil
        final_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(best_path, final_path)
        
        print(f"\n✅ Training complete!")
        print(f"📁 Model saved to: {final_path}")
        print(f"\n🔧 Update config/ai_config.yaml:")
        print(f"   yolo:")
        print(f"     enabled: true")
        
        # Test model
        test = input("\nTest model now? (y/n): ").strip().lower()
        if test == 'y':
            test_model(str(final_path))
    else:
        print("❌ Training failed - best.pt not found")

def test_model(model_path: str):
    """Test trained model"""
    from ultralytics import YOLO
    import sys
    
    model = YOLO(model_path)
    
    img_path = input("Path to test image: ").strip()
    
    if not Path(img_path).exists():
        print(f"❌ Image not found: {img_path}")
        return
    
    print(f"\n🧪 Testing model...")
    
    results = model(img_path, conf=0.7)
    
    for result in results:
        print(f"\n📊 Detected {len(result.boxes)} cards:")
        
        if result.boxes:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                name = result.names.get(cls_id, "Unknown")
                print(f"  - {name} ({conf:.2%})")
        
        # Save result
        output = "test_result.jpg"
        result.save(filename=output)
        print(f"\n💾 Saved visualization: {output}")

if __name__ == "__main__":
    main()