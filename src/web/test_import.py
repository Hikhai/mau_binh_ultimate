"""Test import"""
import sys
import os

print("=" * 50)
print("Testing imports...")
print("=" * 50)

# Current directory
print(f"\nCurrent dir: {os.getcwd()}")
print(f"Python path: {sys.path[:3]}...")

# Check file exists
components_dir = os.path.join(os.getcwd(), "components")
print(f"\nComponents dir: {components_dir}")
print(f"Exists: {os.path.exists(components_dir)}")

if os.path.exists(components_dir):
    print(f"Files: {os.listdir(components_dir)}")

# Try import
print("\n" + "=" * 50)
print("Trying imports...")
print("=" * 50)

try:
    from components.image_input import image_input_component
    print("✅ image_input imported OK!")
except ImportError as e:
    print(f"❌ ImportError: {e}")
except Exception as e:
    print(f"❌ Other error: {type(e).__name__}: {e}")

# Try importing card_detector_ai
print("\n" + "=" * 50)
print("Testing card_detector_ai...")
print("=" * 50)

ml_dir = os.path.join(os.path.dirname(os.getcwd()), "ml")
print(f"ML dir: {ml_dir}")
print(f"Exists: {os.path.exists(ml_dir)}")

if os.path.exists(ml_dir):
    print(f"Files: {os.listdir(ml_dir)}")
    
    sys.path.insert(0, ml_dir)
    
    try:
        from card_detector_ai import AICardDetector
        print("✅ card_detector_ai imported OK!")
    except ImportError as e:
        print(f"❌ ImportError: {e}")
    except Exception as e:
        print(f"❌ Other error: {type(e).__name__}: {e}")

print("\n" + "=" * 50)
print("Done!")