"""
Simple test script to verify OCR installation and functionality
Tests both PaddleOCR and EasyOCR
"""
import sys
from PIL import Image
import numpy as np

def test_ocr():
    """Test OCR with PaddleOCR first, fallback to EasyOCR"""
    # Try PaddleOCR first
    try:
        print("Testing PaddleOCR installation...")
        from paddleocr import PaddleOCR
        
        print("Initializing PaddleOCR...")
        ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=False, show_log=False)
        print("✓ PaddleOCR initialized successfully!")
        ocr_type = 'paddleocr'
    except ImportError:
        print("PaddleOCR not available, trying EasyOCR...")
        try:
            import easyocr
            print("Initializing EasyOCR...")
            ocr = easyocr.Reader(['en'], gpu=False)
            print("✓ EasyOCR initialized successfully!")
            ocr_type = 'easyocr'
        except ImportError:
            print("✗ Neither PaddleOCR nor EasyOCR is installed")
            print("Please install: pip install easyocr")
            return False
    except Exception as e:
        print(f"✗ Error initializing PaddleOCR: {e}")
        print("Trying EasyOCR as fallback...")
        try:
            import easyocr
            ocr = easyocr.Reader(['en'], gpu=False)
            ocr_type = 'easyocr'
        except Exception as e2:
            print(f"✗ Error initializing EasyOCR: {e2}")
            return False
    
    # Create a simple test image with text
    print("\nCreating test image...")
    img = Image.new('RGB', (200, 50), color='white')
    from PIL import ImageDraw, ImageFont
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default()
    except:
        font = None
    draw.text((10, 10), "Test OCR", fill='black', font=font)
    
    print(f"Running OCR on test image (using {ocr_type})...")
    img_array = np.array(img)
    
    try:
        if ocr_type == 'paddleocr':
            result = ocr.ocr(img_array, cls=True)
            if result and result[0]:
                print("✓ OCR test successful!")
                detected_text = result[0][0][1][0] if result[0] else "No text"
                print(f"Detected text: {detected_text}")
            else:
                print("⚠ OCR returned no results")
        else:  # easyocr
            result = ocr.readtext(img_array)
            if result:
                print("✓ OCR test successful!")
                detected_text = result[0][1] if result else "No text"
                print(f"Detected text: {detected_text}")
            else:
                print("⚠ OCR returned no results")
        
        return True
    except Exception as e:
        print(f"✗ Error testing OCR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_ocr()
    sys.exit(0 if success else 1)

