@echo off
echo Installing essential dependencies...
call venv\Scripts\activate.bat

echo Step 1: Installing FastAPI and core dependencies...
pip install fastapi==0.104.1
pip install uvicorn[standard]==0.24.0
pip install python-multipart==0.0.6
pip install pydantic==2.5.0

echo Step 2: Installing image processing...
pip install pillow==10.1.0
pip install numpy==1.24.3

echo Step 3: Installing AI models (this may take a while)...
pip install transformers==4.35.0
pip install torch==2.1.0
pip install accelerate==0.24.1
pip install sentencepiece==0.1.99
pip install protobuf==3.20.3

echo Step 4: Installing OCR (EasyOCR for Windows)...
pip install easyocr

echo.
echo Essential installation complete!
echo You can now start the server with: uvicorn main:app --reload --port 8000
pause








