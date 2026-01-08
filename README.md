<<<<<<< HEAD


# 🥗 Smart Nutrition – AI-Powered Food Recognition App

Smart Nutrition is an AI-driven nutrition analysis application built with **React** and **Firebase**.  
The system allows users to upload a food image, recognizes the food using an **AI Vision model**, and instantly provides nutritional data such as calories, protein, carbs, and fat.  
Meals are saved securely to each user’s account, allowing daily and weekly nutrition tracking.

---

## 🚀 Features

### 🧠 AI Food Recognition
- Upload a photo of any meal  
- AI Vision (OpenAI / HuggingFace) identifies the food item  
- Fast and accurate predictions  

### 🍎 Nutrition Analysis
- Automatic calculation of:
  - Calories  
  - Protein  
  - Carbohydrates  
  - Fats  
- Clean, user-friendly nutrition table  

### 🔐 Firebase Authentication
- Register, Login, Logout  
- Password reset  
- Secure session handling  

### ☁️ Cloud Storage & Database
- Firebase Firestore stores:
  - Food name  
  - Nutrition values  
  - Image URL  
  - Timestamp  
- Firebase Storage stores uploaded images  

### 📊 User Dashboard
- Total calories for today  
- Weekly nutrition statistics  
- Interactive charts and graphs  
- Full meal history with photos  

### 🎨 Modern & Responsive UI
- Clean design  
- Mobile-friendly  
- Dark mode support  

---

## 🏗️ Tech Stack

| Layer | Technology |
|-------|-------------|
| **Frontend** | React (Vite), JavaScript, Tailwind/MUI |
| **Backend** | Firebase Auth, Firestore, Firebase Storage |
| **AI Model** | OpenAI Vision API / HuggingFace |
| **Charts** | Recharts / Chart.js |

---

## 📂 Project Structure
=======
# AI Study Companion

An intelligent web application that helps students study by extracting text from images, summarizing content, classifying topics, and generating practice questions using AI.

## Features

- 📸 **Image OCR**: Upload images of notes or exams and extract text using EasyOCR/PaddleOCR
- 📝 **Text Summarization**: Get concise summaries (50-150 words) using HuggingFace BART
- 🏷️ **Topic Classification**: Automatically classify content into 16 academic subjects
- ❓ **Question Generation**: Generate practice questions from study materials using T5
- 🎨 **Modern UI**: Beautiful, responsive React interface with drag-and-drop

## Tech Stack

- **Frontend**: React 19 with modern UI/UX
- **Backend**: FastAPI (Python 3.8+)
- **OCR**: EasyOCR (primary), PaddleOCR (fallback)
- **AI Models**: HuggingFace Transformers
  - BART (facebook/bart-large-cnn) for summarization
  - BART-MNLI (facebook/bart-large-mnli) for zero-shot classification
  - T5 (valhalla/t5-base-qa-qg-hl) for question generation

## Setup Instructions

### Prerequisites

- Python 3.8 or higher
- Node.js 16+ and npm
- 4GB+ RAM (for AI models)
- 5GB+ free disk space (for model downloads)

### Backend Setup

1. Navigate to the backend directory:
```bash
cd backend
```

2. Create and activate virtual environment:
   - Windows:
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```
   - Linux/Mac:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Install OCR library (Windows-compatible):
```bash
pip install easyocr
```

5. Start the FastAPI server:
```bash
uvicorn main:app --reload --port 8000
```

The API will be available at `http://localhost:8000`

**First Run Notes**:
- HuggingFace models download automatically (~4.5GB total)
- Download time: 5-15 minutes depending on internet speed
- Models are cached locally for subsequent runs
- OCR models download on first use (~500MB)

### Frontend Setup

1. Navigate to the frontend directory:
```bash
cd frontend
```

2. Install dependencies:
```bash
npm install
```

3. Start the development server:
```bash
npm start
```

The application will open at `http://localhost:3000`

## API Endpoints

### Core Endpoints

- **`POST /ocr`** - Extract text from uploaded images
  - Input: Image file (PNG, JPG, JPEG, max 10MB)
  - Output: `{"text": "extracted text"}` or `{"error": "message"}`
  
- **`POST /summarize`** - Generate concise summary
  - Input: `{"text": "your text here"}` (max 2000 chars)
  - Output: `{"summary": "generated summary"}` (50-150 words)
  
- **`POST /classify`** - Classify text into academic topics
  - Input: `{"text": "your text here"}` (max 1000 chars)
  - Output: `{"topics": [{"topic": "Mathematics", "score": 0.85}, ...]}` (top 5)
  
- **`POST /generate-questions`** - Generate practice questions
  - Input: `{"text": "your text here"}`
  - Output: `{"questions": [{"question": "...", "context": "..."}, ...]}` (up to 5)

### Utility Endpoints

- **`GET /`** - API information
- **`GET /health`** - Health check and model status

### Model Sizes & Limitations

| Feature | Model | Size | Input Limit | Processing Time |
|---------|-------|------|-------------|-----------------|
| OCR | EasyOCR | ~500MB | 10MB image | 2-5 seconds |
| Summarization | BART-Large-CNN | ~1.6GB | 2000 chars | 3-8 seconds |
| Classification | BART-MNLI | ~1.6GB | 1000 chars | 2-5 seconds |
| Question Gen | T5-Base | ~850MB | Variable | 5-15 seconds |

**Total Download Size**: ~4.5GB (one-time, first run)

**Limitations**:
- All models run on CPU (GPU support requires CUDA setup)
- Text inputs are truncated to model limits
- Processing time increases with input length
- Memory usage: ~2-3GB RAM when all models loaded

## Usage

1. **Upload an Image**: Click the upload button and select an image of your notes or exam
2. **Or Paste Text**: Type or paste your lecture text directly into the text area
3. **Process Text**: Use the action buttons to:
   - Summarize the content
   - Classify topics
   - Generate practice questions
   - Or click "Process All" to do everything at once

## Project Structure

```
study-companion/
├── backend/
│   ├── main.py              # FastAPI application
│   ├── requirements.txt     # Python dependencies
│   └── venv/                # Virtual environment
├── frontend/
│   ├── src/
│   │   ├── App.js           # Main React component
│   │   └── App.css          # Styles
│   └── package.json         # Node dependencies
└── README.md
```

## Usage Tips

- **Image Quality**: Clear, well-lit images with readable text work best
- **Text Input**: Longer texts are automatically truncated to model limits
- **Processing**: First request per model may be slower (model initialization)
- **Memory**: Ensure sufficient RAM (4GB+ recommended) when all models are loaded
- **CPU Mode**: All models run on CPU by default for compatibility

## Troubleshooting

### Common Issues

1. **"No OCR library available"**
   - Solution: `pip install easyocr`

2. **"Transformers not available"**
   - Solution: `pip install transformers torch`

3. **Models fail to load**
   - Check internet connection (first-time download required)
   - Verify sufficient disk space (~5GB)
   - Check Python version (3.8+ required)

4. **Out of memory errors**
   - Close other applications
   - Process one feature at a time
   - Consider using GPU if available

### Performance Optimization

- Models stay in memory after first load (faster subsequent requests)
- Use `/health` endpoint to check model status
- For production, consider GPU acceleration or model quantization

## Project Structure

```
study-companion/
├── backend/
│   ├── main.py              # FastAPI application with AI endpoints
│   ├── requirements.txt     # Python dependencies
│   ├── test_ocr.py          # OCR testing utility
│   └── venv/                # Virtual environment (gitignored)
├── frontend/
│   ├── src/
│   │   ├── App.js           # Main React component
│   │   ├── App.css          # Styles
│   │   └── index.js         # React entry point
│   └── package.json         # Node dependencies
├── AI_MODELS_EXPLANATION.md  # Detailed AI models documentation
├── README.md                 # This file
└── .gitignore               # Git ignore rules
```

## License

MIT

## Acknowledgments

- HuggingFace for transformer models
- EasyOCR for OCR capabilities
- FastAPI and React communities





>>>>>>> 94ee531 (initial commit)
