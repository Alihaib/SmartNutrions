from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Tuple
from PIL import Image
import io
import numpy as np
import traceback

# Try to import transformers (optional - for AI features)
try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Transformers not available - AI features (summarize, classify, generate-questions) will be disabled")
    print("Install with: pip install transformers torch")

# Try to import OCR libraries (PaddleOCR preferred, EasyOCR as fallback)
try:
    from paddleocr import PaddleOCR  # type: ignore
    PADDLEOCR_AVAILABLE = True
except ImportError:
    PADDLEOCR_AVAILABLE = False
    print("PaddleOCR not available, will use EasyOCR as fallback")

try:
    import easyocr  # type: ignore
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False
    print("EasyOCR not available")

# Try to import PDF processing library
try:
    import PyPDF2  # type: ignore
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    print("PyPDF2 not available - PDF upload will be disabled")
    print("Install with: pip install PyPDF2")

# Configuration constants
MAX_IMAGE_SIZE_MB = 10
MAX_IMAGE_SIZE_BYTES = MAX_IMAGE_SIZE_MB * 1024 * 1024
MAX_PDF_SIZE_MB = 20
MAX_PDF_SIZE_BYTES = MAX_PDF_SIZE_MB * 1024 * 1024
ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/jpg", "image/png", "image/bmp", "image/tiff"}
ALLOWED_PDF_TYPE = "application/pdf"
MAX_SUMMARY_INPUT_LENGTH = 2000  # Legacy limit for backward compatibility
MAX_CLASSIFY_INPUT_LENGTH = 1000
MAX_QUESTION_SENTENCES = 10
MIN_SENTENCE_LENGTH = 20
MIN_QUESTION_LENGTH = 10
MAX_TOPICS_RETURNED = 5
CLASSIFICATION_THRESHOLD = 0.1
# Summarization parameters - optimized for high quality and fast processing
SUMMARY_MAX_LENGTH = 150  # Maximum summary length for fast processing
SUMMARY_MIN_LENGTH = 50   # Minimum summary length for comprehensive coverage
QUESTION_MAX_LENGTH = 100
QUESTION_MIN_LENGTH = 10
QUESTION_TEMPERATURE = 0.7

# Academic subject labels for classification
ACADEMIC_SUBJECTS = [
    "Mathematics", "Physics", "Chemistry", "Biology",
    "Computer Science", "History", "Literature", "Philosophy",
    "Economics", "Psychology", "Engineering", "Medicine",
    "Law", "Business", "Art", "Geography"
]

app = FastAPI(
    title="AI Study Companion API",
    description="AI-powered study assistant with OCR, summarization, classification, and question generation",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize OCR (lazy initialization)
ocr = None
ocr_type = None  # 'paddleocr' or 'easyocr'

def get_ocr():
    """
    Lazy initialization of OCR engine.
    Tries PaddleOCR first, falls back to EasyOCR if unavailable.
    
    Returns:
        OCR instance (PaddleOCR or EasyOCR)
    
    Raises:
        Exception: If neither OCR library is available
    """
    global ocr, ocr_type
    if ocr is None:
        if PADDLEOCR_AVAILABLE:
            try:
                print("Initializing PaddleOCR...")
                ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=False, show_log=False)
                ocr_type = 'paddleocr'
                print("PaddleOCR initialized successfully!")
                return ocr
            except Exception as e:
                print(f"Error initializing PaddleOCR: {e}")
                print("Falling back to EasyOCR...")
        
        if EASYOCR_AVAILABLE:
            try:
                print("Initializing EasyOCR...")
                ocr = easyocr.Reader(['en'], gpu=False)
                ocr_type = 'easyocr'
                print("EasyOCR initialized successfully!")
                return ocr
            except Exception as e:
                print(f"Error initializing EasyOCR: {e}")
                raise RuntimeError(
                    "Neither PaddleOCR nor EasyOCR could be initialized. "
                    "Please install one of them: pip install easyocr"
                )
        else:
            raise RuntimeError(
                "No OCR library available. Please install: pip install easyocr"
            )
    
    return ocr

# Initialize HuggingFace models
summarizer = None
classifier = None
question_generator = None

def load_models():
    """
    Load HuggingFace transformer models for AI features.
    Models are loaded into memory and reused for subsequent requests.
    """
    global summarizer, classifier, question_generator
    if not TRANSFORMERS_AVAILABLE:
        print("Transformers not available - skipping model loading")
        return
    
    try:
        print("Loading summarization model...")
        summarizer = pipeline(
            "summarization",
            model="facebook/bart-large-cnn",
            device=-1  # CPU mode
        )
        
        print("Loading classification model...")
        classifier = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
            device=-1  # CPU mode
        )
        
        print("Loading question generation model...")
        question_generator = pipeline(
            "text2text-generation",
            model="valhalla/t5-base-qa-qg-hl",
            device=-1  # CPU mode
        )
        print("All models loaded successfully!")
    except Exception as e:
        print(f"Error loading models: {e}")
        print("Models will be loaded on first use")

# Request/Response models
class TextInput(BaseModel):
    text: str

class SummaryResponse(BaseModel):
    summary: str
    confidence: str  # "Low", "Medium", "High"
    keywords: List[str]  # Deprecated - always empty, kept for backward compatibility
    original_length: int
    summary_length: int

class ClassificationResponse(BaseModel):
    topics: List[dict]
    confidence: str  # "Low", "Medium", "High"

class QuestionResponse(BaseModel):
    questions: List[dict]
    confidence: str  # "Low", "Medium", "High"

# Helper functions for confidence estimation and analysis
def estimate_summary_confidence(original_text: str, summary_text: str) -> str:
    """
    Estimate summary quality confidence based on simple heuristics.
    Higher confidence when: good compression ratio, reasonable length, contains key terms.
    """
    if not original_text or not summary_text:
        return "Low"
    
    orig_len = len(original_text.split())
    summ_len = len(summary_text.split())
    
    if orig_len < 50 or summ_len < 20:
        return "Low"
    
    compression_ratio = summ_len / orig_len if orig_len > 0 else 0
    
    # High confidence: Good compression (15-35%) with appropriate length (50-150 words)
    if 0.15 <= compression_ratio <= 0.35 and 50 <= summ_len <= 150:
        return "High"
    # Medium confidence: Acceptable compression (10-50%) with reasonable length (30-200 words)
    elif 0.10 <= compression_ratio <= 0.50 and 30 <= summ_len <= 200:
        return "Medium"
    # Low confidence: Outside optimal ranges
    else:
        return "Low"

def estimate_classification_confidence(topics: List[dict]) -> str:
    """
    Estimate classification reliability based on score distribution.
    Higher confidence when: clear top score, good score separation, multiple topics.
    """
    if not topics or len(topics) == 0:
        return "Low"
    
    if len(topics) == 1:
        return "Low"
    
    top_score = topics[0].get("score", 0.0)
    second_score = topics[1].get("score", 0.0) if len(topics) > 1 else 0.0
    
    if top_score > 0.7 and (top_score - second_score) > 0.2:
        return "High"
    elif top_score > 0.5 and (top_score - second_score) > 0.1:
        return "Medium"
    else:
        return "Low"

def estimate_question_confidence(questions: List[dict], original_text: str) -> str:
    """
    Estimate question generation reliability.
    Higher confidence when: multiple questions generated, good question length, diverse questions.
    """
    if not questions or len(questions) == 0:
        return "Low"
    
    if len(questions) >= 3:
        avg_length = sum(len(q.get("question", "").split()) for q in questions) / len(questions)
        if 8 <= avg_length <= 20:
            return "High"
        elif 5 <= avg_length <= 25:
            return "Medium"
    
    return "Low" if len(questions) < 2 else "Medium"

def classify_question_type(question: str) -> str:
    """
    Classify question as Factual, Conceptual, or Analytical based on keywords.
    Simple heuristic-based classification.
    """
    question_lower = question.lower()
    
    # Analytical indicators
    analytical_keywords = ["why", "how", "analyze", "compare", "evaluate", "explain", "discuss", "interpret"]
    if any(keyword in question_lower for keyword in analytical_keywords):
        return "Analytical"
    
    # Conceptual indicators
    conceptual_keywords = ["what is", "define", "describe", "concept", "principle", "theory"]
    if any(keyword in question_lower for keyword in conceptual_keywords):
        return "Conceptual"
    
    # Default to Factual
    return "Factual"

def extract_keywords(text: str, max_keywords: int = 10) -> List[str]:
    """
    Extract key terms from text using simple frequency-based approach.
    Filters out common stop words and returns most frequent meaningful terms.
    """
    import re
    from collections import Counter
    
    # Simple stop words list
    stop_words = {
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
        "of", "with", "by", "from", "as", "is", "was", "are", "were", "be",
        "been", "being", "have", "has", "had", "do", "does", "did", "will",
        "would", "could", "should", "may", "might", "must", "can", "this",
        "that", "these", "those", "i", "you", "he", "she", "it", "we", "they"
    }
    
    # Extract words (alphanumeric, at least 3 characters)
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    
    # Filter stop words and count
    meaningful_words = [w for w in words if w not in stop_words]
    word_freq = Counter(meaningful_words)
    
    # Return top keywords
    keywords = [word for word, _ in word_freq.most_common(max_keywords)]
    return keywords

@app.on_event("startup")
async def startup_event():
    print("=" * 50)
    print("Starting AI Study Companion API...")
    print("=" * 50)
    
    # Test OCR initialization
    try:
        print("Testing OCR initialization...")
        get_ocr()
        print(f"✓ OCR ready (using {ocr_type})")
    except Exception as e:
        print(f"⚠ Warning: OCR initialization failed: {e}")
        print("OCR endpoint may not work until this is resolved.")
        print("Please install: pip install easyocr")
    
    # Load HuggingFace models
    load_models()
    print("=" * 50)

@app.post("/ocr")
async def ocr_image(file: UploadFile = File(...)):
    """
    Extract text from uploaded image using OCR.
    
    Args:
        file: Image file (PNG, JPG, JPEG, etc.)
    
    Returns:
        JSON with extracted text or error message
    """
    try:
        contents = await file.read()
        if not contents:
            return {"error": "Empty file uploaded", "text": ""}
        
        if len(contents) > MAX_IMAGE_SIZE_BYTES:
            return {
                "error": f"Image file too large. Maximum size: {MAX_IMAGE_SIZE_MB}MB",
                "text": ""
            }
        
        image = Image.open(io.BytesIO(contents))
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        img_array = np.array(image)
        ocr_instance = get_ocr()
        
        print(f"Processing image: {file.filename}, size: {image.size}, OCR type: {ocr_type}")
        text = ""
        
        if ocr_type == 'paddleocr':
            result = ocr_instance.ocr(img_array, cls=True)
            if result and result[0]:
                for line in result[0]:
                    if line and len(line) > 1:
                        if isinstance(line[1], tuple) and len(line[1]) > 0:
                            text += line[1][0] + "\n"
                        elif isinstance(line[1], str):
                            text += line[1] + "\n"
        elif ocr_type == 'easyocr':
            result = ocr_instance.readtext(img_array)
            if result:
                for detection in result:
                    if len(detection) > 1:
                        text += detection[1] + "\n"
        
        extracted_text = text.strip()
        if not extracted_text:
            return {"text": "", "warning": "No text detected in image"}
        
        return {"text": extracted_text}
    except Exception as e:
        error_details = traceback.format_exc()
        print(f"OCR Error: {error_details}")
        return {"error": str(e), "text": ""}

@app.post("/summarize", response_model=SummaryResponse)
async def summarize_text(input_data: TextInput):
    """
    Generate a high-quality, accurate summary of the input text using BART model.
    Optimized for reliability and fast processing with manageable summary lengths.
    
    Features:
    - Concise summaries (50-150 words) for fast processing
    - High accuracy and reliability through deterministic processing
    - Confidence scoring based on summary quality (compression ratio and length)
    - Dynamic length adjustment based on input size for optimal quality
    - Focus on preserving the most important points of the text
    
    Args:
        input_data: TextInput containing text to summarize
    
    Returns:
        SummaryResponse with generated summary, confidence, keywords, and metadata
    """
    if not TRANSFORMERS_AVAILABLE:
        return SummaryResponse(
            summary="Error: Transformers library not installed. Install with: pip install transformers torch",
            confidence="Low",
            keywords=[],
            original_length=0,
            summary_length=0
        )
    
    if not input_data.text or not input_data.text.strip():
        return SummaryResponse(
            summary="Error: Empty text provided",
            confidence="Low",
            keywords=[],
            original_length=0,
            summary_length=0
        )
    
    try:
        if summarizer is None:
            load_models()
        
        if summarizer is None:
            return SummaryResponse(
                summary="Error: Could not load summarization model",
                confidence="Low",
                keywords=[],
                original_length=0,
                summary_length=0
            )
        
        # Get original text and word count
        original_text = input_data.text.strip()
        word_count = len(original_text.split())
        original_length = word_count
        
        # Truncate text if too long to prevent token overflow (BART has ~1024 token limit)
        # Roughly 2000 characters or ~400 words is safe for direct processing
        text_to_process = original_text
        if len(text_to_process) > MAX_SUMMARY_INPUT_LENGTH:
            text_to_process = text_to_process[:MAX_SUMMARY_INPUT_LENGTH].strip()
            if not text_to_process:
                return SummaryResponse(
                    summary="Error: Text too short after truncation",
                    confidence="Low",
                    keywords=[],
                    original_length=original_length,
                    summary_length=0
                )
        
        # Direct summarization - optimized for high quality and accuracy
        # Using do_sample=False ensures deterministic, reliable summaries
        # Adjust length parameters based on input size for optimal quality
        input_words = len(text_to_process.split())
        
        # Dynamically adjust summary length for better quality
        # Longer inputs get longer summaries (up to max), shorter inputs get proportional summaries
        if input_words > 300:
            # For longer texts, use full range for comprehensive summary
            target_max = SUMMARY_MAX_LENGTH
            target_min = SUMMARY_MIN_LENGTH
        elif input_words > 150:
            # For medium texts, use proportional length
            target_max = min(SUMMARY_MAX_LENGTH, max(SUMMARY_MIN_LENGTH, input_words // 3))
            target_min = SUMMARY_MIN_LENGTH
        else:
            # For shorter texts, use minimum length
            target_max = SUMMARY_MAX_LENGTH
            target_min = max(30, min(SUMMARY_MIN_LENGTH, input_words // 3))
        
        summary_result = summarizer(
            text_to_process,
            max_length=target_max,
            min_length=target_min,
            do_sample=False  # Deterministic output for reliability
        )
        summary_text = summary_result[0]['summary_text'].strip()
        
        # Calculate confidence based on summary quality
        # Confidence reflects how well the summary captures important information
        # Focus on compression ratio and summary length for reliability assessment
        confidence = estimate_summary_confidence(original_text, summary_text)
        
        # Adjust confidence if text was too short (short texts produce less reliable summaries)
        if word_count < 100 and confidence == "High":
            confidence = "Medium"  # Lower confidence for very short texts
        
        return SummaryResponse(
            summary=summary_text,
            confidence=confidence,
            keywords=[],  # Key terms not extracted - focus on summary text only
            original_length=original_length,
            summary_length=len(summary_text.split())
        )
    except Exception as e:
        print(f"Summarization error: {e}")
        return SummaryResponse(
            summary=f"Error: {str(e)}",
            confidence="Low",
            keywords=[],
            original_length=0,
            summary_length=0
        )

@app.post("/classify", response_model=ClassificationResponse)
async def classify_topics(input_data: TextInput):
    """
    Classify text into academic topics using zero-shot classification.
    
    Args:
        input_data: TextInput containing text to classify
    
    Returns:
        ClassificationResponse with top matching topics and confidence scores
    """
    if not TRANSFORMERS_AVAILABLE:
        return ClassificationResponse(
            topics=[{
                "topic": "Error",
                "score": 0.0,
                "error": "Transformers library not installed. Install with: pip install transformers torch"
            }],
            confidence="Low"
        )
    
    if not input_data.text or not input_data.text.strip():
        return ClassificationResponse(
            topics=[{"topic": "Error", "score": 0.0, "error": "Empty text provided"}],
            confidence="Low"
        )
    
    try:
        if classifier is None:
            load_models()
        
        if classifier is None:
            return ClassificationResponse(
                topics=[{"topic": "Error", "score": 0.0, "error": "Could not load classification model"}],
                confidence="Low"
            )
        
        text = input_data.text[:MAX_CLASSIFY_INPUT_LENGTH].strip()
        if not text:
            return ClassificationResponse(
                topics=[{"topic": "Error", "score": 0.0, "error": "Text too short after truncation"}]
            )
        
        result = classifier(text, ACADEMIC_SUBJECTS, multi_label=True)
        
        topics = [
            {"topic": label, "score": float(score)}
            for label, score in zip(result["labels"], result["scores"])
            if score > CLASSIFICATION_THRESHOLD
        ]
        
        topics.sort(key=lambda x: x["score"], reverse=True)
        
        top_topics = topics[:MAX_TOPICS_RETURNED]
        confidence = estimate_classification_confidence(top_topics)
        
        return ClassificationResponse(
            topics=top_topics,
            confidence=confidence
        )
    except Exception as e:
        print(f"Classification error: {e}")
        return ClassificationResponse(
            topics=[{"topic": "Error", "score": 0.0, "error": str(e)}],
            confidence="Low"
        )

@app.post("/generate-questions", response_model=QuestionResponse)
async def generate_questions(input_data: TextInput):
    """
    Generate practice questions from input text using T5 model.
    
    Args:
        input_data: TextInput containing text to generate questions from
    
    Returns:
        QuestionResponse with generated questions and context
    """
    if not TRANSFORMERS_AVAILABLE:
        return QuestionResponse(
            questions=[{
                "question": "Error: Transformers library not installed. Install with: pip install transformers torch",
                "context": "",
                "type": "Factual"
            }],
            confidence="Low"
        )
    
    if not input_data.text or not input_data.text.strip():
        return QuestionResponse(
            questions=[{"question": "Error: Empty text provided", "context": "", "type": "Factual"}],
            confidence="Low"
        )
    
    try:
        if question_generator is None:
            load_models()
        
        if question_generator is None:
            return QuestionResponse(
                questions=[{"question": "Error: Could not load question generation model", "context": "", "type": "Factual"}],
                confidence="Low"
            )
        
        text = input_data.text.strip()
        sentences = text.split('.')
        questions = []
        
        for sentence in sentences[:MAX_QUESTION_SENTENCES]:
            sentence = sentence.strip()
            if len(sentence) > MIN_SENTENCE_LENGTH:
                try:
                    prompt = f"generate question: {sentence}"
                    result = question_generator(
                        prompt,
                        max_length=QUESTION_MAX_LENGTH,
                        min_length=QUESTION_MIN_LENGTH,
                        num_return_sequences=1,
                        do_sample=True,
                        temperature=QUESTION_TEMPERATURE
                    )
                    
                    if result and len(result) > 0:
                        question_text = result[0]['generated_text'].strip()
                        question_text = question_text.replace("question:", "").strip()
                        
                    if (question_text and
                        len(question_text) > MIN_QUESTION_LENGTH and
                        not question_text.startswith("generate")):
                            question_type = classify_question_type(question_text)
                            questions.append({
                                "question": question_text,
                                "context": sentence[:200],
                                "type": question_type
                            })
                except Exception as e:
                    print(f"Error generating question from sentence: {e}")
                    continue
        
        if not questions:
            questions.append({
                "question": "What are the main points discussed in this text?",
                "context": text[:200],
                "type": "Analytical"
            })
        
        final_questions = questions[:MAX_TOPICS_RETURNED]
        confidence = estimate_question_confidence(final_questions, text)
        
        return QuestionResponse(
            questions=final_questions,
            confidence=confidence
        )
    except Exception as e:
        print(f"Question generation error: {e}")
        return QuestionResponse(
            questions=[{"question": f"Error generating questions: {str(e)}", "context": "", "type": "Factual"}],
            confidence="Low"
        )

@app.get("/")
async def root():
    """API root endpoint."""
    return {"message": "AI Study Companion API", "version": "1.0.0"}

def validate_file_size(file_size: int, max_size: int, file_type: str) -> Tuple[bool, str]:
    """
    Validate file size against maximum allowed size.
    
    Args:
        file_size: Size of file in bytes
        max_size: Maximum allowed size in bytes
        file_type: Type of file for error message
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    if file_size > max_size:
        max_mb = max_size / (1024 * 1024)
        return False, f"{file_type} file too large. Maximum size: {max_mb}MB"
    return True, ""

def validate_file_type(content_type: str, allowed_types: set) -> Tuple[bool, str]:
    """
    Validate file MIME type.
    
    Args:
        content_type: MIME type of uploaded file
        allowed_types: Set of allowed MIME types
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    if content_type not in allowed_types:
        return False, f"Invalid file type. Allowed types: {', '.join(allowed_types)}"
    return True, ""

def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """
    Extract text from PDF file bytes.
    
    Args:
        pdf_bytes: PDF file content as bytes
    
    Returns:
        Extracted text from all pages
    """
    if not PDF_AVAILABLE:
        raise RuntimeError("PDF processing not available. Install PyPDF2: pip install PyPDF2")
    
    text = ""
    try:
        pdf_file = io.BytesIO(pdf_bytes)
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        
        for page_num, page in enumerate(pdf_reader.pages):
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        
        return text.strip()
    except Exception as e:
        raise RuntimeError(f"Error extracting text from PDF: {str(e)}")

@app.post("/upload-image")
async def upload_image(file: UploadFile = File(...)):
    """
    Upload and process image file using OCR.
    Alternative endpoint to /ocr with enhanced validation.
    
    Args:
        file: Image file (JPG, PNG, etc.)
    
    Returns:
        JSON with extracted text or error message
    """
    try:
        # Validate file type
        if file.content_type not in ALLOWED_IMAGE_TYPES:
            return {
                "error": f"Invalid file type. Allowed types: {', '.join(ALLOWED_IMAGE_TYPES)}",
                "text": ""
            }
        
        # Read and validate file size
        contents = await file.read()
        is_valid, error_msg = validate_file_size(len(contents), MAX_IMAGE_SIZE_BYTES, "Image")
        if not is_valid:
            return {"error": error_msg, "text": ""}
        
        if not contents:
            return {"error": "Empty file uploaded", "text": ""}
        
        # Process image with OCR (reuse existing OCR logic)
        image = Image.open(io.BytesIO(contents))
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        img_array = np.array(image)
        ocr_instance = get_ocr()
        
        print(f"Processing image: {file.filename}, size: {image.size}, OCR type: {ocr_type}")
        text = ""
        
        if ocr_type == 'paddleocr':
            result = ocr_instance.ocr(img_array, cls=True)
            if result and result[0]:
                for line in result[0]:
                    if line and len(line) > 1:
                        if isinstance(line[1], tuple) and len(line[1]) > 0:
                            text += line[1][0] + "\n"
                        elif isinstance(line[1], str):
                            text += line[1] + "\n"
        elif ocr_type == 'easyocr':
            result = ocr_instance.readtext(img_array)
            if result:
                for detection in result:
                    if len(detection) > 1:
                        text += detection[1] + "\n"
        
        extracted_text = text.strip()
        if not extracted_text:
            return {"text": "", "warning": "No text detected in image"}
        
        return {"text": extracted_text, "filename": file.filename}
    except Exception as e:
        error_details = traceback.format_exc()
        print(f"Image upload error: {error_details}")
        return {"error": str(e), "text": ""}

@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Upload and extract text from PDF file.
    Extracts text from all pages and returns it for further processing.
    
    Args:
        file: PDF file
    
    Returns:
        JSON with extracted text from all pages
    """
    try:
        # Validate file type
        if file.content_type != ALLOWED_PDF_TYPE:
            return {
                "error": f"Invalid file type. Expected: {ALLOWED_PDF_TYPE}",
                "text": ""
            }
        
        # Read and validate file size
        contents = await file.read()
        is_valid, error_msg = validate_file_size(len(contents), MAX_PDF_SIZE_BYTES, "PDF")
        if not is_valid:
            return {"error": error_msg, "text": ""}
        
        if not contents:
            return {"error": "Empty file uploaded", "text": ""}
        
        # Extract text from PDF
        extracted_text = extract_text_from_pdf(contents)
        
        if not extracted_text or not extracted_text.strip():
            return {
                "text": "",
                "warning": "No text could be extracted from PDF. The PDF may be image-based or empty."
            }
        
        # Limit text length for display (full text available for processing)
        display_text = extracted_text[:MAX_SUMMARY_INPUT_LENGTH] if len(extracted_text) > MAX_SUMMARY_INPUT_LENGTH else extracted_text
        
        return {
            "text": extracted_text,
            "display_text": display_text,
            "filename": file.filename,
            "total_length": len(extracted_text),
            "pages_processed": extracted_text.count('\n\n') + 1  # Rough estimate
        }
    except RuntimeError as e:
        return {"error": str(e), "text": ""}
    except Exception as e:
        error_details = traceback.format_exc()
        print(f"PDF upload error: {error_details}")
        return {"error": f"Error processing PDF: {str(e)}", "text": ""}

@app.get("/health")
async def health_check():
    """
    Health check endpoint to verify API and model status.
    
    Returns:
        JSON with API status and model loading state
    """
    return {
        "status": "healthy",
        "ocr_available": ocr is not None,
        "ocr_type": ocr_type,
        "pdf_available": PDF_AVAILABLE,
        "transformers_available": TRANSFORMERS_AVAILABLE,
        "models_loaded": {
            "summarizer": summarizer is not None,
            "classifier": classifier is not None,
            "question_generator": question_generator is not None
        },
        "file_limits": {
            "max_image_size_mb": MAX_IMAGE_SIZE_MB,
            "max_pdf_size_mb": MAX_PDF_SIZE_MB
        }
    }

# TODO: Future Enhancements
# 1. Study tips generation based on text analysis
#    - Analyze text structure and content
#    - Generate personalized study recommendations
#    - Suggest study techniques based on topic type
#
# 2. Multi-language support
#    - Extend OCR to support multiple languages
#    - Add language detection for text input
#    - Support summarization/classification in multiple languages
#    - Use multilingual models (mBERT, mT5)
#
# 3. Caching frequent queries
#    - Implement in-memory cache for repeated inputs
#    - Use hash-based cache keys
#    - Add TTL (time-to-live) for cache entries
#    - Consider Redis for distributed caching
#
# 4. Additional enhancements
#    - Export functionality (PDF, DOCX)
#    - Study session history
#    - Progress tracking
#    - Spaced repetition scheduling
