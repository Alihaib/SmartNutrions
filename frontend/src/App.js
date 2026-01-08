import React, { useState, useRef, useCallback } from 'react';
import axios from 'axios';
import './App.css';

const API_BASE_URL = 'http://localhost:8000';

function App() {
  const [imageFile, setImageFile] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [pdfFile, setPdfFile] = useState(null);
  const [uploadType, setUploadType] = useState('image'); // 'image' or 'pdf'
  const [textInput, setTextInput] = useState('');
  const [ocrText, setOcrText] = useState('');
  const [summary, setSummary] = useState('');
  const [summaryData, setSummaryData] = useState(null); // Store full summary response
  const [topics, setTopics] = useState([]);
  const [topicsConfidence, setTopicsConfidence] = useState(null);
  const [questions, setQuestions] = useState([]);
  const [questionsConfidence, setQuestionsConfidence] = useState(null);
  const [showOriginalText, setShowOriginalText] = useState(false);
  const [loading, setLoading] = useState({
    ocr: false,
    summarize: false,
    classify: false,
    generateQuestions: false
  });
  const [toast, setToast] = useState(null);
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef(null);
  const dropZoneRef = useRef(null);

  const showToast = useCallback((message, type = 'info') => {
    setToast({ message, type });
    setTimeout(() => setToast(null), 4000);
  }, []);

  const processPdf = useCallback(async (file) => {
    setLoading(prev => ({ ...prev, ocr: true }));

    try {
      const formData = new FormData();
      formData.append('file', file);

      const response = await axios.post(`${API_BASE_URL}/upload-pdf`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        timeout: 120000, // 2 minutes for PDF processing
      });

      if (response.data.error) {
        showToast(`Error: ${response.data.error}`, 'error');
        return;
      }

      if (response.data.warning) {
        showToast(response.data.warning, 'warning');
      }

      const extractedText = response.data.text || '';
      setOcrText(extractedText);
      setTextInput(extractedText);
      
      if (!extractedText) {
        showToast('No text could be extracted from PDF.', 'warning');
      } else {
        showToast(`PDF processed successfully! Extracted ${response.data.total_length || extractedText.length} characters.`, 'success');
      }
    } catch (error) {
      console.error('PDF Error:', error);
      let errorMessage = 'Error processing PDF. ';
      
      if (error.response) {
        errorMessage += error.response.data?.error || `Server error: ${error.response.status}`;
      } else if (error.request) {
        errorMessage += 'No response from server. Please check if the backend is running.';
      } else {
        errorMessage += error.message;
      }
      
      showToast(errorMessage, 'error');
    } finally {
      setLoading(prev => ({ ...prev, ocr: false }));
    }
  }, [showToast]);

  const processImage = useCallback(async (file) => {
    setLoading(prev => ({ ...prev, ocr: true }));

    try {
      const formData = new FormData();
      formData.append('file', file);

      const response = await axios.post(`${API_BASE_URL}/ocr`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        timeout: 60000,
      });

      if (response.data.error) {
        showToast(`Error: ${response.data.error}`, 'error');
        return;
      }

      if (response.data.warning) {
        showToast(response.data.warning, 'warning');
      }

      const extractedText = response.data.text || '';
      setOcrText(extractedText);
      setTextInput(extractedText);
      
      if (!extractedText) {
        showToast('No text detected in the image. Please try a clearer image.', 'warning');
      } else {
        showToast('Text extracted successfully!', 'success');
      }
    } catch (error) {
      console.error('OCR Error:', error);
      let errorMessage = 'Error processing image. ';
      
      if (error.response) {
        errorMessage += error.response.data?.error || `Server error: ${error.response.status}`;
      } else if (error.request) {
        errorMessage += 'No response from server. Please check if the backend is running.';
      } else {
        errorMessage += error.message;
      }
      
      showToast(errorMessage, 'error');
    } finally {
      setLoading(prev => ({ ...prev, ocr: false }));
    }
  }, [showToast]);

  const handleFileSelect = useCallback((file) => {
    if (!file) return;

    // Determine file type and validate
    if (file.type.startsWith('image/')) {
      if (file.size > 10 * 1024 * 1024) {
        showToast('Image file is too large. Please upload an image smaller than 10MB.', 'error');
        return;
      }
      setUploadType('image');
      setImageFile(file);
      setPdfFile(null);
      
      // Create preview for images
      const reader = new FileReader();
      reader.onloadend = () => {
        setImagePreview(reader.result);
      };
      reader.readAsDataURL(file);

      // Auto-process the image
      processImage(file);
    } else if (file.type === 'application/pdf') {
      if (file.size > 20 * 1024 * 1024) {
        showToast('PDF file is too large. Please upload a PDF smaller than 20MB.', 'error');
        return;
      }
      setUploadType('pdf');
      setPdfFile(file);
      setImageFile(null);
      setImagePreview(null);
      
      // Auto-process the PDF
      processPdf(file);
    } else {
      showToast('Please upload an image file (PNG, JPG, JPEG) or PDF file.', 'error');
      return;
    }
  }, [showToast, processImage, processPdf]);

  const handleDragEnter = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  }, []);

  const handleDragOver = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
  }, []);

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);

    const files = e.dataTransfer.files;
    if (files && files.length > 0) {
      handleFileSelect(files[0]);
    }
  }, [handleFileSelect]);

  const handleFileUpload = (e) => {
    const file = e.target.files[0];
    if (file) {
      handleFileSelect(file);
    }
  };

  const handleSummarize = async () => {
    if (!textInput.trim()) {
      showToast('Please provide text to summarize', 'warning');
      return;
    }

    // Check if text is too short
    const wordCount = textInput.trim().split(/\s+/).length;
    if (wordCount < 150) {
      showToast('For better AI results, please provide a longer or more detailed text (150+ words recommended).', 'info');
    }

    setLoading(prev => ({ ...prev, summarize: true }));

    try {
      const response = await axios.post(`${API_BASE_URL}/summarize`, {
        text: textInput
      });
      setSummary(response.data.summary);
      setSummaryData(response.data); // Store full response with confidence, keywords, etc.
      showToast('Summary generated successfully!', 'success');
    } catch (error) {
      console.error('Summarize Error:', error);
      const errorMsg = error.response?.data?.summary || 'Error generating summary. Please try again.';
      showToast(errorMsg, 'error');
    } finally {
      setLoading(prev => ({ ...prev, summarize: false }));
    }
  };

  const handleClassify = async () => {
    if (!textInput.trim()) {
      showToast('Please provide text to classify', 'warning');
      return;
    }

    setLoading(prev => ({ ...prev, classify: true }));

    try {
      const response = await axios.post(`${API_BASE_URL}/classify`, {
        text: textInput
      });
      setTopics(response.data.topics);
      setTopicsConfidence(response.data.confidence);
      if (response.data.topics.length > 0) {
        showToast('Topics classified successfully!', 'success');
      } else {
        showToast('No topics detected', 'warning');
      }
    } catch (error) {
      console.error('Classify Error:', error);
      showToast('Error classifying topics. Please try again.', 'error');
    } finally {
      setLoading(prev => ({ ...prev, classify: false }));
    }
  };

  const handleGenerateQuestions = async () => {
    if (!textInput.trim()) {
      showToast('Please provide text to generate questions from', 'warning');
      return;
    }

    setLoading(prev => ({ ...prev, generateQuestions: true }));

    try {
      const response = await axios.post(`${API_BASE_URL}/generate-questions`, {
        text: textInput
      });
      setQuestions(response.data.questions);
      setQuestionsConfidence(response.data.confidence);
      if (response.data.questions.length > 0) {
        showToast('Questions generated successfully!', 'success');
      } else {
        showToast('No questions could be generated', 'warning');
      }
    } catch (error) {
      console.error('Generate Questions Error:', error);
      showToast('Error generating questions. Please try again.', 'error');
    } finally {
      setLoading(prev => ({ ...prev, generateQuestions: false }));
    }
  };

  const handleProcessAll = async () => {
    if (!textInput.trim()) {
      showToast('Please provide text first (upload image or paste text)', 'warning');
      return;
    }

    showToast('Processing all features...', 'info');
    await Promise.all([
      handleSummarize(),
      handleClassify(),
      handleGenerateQuestions()
    ]);
  };

  const copyToClipboard = (text) => {
    navigator.clipboard.writeText(text);
    showToast('Copied to clipboard!', 'success');
  };

  const clearAll = () => {
    setImageFile(null);
    setImagePreview(null);
    setPdfFile(null);
    setUploadType('image');
    setTextInput('');
    setOcrText('');
    setSummary('');
    setSummaryData(null);
    setTopics([]);
    setTopicsConfidence(null);
    setQuestions([]);
    setQuestionsConfidence(null);
    setShowOriginalText(false);
    showToast('All data cleared', 'info');
  };

  // Helper function to highlight keywords in text
  const highlightKeywords = (text, keywords) => {
    if (!keywords || keywords.length === 0) return text;
    
    let highlightedText = text;
    keywords.forEach(keyword => {
      const regex = new RegExp(`\\b${keyword}\\b`, 'gi');
      highlightedText = highlightedText.replace(regex, `<mark>${keyword}</mark>`);
    });
    return highlightedText;
  };

  // Helper function to get confidence badge color
  const getConfidenceColor = (confidence) => {
    switch (confidence?.toLowerCase()) {
      case 'high': return '#10b981'; // green
      case 'medium': return '#f59e0b'; // amber
      case 'low': return '#ef4444'; // red
      default: return '#64748b'; // gray
    }
  };

  // Helper function to get confidence badge text
  const getConfidenceBadge = (confidence) => {
    if (!confidence) return null;
    const color = getConfidenceColor(confidence);
    return (
      <span className="confidence-badge" style={{ backgroundColor: color }}>
        {confidence} Confidence
      </span>
    );
  };

  return (
    <div className="App">
      {/* Toast Notification */}
      {toast && (
        <div className={`toast toast-${toast.type}`}>
          <span className="toast-message">{toast.message}</span>
          <button className="toast-close" onClick={() => setToast(null)}>×</button>
        </div>
      )}

      {/* Header */}
      <header className="app-header">
        <div className="header-content">
          <div className="logo">
            <span className="logo-icon">📚</span>
            <h1>AI Study Companion</h1>
          </div>
          <p className="header-subtitle">Transform your notes into study materials with AI</p>
        </div>
      </header>

      <main className="app-main">
        {/* Upload Section */}
        <section className="section upload-section">
          <div className="section-header">
            <h2>
              <span className="icon">📤</span>
              Upload or Paste Content
            </h2>
            {(imageFile || textInput) && (
              <button className="btn-clear" onClick={clearAll}>
                Clear All
              </button>
            )}
          </div>

          <div className="upload-container">
            {/* Drag & Drop Zone */}
            <div
              ref={dropZoneRef}
              className={`drop-zone ${isDragging ? 'dragging' : ''} ${imagePreview ? 'has-image' : ''} ${pdfFile ? 'has-pdf' : ''}`}
              onDragEnter={handleDragEnter}
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
              onClick={() => fileInputRef.current?.click()}
            >
              <input
                ref={fileInputRef}
                type="file"
                accept="image/*,application/pdf"
                onChange={handleFileUpload}
                className="file-input"
              />
              
              {imagePreview ? (
                <div className="image-preview-container">
                  <img src={imagePreview} alt="Preview" className="image-preview" />
                  {loading.ocr && (
                    <div className="loading-overlay">
                      <div className="spinner"></div>
                      <p>Extracting text...</p>
                    </div>
                  )}
                </div>
              ) : pdfFile ? (
                <div className="pdf-preview-container">
                  <div className="pdf-icon">📄</div>
                  <p className="pdf-filename">{pdfFile.name}</p>
                  {loading.ocr && (
                    <div className="loading-overlay">
                      <div className="spinner"></div>
                      <p>Extracting text from PDF...</p>
                    </div>
                  )}
                </div>
              ) : (
                <div className="drop-zone-content">
                  <div className="drop-zone-icon">📎</div>
                  <p className="drop-zone-text">
                    {isDragging ? 'Drop file here' : 'Drag & drop image/PDF or click to upload'}
                  </p>
                  <p className="drop-zone-hint">
                    Supports: Images (PNG, JPG, JPEG - Max 10MB) or PDF (Max 20MB)
                  </p>
                </div>
              )}
            </div>

            <div className="divider">
              <span>OR</span>
            </div>

            {/* Text Input */}
            <div className="text-input-container">
              <label className="input-label">
                <span className="icon">✍️</span>
                Paste your text here
              </label>
              <textarea
                className="text-input"
                placeholder="Paste your lecture notes, exam questions, or any study material here..."
                value={textInput}
                onChange={(e) => setTextInput(e.target.value)}
                rows="8"
              />
              {textInput && (
                <div className="text-stats">
                  <span>{textInput.length} characters</span>
                  <span>{textInput.split(/\s+/).filter(Boolean).length} words</span>
                  {textInput.split(/\s+/).filter(Boolean).length < 150 && (
                    <span className="text-warning">
                      ⚠️ For better AI results, provide 150+ words
                    </span>
                  )}
                </div>
              )}
            </div>
          </div>
        </section>

        {/* OCR Result Section */}
        {ocrText && (
          <section className="section ocr-section fade-in">
            <div className="section-header">
              <h2>
                <span className="icon">🔍</span>
                Extracted Text
              </h2>
              <button className="btn-icon" onClick={() => copyToClipboard(ocrText)} title="Copy">
                📋
              </button>
            </div>
            <div className="result-box">
              <pre className="result-text">{ocrText}</pre>
            </div>
          </section>
        )}

        {/* Action Buttons */}
        {textInput && (
          <section className="section actions-section fade-in">
            <h2>
              <span className="icon">⚡</span>
              AI Analysis Tools
            </h2>
            <div className="button-group">
              <button
                onClick={handleSummarize}
                disabled={loading.summarize}
                className="btn btn-primary"
              >
                {loading.summarize ? (
                  <>
                    <span className="spinner-small"></span>
                    Processing...
                  </>
                ) : (
                  <>
                    <span className="btn-icon">📝</span>
                    Summarize
                  </>
                )}
              </button>
              <button
                onClick={handleClassify}
                disabled={loading.classify}
                className="btn btn-primary"
              >
                {loading.classify ? (
                  <>
                    <span className="spinner-small"></span>
                    Processing...
                  </>
                ) : (
                  <>
                    <span className="btn-icon">🏷️</span>
                    Classify Topics
                  </>
                )}
              </button>
              <button
                onClick={handleGenerateQuestions}
                disabled={loading.generateQuestions}
                className="btn btn-primary"
              >
                {loading.generateQuestions ? (
                  <>
                    <span className="spinner-small"></span>
                    Processing...
                  </>
                ) : (
                  <>
                    <span className="btn-icon">❓</span>
                    Generate Questions
                  </>
                )}
              </button>
              <button
                onClick={handleProcessAll}
                disabled={loading.summarize || loading.classify || loading.generateQuestions}
                className="btn btn-secondary"
              >
                <span className="btn-icon">🚀</span>
                Process All
              </button>
            </div>
          </section>
        )}

        {/* Summary Section */}
        {summary && (
          <section className="section summary-section fade-in">
            <div className="section-header">
              <h2>
                <span className="icon">📄</span>
                Summary
                {summaryData && getConfidenceBadge(summaryData.confidence)}
              </h2>
              <div className="header-actions">
                <button 
                  className="btn-toggle" 
                  onClick={() => setShowOriginalText(!showOriginalText)}
                  title="Toggle original text"
                >
                  {showOriginalText ? '📝 Show Summary' : '📄 Show Original'}
                </button>
                <button className="btn-icon" onClick={() => copyToClipboard(summary)} title="Copy">
                  📋
                </button>
              </div>
            </div>
            
            {summaryData && (
              <div className="summary-meta">
                <span>Original: {summaryData.original_length} words</span>
                <span>Summary: {summaryData.summary_length} words</span>
              </div>
            )}

            <div className="comparison-container">
              {showOriginalText ? (
                <div className="result-box original-text-box">
                  <h3>Original Text</h3>
                  <p className="result-text">{textInput}</p>
                </div>
              ) : (
                <div className="result-box">
                  <h3>Summary</h3>
                  <p className="result-text">{summary}</p>
                </div>
              )}
            </div>
          </section>
        )}

        {/* Topics Section */}
        {topics.length > 0 && (
          <section className="section topics-section fade-in">
            <div className="section-header">
              <h2>
                <span className="icon">🏷️</span>
                Classified Topics
                {topicsConfidence && getConfidenceBadge(topicsConfidence)}
              </h2>
            </div>
            <div className="topics-container">
              {topics.map((topic, index) => (
                <div key={index} className="topic-card" style={{ animationDelay: `${index * 0.1}s` }}>
                  <span className="topic-name">{topic.topic}</span>
                  <div className="topic-score-container">
                    <div className="topic-score-bar">
                      <div 
                        className="topic-score-fill" 
                        style={{ width: `${topic.score * 100}%` }}
                      ></div>
                    </div>
                    <span className="topic-score-text">
                      {(topic.score * 100).toFixed(1)}%
                    </span>
                  </div>
                </div>
              ))}
            </div>
          </section>
        )}

        {/* Questions Section */}
        {questions.length > 0 && (
          <section className="section questions-section fade-in">
            <div className="section-header">
              <h2>
                <span className="icon">❓</span>
                Practice Questions
                {questionsConfidence && getConfidenceBadge(questionsConfidence)}
              </h2>
            </div>
            <div className="questions-container">
              {questions.map((q, index) => (
                <div key={index} className="question-card" style={{ animationDelay: `${index * 0.1}s` }}>
                  <div className="question-header">
                    <div className="question-number">
                      Question {index + 1}
                      {q.type && (
                        <span className={`question-type question-type-${q.type.toLowerCase()}`}>
                          {q.type}
                        </span>
                      )}
                    </div>
                    <button 
                      className="btn-icon-small" 
                      onClick={() => copyToClipboard(q.question)}
                      title="Copy question"
                    >
                      📋
                    </button>
                  </div>
                  <div className="question-text">{q.question}</div>
                  {q.context && (
                    <details className="question-context">
                      <summary>View Context</summary>
                      <p>{q.context}</p>
                    </details>
                  )}
                </div>
              ))}
            </div>
          </section>
        )}
      </main>

      <footer className="app-footer">
        <div className="footer-content">
          <p>AI Study Companion</p>
        </div>
      </footer>
    </div>
  );
}

export default App;
