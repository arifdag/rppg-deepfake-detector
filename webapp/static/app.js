// Translations
const translations = {
    en: {
        // Navigation
        'nav.analyze': 'Analyze',
        'nav.about': 'About Model',
        
        // Analyze Section
        'analyze.title': 'Video Authenticity Analysis',
        'analyze.subtitle': 'Upload a video to detect potential deepfake manipulation using advanced visual and physiological signal analysis',
        
        // Upload
        'upload.dropHere': 'Drop video file here',
        'upload.orClick': 'or click to browse',
        'upload.uploading': 'Uploading...',
        'upload.remove': 'Remove',
        'upload.analyzeBtn': 'Analyze Video',
        
        // Results
        'results.placeholder': 'Results will appear here after analysis',
        'results.analyzing': 'Analyzing Video',
        'results.extracting': 'Extracting frames and features...',
        'results.verdict': 'Verdict',
        'results.verdictReal': 'AUTHENTIC',
        'results.verdictFake': 'DEEPFAKE',
        'results.trustScore': 'Trust Score',
        'results.trustDesc': 'Overall reliability of the analysis',
        'results.fakeProb': 'Fake Probability',
        'results.realProb': 'Real Probability',
        'results.confidence': 'Confidence',
        'results.consistency': 'Consistency',
        'results.detailsTitle': 'Analysis Details',
        'results.framesAnalyzed': 'Frames Analyzed',
        'results.videoFps': 'Video FPS',
        'results.analysisPasses': 'Analysis Passes',
        'results.noticeTitle': 'Important Notice',
        'results.noticeText': 'This model is trained on face-swap deepfakes (Celeb-DF v2). It may not reliably detect:',
        'results.noticeItem1': 'AI-generated videos (Gemini, Sora, Runway, etc.)',
        'results.noticeItem2': 'Fully synthetic faces or avatars',
        'results.noticeItem3': 'Audio-only manipulations',
        'results.noticeTip': 'For best results, use with videos containing face manipulation or face-swapping.',
        
        // About
        'about.title': 'Model Architecture',
        'about.subtitle': 'Technical details about the Visual + rPPG Fusion detection system',
        'about.visualTitle': 'Visual Branch',
        'about.visualDesc': 'EfficientNet-B2 backbone with Noisy Student pretraining extracts spatial features from face crops. Temporal attention pooling aggregates features across multiple frames.',
        'about.framesPerVideo': '8 frames/video',
        'about.rppgTitle': 'rPPG Branch',
        'about.rppgDesc': 'Remote photoplethysmography extracts subtle blood flow patterns from facial video. The POS (Plane Orthogonal to Skin) method analyzes physiological signals that are difficult to fake.',
        'about.posAlgorithm': 'POS Algorithm',
        'about.fftFeatures': 'FFT Features',
        'about.bpmBand': '42-240 BPM Band',
        'about.fusionTitle': 'Fusion Head',
        'about.fusionDesc': 'Features from both branches are concatenated and processed through a multi-layer perceptron. Focal loss handles class imbalance during training.',
        'about.mlpFusion': 'MLP Fusion',
        'about.perfTitle': 'Performance Metrics',
        'about.accuracy': 'Accuracy',
        'about.balancedAcc': 'Balanced Acc',
        'about.perfNote': 'Evaluated on Celeb-DF v2 test set with 518 videos',
        
        // Footer
        'footer.system': 'Visual + rPPG Fusion Deepfake Detection System',
        'footer.dataset': 'Trained on Celeb-DF v2 Dataset',
        
        // Loading status messages
        'status.extractingFrames': 'Extracting video frames...',
        'status.detectingFaces': 'Detecting faces...',
        'status.extractingRppg': 'Extracting rPPG features...',
        'status.visualAnalysis': 'Running visual analysis...',
        'status.fusingFeatures': 'Fusing multimodal features...',
        'status.computing': 'Computing final prediction...',
        
        // Alerts
        'alert.invalidFile': 'Please select a valid video file (MP4, AVI, MOV, MKV, or WebM)',
        'alert.error': 'Error: ',
        'alert.analysisError': 'An error occurred during analysis. Please try again.'
    },
    tr: {
        // Navigation
        'nav.analyze': 'Analiz',
        'nav.about': 'Model Hakkında',
        
        // Analyze Section
        'analyze.title': 'Video Gerçeklik Analizi',
        'analyze.subtitle': 'Gelişmiş görsel ve fizyolojik sinyal analizi kullanarak potansiyel deepfake manipülasyonunu tespit etmek için bir video yükleyin',
        
        // Upload
        'upload.dropHere': 'Video dosyasını buraya bırakın',
        'upload.orClick': 'veya göz atmak için tıklayın',
        'upload.uploading': 'Yükleniyor...',
        'upload.remove': 'Kaldır',
        'upload.analyzeBtn': 'Videoyu Analiz Et',
        
        // Results
        'results.placeholder': 'Analiz sonrasında sonuçlar burada görünecek',
        'results.analyzing': 'Video Analiz Ediliyor',
        'results.extracting': 'Kareler ve özellikler çıkarılıyor...',
        'results.verdict': 'Sonuç',
        'results.verdictReal': 'GERÇEKTİR',
        'results.verdictFake': 'DEEPFAKE',
        'results.trustScore': 'Güven Puanı',
        'results.trustDesc': 'Analizin genel güvenilirliği',
        'results.fakeProb': 'Sahte Olasılığı',
        'results.realProb': 'Gerçek Olasılığı',
        'results.confidence': 'Güven',
        'results.consistency': 'Tutarlılık',
        'results.detailsTitle': 'Analiz Detayları',
        'results.framesAnalyzed': 'Analiz Edilen Kare',
        'results.videoFps': 'Video FPS',
        'results.analysisPasses': 'Analiz Geçişleri',
        'results.noticeTitle': 'Önemli Uyarı',
        'results.noticeText': 'Bu model yüz değiştirme deepfake\'leri (Celeb-DF v2) üzerinde eğitilmiştir. Aşağıdakileri güvenilir şekilde tespit edemeyebilir:',
        'results.noticeItem1': 'Yapay zeka ile üretilen videolar (Gemini, Sora, Runway, vb.)',
        'results.noticeItem2': 'Tamamen sentetik yüzler veya avatarlar',
        'results.noticeItem3': 'Sadece ses manipülasyonları',
        'results.noticeTip': 'En iyi sonuçlar için, yüz manipülasyonu veya yüz değiştirme içeren videolarla kullanın.',
        
        // About
        'about.title': 'Model Mimarisi',
        'about.subtitle': 'Visual + rPPG Fusion tespit sistemi hakkında teknik detaylar',
        'about.visualTitle': 'Görsel Dal',
        'about.visualDesc': 'Noisy Student ön eğitimi ile EfficientNet-B2 omurgası, yüz kırpımlarından uzamsal özellikler çıkarır. Temporal dikkat havuzlama, birden fazla karedeki özellikleri birleştirir.',
        'about.framesPerVideo': '8 kare/video',
        'about.rppgTitle': 'rPPG Dalı',
        'about.rppgDesc': 'Uzaktan fotopletismografi, yüz videosundan ince kan akışı desenlerini çıkarır. POS (Cilde Ortogonal Düzlem) yöntemi, taklit edilmesi zor fizyolojik sinyalleri analiz eder.',
        'about.posAlgorithm': 'POS Algoritması',
        'about.fftFeatures': 'FFT Özellikleri',
        'about.bpmBand': '42-240 BPM Bandı',
        'about.fusionTitle': 'Birleştirme Katmanı',
        'about.fusionDesc': 'Her iki daldan gelen özellikler birleştirilir ve çok katmanlı algılayıcı üzerinden işlenir. Focal loss, eğitim sırasında sınıf dengesizliğini ele alır.',
        'about.mlpFusion': 'MLP Birleştirme',
        'about.perfTitle': 'Performans Metrikleri',
        'about.accuracy': 'Doğruluk',
        'about.balancedAcc': 'Dengeli Doğ.',
        'about.perfNote': 'Celeb-DF v2 test seti üzerinde 518 video ile değerlendirildi',
        
        // Footer
        'footer.system': 'Visual + rPPG Fusion Deepfake Tespit Sistemi',
        'footer.dataset': 'Celeb-DF v2 Veri Seti Üzerinde Eğitildi',
        
        // Loading status messages
        'status.extractingFrames': 'Video kareleri çıkarılıyor...',
        'status.detectingFaces': 'Yüzler tespit ediliyor...',
        'status.extractingRppg': 'rPPG özellikleri çıkarılıyor...',
        'status.visualAnalysis': 'Görsel analiz yapılıyor...',
        'status.fusingFeatures': 'Çok modlu özellikler birleştiriliyor...',
        'status.computing': 'Son tahmin hesaplanıyor...',
        
        // Alerts
        'alert.invalidFile': 'Lütfen geçerli bir video dosyası seçin (MP4, AVI, MOV, MKV veya WebM)',
        'alert.error': 'Hata: ',
        'alert.analysisError': 'Analiz sırasında bir hata oluştu. Lütfen tekrar deneyin.'
    }
};

// Current language
let currentLang = localStorage.getItem('deepguard-lang') || 'en';

// Translation function
function t(key) {
    return translations[currentLang][key] || translations['en'][key] || key;
}

// Apply translations to all elements with data-i18n attribute
function applyTranslations() {
    document.querySelectorAll('[data-i18n]').forEach(el => {
        const key = el.getAttribute('data-i18n');
        const text = t(key);
        if (text) {
            el.textContent = text;
        }
    });
    
    // Update HTML lang attribute
    document.documentElement.lang = currentLang;
}

// Switch language
function switchLanguage(lang) {
    currentLang = lang;
    localStorage.setItem('deepguard-lang', lang);
    
    // Update button states
    document.querySelectorAll('.lang-btn').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.lang === lang);
    });
    
    applyTranslations();
}

document.addEventListener('DOMContentLoaded', () => {
    // Initialize language
    applyTranslations();
    document.querySelectorAll('.lang-btn').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.lang === currentLang);
        btn.addEventListener('click', () => switchLanguage(btn.dataset.lang));
    });
    // Elements
    const uploadZone = document.getElementById('upload-zone');
    const videoInput = document.getElementById('video-input');
    const fileInfo = document.getElementById('file-info');
    const filePreview = document.getElementById('file-preview');
    const videoPreview = document.getElementById('video-preview');
    const fileName = document.getElementById('file-name');
    const clearBtn = document.getElementById('clear-btn');
    const analyzeBtn = document.getElementById('analyze-btn');
    const uploadProgress = document.getElementById('upload-progress');
    const progressFill = document.getElementById('progress-fill');
    const progressText = document.getElementById('progress-text');
    
    const resultsPanel = document.getElementById('results-panel');
    const resultsPlaceholder = resultsPanel.querySelector('.results-placeholder');
    const resultsLoading = document.getElementById('results-loading');
    const loadingStatus = document.getElementById('loading-status');
    const resultsContent = document.getElementById('results-content');
    
    // Navigation
    const navBtns = document.querySelectorAll('.nav-btn');
    const sections = document.querySelectorAll('.section');
    
    let selectedFile = null;
    
    // Navigation handling
    navBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            const sectionId = btn.dataset.section;
            
            navBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            
            sections.forEach(s => s.classList.remove('active'));
            document.getElementById(`${sectionId}-section`).classList.add('active');
        });
    });
    
    // Drag and drop handling
    uploadZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadZone.classList.add('dragover');
    });
    
    uploadZone.addEventListener('dragleave', () => {
        uploadZone.classList.remove('dragover');
    });
    
    uploadZone.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadZone.classList.remove('dragover');
        
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFileSelect(files[0]);
        }
    });
    
    // Click to upload
    uploadZone.addEventListener('click', () => {
        videoInput.click();
    });
    
    videoInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFileSelect(e.target.files[0]);
        }
    });
    
    // Handle file selection
    function handleFileSelect(file) {
        const validTypes = ['video/mp4', 'video/avi', 'video/quicktime', 'video/x-matroska', 'video/webm'];
        const validExtensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm'];
        
        const extension = '.' + file.name.split('.').pop().toLowerCase();
        
        if (!validTypes.includes(file.type) && !validExtensions.includes(extension)) {
            alert(t('alert.invalidFile'));
            return;
        }
        
        selectedFile = file;
        
        // Show file info
        uploadZone.classList.add('hidden');
        fileInfo.classList.add('active');
        
        // Set file name
        fileName.textContent = file.name;
        
        // Create video preview
        const url = URL.createObjectURL(file);
        videoPreview.src = url;
        
        // Enable analyze button
        analyzeBtn.disabled = false;
        
        // Reset results
        resetResults();
    }
    
    // Clear file selection
    clearBtn.addEventListener('click', () => {
        clearSelection();
    });
    
    function clearSelection() {
        selectedFile = null;
        videoInput.value = '';
        
        uploadZone.classList.remove('hidden');
        fileInfo.classList.remove('active');
        
        if (videoPreview.src) {
            URL.revokeObjectURL(videoPreview.src);
            videoPreview.src = '';
        }
        
        analyzeBtn.disabled = true;
        resetResults();
    }
    
    function resetResults() {
        resultsPlaceholder.classList.remove('hidden');
        resultsLoading.classList.remove('active');
        resultsContent.classList.remove('active');
    }
    
    // Analyze video
    analyzeBtn.addEventListener('click', async () => {
        if (!selectedFile) return;
        
        // Show loading state
        resultsPlaceholder.classList.add('hidden');
        resultsContent.classList.remove('active');
        resultsLoading.classList.add('active');
        
        // Disable button
        analyzeBtn.disabled = true;
        
        // Update loading status
        const statusMessages = [
            t('status.extractingFrames'),
            t('status.detectingFaces'),
            t('status.extractingRppg'),
            t('status.visualAnalysis'),
            t('status.fusingFeatures'),
            t('status.computing')
        ];
        
        let statusIndex = 0;
        const statusInterval = setInterval(() => {
            statusIndex = (statusIndex + 1) % statusMessages.length;
            loadingStatus.textContent = statusMessages[statusIndex];
        }, 2000);
        
        try {
            const formData = new FormData();
            formData.append('video', selectedFile);
            
            const response = await fetch('/analyze', {
                method: 'POST',
                body: formData
            });
            
            const result = await response.json();
            
            clearInterval(statusInterval);
            
            if (result.error) {
                alert(t('alert.error') + result.error);
                resetResults();
                analyzeBtn.disabled = false;
                return;
            }
            
            // Display results
            displayResults(result);
            
        } catch (error) {
            clearInterval(statusInterval);
            console.error('Analysis error:', error);
            alert(t('alert.analysisError'));
            resetResults();
        }
        
        analyzeBtn.disabled = false;
    });
    
    // Display analysis results
    function displayResults(result) {
        resultsLoading.classList.remove('active');
        resultsContent.classList.add('active');
        
        const isFake = result.prediction === 'FAKE';
        
        // Verdict
        const verdictIndicator = document.getElementById('verdict-indicator');
        const verdictText = document.getElementById('verdict-text');
        
        verdictIndicator.className = 'verdict-indicator ' + (isFake ? 'fake' : 'real');
        verdictText.textContent = isFake ? t('results.verdictFake') : t('results.verdictReal');
        verdictText.className = 'verdict-text ' + (isFake ? 'fake' : 'real');
        
        // Trust Score
        document.getElementById('trust-score').textContent = result.trust_score + '%';
        animateFill('trust-fill', result.trust_score);
        
        // Probabilities
        document.getElementById('fake-prob').textContent = result.fake_probability + '%';
        animateFill('fake-fill', result.fake_probability);
        
        document.getElementById('real-prob').textContent = result.real_probability + '%';
        animateFill('real-fill', result.real_probability);
        
        // Confidence & Consistency
        document.getElementById('confidence').textContent = result.confidence + '%';
        animateFill('confidence-fill', result.confidence);
        
        document.getElementById('consistency').textContent = result.consistency + '%';
        animateFill('consistency-fill', result.consistency);
        
        // Analysis details
        document.getElementById('frames-count').textContent = result.num_frames_analyzed;
        document.getElementById('video-fps').textContent = result.fps;
    }
    
    function animateFill(elementId, percentage) {
        const element = document.getElementById(elementId);
        setTimeout(() => {
            element.style.width = percentage + '%';
        }, 100);
    }
    
    // Load model info on about section
    async function loadModelInfo() {
        try {
            const response = await fetch('/model-info');
            const info = await response.json();
            
            if (info.validation_auc) {
                const perfAuc = document.getElementById('perf-auc');
                if (perfAuc) {
                    perfAuc.textContent = (info.test_auc * 100).toFixed(2) + '%';
                }
            }
        } catch (error) {
            console.log('Could not load model info');
        }
    }
    
    loadModelInfo();
});
