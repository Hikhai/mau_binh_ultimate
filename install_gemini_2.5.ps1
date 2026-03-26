# ============================================
# Install Gemini 2.5 Flash Dependencies
# ============================================

Write-Host "🚀 Installing Mau Binh Ultimate (Gemini 2.5)" -ForegroundColor Cyan

# 1. Upgrade pip
Write-Host "`n📦 Upgrading pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip setuptools wheel

# 2. Uninstall conflicting packages
Write-Host "`n🗑️ Removing old versions..." -ForegroundColor Yellow
pip uninstall -y pyparsing matplotlib google-generativeai google-ai-generativelanguage

# 3. Install core (numpy, pandas, scipy)
Write-Host "`n📊 Installing core libs..." -ForegroundColor Yellow
pip install numpy==1.26.4 pandas==2.1.4 scipy==1.11.4

# 4. Install matplotlib (with compatible pyparsing)
Write-Host "`n📈 Installing matplotlib..." -ForegroundColor Yellow
pip install pyparsing==3.1.1 matplotlib==3.8.2 seaborn==0.13.0

# 5. Install Gemini 2.5 SDK (UPGRADED)
Write-Host "`n🤖 Installing Gemini 2.5 Flash SDK..." -ForegroundColor Green
pip install `
  protobuf==4.25.3 `
  grpcio==1.60.1 `
  google-auth==2.27.0 `
  google-api-core==2.15.0 `
  google-ai-generativelanguage==0.6.10 `
  google-generativeai==0.8.3

# 6. Install Streamlit
Write-Host "`n🌐 Installing Streamlit..." -ForegroundColor Yellow
pip install streamlit==1.31.1 streamlit-paste-button==0.1.2

# 7. Install ML/CV
Write-Host "`n🧠 Installing ML/CV libs..." -ForegroundColor Yellow
pip install `
  torch==2.1.2 `
  torchvision==0.16.2 `
  opencv-python==4.9.0.80 `
  Pillow==10.2.0 `
  scikit-learn==1.3.2

# 8. Install utils
Write-Host "`n🔧 Installing utilities..." -ForegroundColor Yellow
pip install pyyaml requests python-dotenv tqdm colorama

# 9. Verify installation
Write-Host "`n✅ Verifying installation..." -ForegroundColor Green

python -c "import google.generativeai as genai; print('✅ Gemini SDK:', genai.__version__)"
python -c "import streamlit as st; print('✅ Streamlit:', st.__version__)"
python -c "import torch; print('✅ PyTorch:', torch.__version__)"
python -c "import numpy as np; print('✅ NumPy:', np.__version__)"

Write-Host "`n🎉 Installation complete!" -ForegroundColor Green
Write-Host "Run: streamlit run src/web/app.py" -ForegroundColor Cyan