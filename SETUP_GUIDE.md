# 📋 Complete Setup Guide for Clinical RAG Assistant

This guide will walk you through setting up and deploying your Clinical RAG Assistant on Streamlit Cloud.

---

## 📦 Part 1: Prepare Files from Google Drive

### Files to Download from Your Google Drive

From your Colab environment, you need to export these files:

#### 1. ChromaDB Database (REQUIRED)
**Location in Colab:** `/content/chroma_db/`

**Steps:**
```python
# Run this in Google Colab to zip ChromaDB
import shutil
shutil.make_archive('/content/drive/MyDrive/chroma_db', 'zip', '/content/chroma_db')
```

**What to download:**
- `chroma_db.zip` (~20-30 MB compressed)

**Where it contains:**
- `chroma.sqlite3` - Main database file
- Collection metadata and embeddings
- Index files

#### 2. Processed Documents (OPTIONAL)
**Location in Colab:** `/content/processed_data/processed_documents.parquet`

**What to download:**
- `processed_documents.parquet` (~2-3 MB)

**Purpose:** Contains document metadata for enhanced display (not critical for core functionality)

---

## 🗂️ Part 2: Create GitHub Repository

### Step 1: Create Repository Structure

Create this exact folder structure locally:

```
clinical-rag-assistant/
│
├── app.py                          # Main Streamlit app (provided)
├── src/
│   ├── __init__.py                 # Empty file
│   └── rag_pipeline.py            # RAG pipeline code (provided)
│
├── data/
│   ├── .gitkeep                   # Keep empty directory in Git
│   └── README.md                  # Instructions for data download
│
├── requirements.txt               # Python dependencies (provided)
├── packages.txt                   # System dependencies (provided)
├── README.md                      # Main README (provided)
├── SETUP_GUIDE.md                 # This file
└── .gitignore                     # Git ignore rules (provided)
```

### Step 2: Create Required Files

#### Create `src/__init__.py`
```bash
touch src/__init__.py
```

#### Create `data/.gitkeep`
```bash
mkdir -p data
touch data/.gitkeep
```

#### Create `data/README.md`
```markdown
# Data Directory

## Required Files

Download these files from Google Drive and place them here:

### ChromaDB Database (REQUIRED)
1. Download `chroma_db.zip` from: [YOUR_DRIVE_LINK]
2. Extract here to create `data/chroma_db/` directory
3. Verify it contains `chroma.sqlite3`

### Processed Documents (OPTIONAL)
1. Download `processed_documents.parquet` from: [YOUR_DRIVE_LINK]
2. Place directly in `data/` directory

## Directory Structure After Setup

```
data/
├── chroma_db/
│   ├── chroma.sqlite3
│   ├── [UUID folders with embeddings]
│   └── [other ChromaDB files]
└── processed_documents.parquet
```

## Verification

Run this to verify setup:
```python
import chromadb
client = chromadb.PersistentClient(path="./data/chroma_db")
collection = client.get_collection(name="clinical_notes")
print(f"Documents in collection: {collection.count()}")  # Should show 934
```
```

---

## 🚀 Part 3: Upload to Google Drive & Get Shareable Links

### Step 1: Upload Files

1. Go to your Google Drive
2. Create folder: `Clinical-RAG-Public/`
3. Upload:
   - `chroma_db.zip`
   - `processed_documents.parquet`

### Step 2: Create Shareable Links

For each file:
1. Right-click → Share
2. Change to "Anyone with the link"
3. Copy the link
4. Extract the FILE_ID from the link

**Example:**
```
https://drive.google.com/file/d/1ABcDeFgHiJkLmNoPqRsTuVwXyZ/view?usp=sharing
                              ^^^^^^^^^^^^^^^^^^^^^^^^
                              This is your FILE_ID
```

### Step 3: Create Direct Download Links

Convert Google Drive links to direct download format:

**Original link:**
```
https://drive.google.com/file/d/FILE_ID/view?usp=sharing
```

**Direct download link:**
```
https://drive.google.com/uc?export=download&id=FILE_ID
```

---

## 📤 Part 4: Push to GitHub

### Step 1: Initialize Git Repository

```bash
cd clinical-rag-assistant

# Initialize git
git init

# Add all files
git add .

# Commit
git commit -m "Initial commit: Clinical RAG Assistant with Mistral-7B"
```

### Step 2: Create GitHub Repository

1. Go to [GitHub](https://github.com)
2. Click "New repository"
3. Name: `clinical-rag-assistant`
4. Description: "AI-powered clinical QA system using RAG with MIMIC-IV-EXT"
5. Keep it **Public** (required for Streamlit Cloud free tier)
6. Don't initialize with README (we already have one)
7. Click "Create repository"

### Step 3: Push to GitHub

```bash
# Add remote
git remote add origin https://github.com/YOUR_USERNAME/clinical-rag-assistant.git

# Push
git branch -M main
git push -u origin main
```

### Step 4: Update README with Drive Links

Edit `README.md` and `data/README.md` to include your actual Google Drive links:

```markdown
**Google Drive Links:**
- ChromaDB: `https://drive.google.com/uc?export=download&id=YOUR_CHROMA_FILE_ID`
- Processed Data: `https://drive.google.com/uc?export=download&id=YOUR_PARQUET_FILE_ID`
```

Commit and push:
```bash
git add README.md data/README.md
git commit -m "Add Google Drive download links"
git push
```

---

## ☁️ Part 5: Deploy on Streamlit Cloud

### Step 1: Sign Up for Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with GitHub account
3. Authorize Streamlit to access your repositories

### Step 2: Deploy App

1. Click "New app"
2. Select:
   - **Repository:** `YOUR_USERNAME/clinical-rag-assistant`
   - **Branch:** `main`
   - **Main file path:** `app.py`
3. Click "Advanced settings" (optional):
   - Python version: `3.9` or `3.10`
4. Click "Deploy!"

### Step 3: Wait for Deployment

**Expected timeline:**
- ⏱️ Initial build: 5-10 minutes
- 📦 Model download (first run): 10-15 minutes
- ✅ Total first deployment: ~20-25 minutes

**What's happening:**
1. Installing system packages (`packages.txt`)
2. Installing Python packages (`requirements.txt`)
3. Downloading models (~14GB):
   - E5-small-v2 (~130 MB)
   - Mistral-7B-Instruct-v0.2 (~14 GB)

### Step 4: Monitor Deployment

Watch the logs for:
```
✓ Installing system packages...
✓ Installing Python requirements...
✓ Loading E5 model...
✓ Loading Mistral-7B model...
✓ App is live!
```

---

## 🎯 Part 6: Setup Data Files (For Users)

### For Local Development

Users who clone your repo need to:

1. **Download ChromaDB:**
```bash
# Download from your Drive link
wget -O chroma_db.zip "YOUR_DIRECT_DOWNLOAD_LINK"

# Extract
unzip chroma_db.zip -d data/

# Verify
ls data/chroma_db/  # Should see chroma.sqlite3
```

2. **Download processed documents (optional):**
```bash
wget -O data/processed_documents.parquet "YOUR_PARQUET_DOWNLOAD_LINK"
```

3. **Run app:**
```bash
streamlit run app.py
```

### For Streamlit Cloud

⚠️ **Important:** Streamlit Cloud cannot download large files from Google Drive during deployment.

**Two options:**

#### Option A: Small Data Bundle (Recommended)
If your `chroma_db` is <100MB compressed:
1. Include it in GitHub repo (not ideal but works)
2. Update `.gitignore` to NOT exclude it:
   ```gitignore
   # Comment out this line:
   # data/chroma_db/
   ```

#### Option B: External Storage (Better for Production)
1. Upload `chroma_db.zip` to:
   - AWS S3 (public bucket)
   - Dropbox (direct link)
   - GitHub Releases (if <100MB)
2. Download in `app.py` on startup:
   ```python
   import urllib.request
   import zipfile
   
   @st.cache_resource
   def download_and_extract_chroma():
       if not Path("data/chroma_db").exists():
           with st.spinner("Downloading ChromaDB..."):
               urllib.request.urlretrieve(
                   "YOUR_DIRECT_DOWNLOAD_URL",
                   "chroma_db.zip"
               )
               with zipfile.ZipFile("chroma_db.zip", 'r') as zip_ref:
                   zip_ref.extractall("data/")
   ```

---

## 🔍 Part 7: Verification Checklist

### Local Verification

```bash
# 1. Check directory structure
tree -L 2

# 2. Verify ChromaDB
python -c "import chromadb; c = chromadb.PersistentClient(path='./data/chroma_db'); print(c.list_collections())"

# 3. Test imports
python -c "from src.rag_pipeline import RAGPipelineMistral; print('OK')"

# 4. Run app
streamlit run app.py
```

### Streamlit Cloud Verification

After deployment, test:
1. ✅ App loads without errors
2. ✅ "Load Models" button works
3. ✅ Can submit a query
4. ✅ Retrieves documents
5. ✅ Generates answer
6. ✅ Shows sources

---

## 🐛 Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'src'"

**Solution:**
```bash
# Ensure src/__init__.py exists
touch src/__init__.py
git add src/__init__.py
git commit -m "Add src/__init__.py"
git push
```

### Issue: "ChromaDB collection not found"

**Solution:**
```bash
# Verify collection name matches
python -c "
import chromadb
client = chromadb.PersistentClient(path='./data/chroma_db')
print(client.list_collections())
"
# Should output: [Collection(name=clinical_notes)]
```

### Issue: "Out of memory" on Streamlit Cloud

**Solutions:**
1. Reduce `max_new_tokens` to 128-256
2. Use smaller batch size for retrieval
3. Add memory cleanup after generation:
   ```python
   import gc
   torch.cuda.empty_cache()
   gc.collect()
   ```

### Issue: Models download timeout

**Solution:** Models are large (~14GB). Be patient on first run.
- Streamlit Cloud: ~15-20 minutes
- Local: ~10 minutes (depends on internet speed)

---

## 📊 Expected Performance

### Streamlit Cloud (Free Tier)
- **Resources:** 1 CPU, 800MB RAM
- **Retrieval:** ~30-50ms
- **Generation:** ~120-180 seconds (CPU-only)
- **Concurrent users:** 1-2 (not production-ready)

### Streamlit Cloud (Paid/Teams)
- **Resources:** Can request GPU
- **Generation:** ~60-90 seconds (with GPU)
- **Concurrent users:** More stable

### Local with GPU
- **Retrieval:** ~10-20ms
- **Generation:** ~60-90 seconds
- **Best for development**

---

## 🎓 Next Steps After Deployment

1. **Share your app:**
   - URL will be: `https://YOUR_USERNAME-clinical-rag-assistant.streamlit.app`
   - Share with colleagues for testing

2. **Monitor usage:**
   - Check Streamlit Cloud dashboard
   - View logs for errors

3. **Iterate and improve:**
   - Add more disease categories
   - Fine-tune prompts
   - Optimize performance

4. **Production deployment:**
   - Consider AWS/GCP with GPU
   - Use FastAPI for API endpoint
   - Add authentication

---

## 📞 Support

If you encounter issues:
1. Check Streamlit Cloud logs
2. Review GitHub Issues
3. Consult Streamlit documentation: [docs.streamlit.io](https://docs.streamlit.io)

---

**Good luck with your deployment! 🚀**