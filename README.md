# Meridian AI Risk Engine (OSINT & NLP Pipeline)

![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)
![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

An automated, end-to-end data pipeline engineered for Third-Party Risk Management (TPRM). This system performs zero-touch vendor due diligence by crawling a vendor's public digital footprint, utilizing local Natural Language Processing (NLP) to detect "Shadow AI" usage, and outputting an actuarial cybersecurity risk score based on deterministic risk matrices.

<img width="1098" height="592" alt="optiv-architecture-diagram" src="https://github.com/user-attachments/assets/0ad60461-b990-4a8a-8eec-bddc99170850" />
<img width="2863" height="556" alt="optiv" src="https://github.com/user-attachments/assets/b5be08ca-f62b-49f5-9c25-295d93cc27ba" />

---

## 📊 System Architecture & Data Flow

The system operates on a linear, feed-forward design, passing data seamlessly in-memory via Pandas DataFrames to eliminate intermediate I/O bottlenecks.

### Core Processing Stages

1. **Stage 1: Criticality Scoring (`src/criticality.py`)**
   Normalizes input schema and calculates a weighted search depth tier (1-4) based on data sensitivity and operational dependency.
2. **Stage 2: OSINT Web Scraper (`src/scraper.py`)**
   Executes rate-limited, targeted domain crawling (Trust Centers, Legal, Jobs, SEC filings) and regex-based evidence extraction.
3. **Stage 3: Data Flattener (`src/pipeline.py`)**
   Cleans HTML/Unicode artifacts and structural noise, transforming nested JSON arrays into a unified DataFrame.
4. **Stage 4: NLP Engine (`src/nlp_engine.py`)**
   Deploys `BART-Large-MNLI` for zero-shot classification (filtering marketing hype) and `Qwen-2.5` for synthesizing executive capability summaries.
5. **Stage 5: Risk Matrix (`src/risk_matrix.py`)**
   Deduplicates evidence, applies Named Entity Recognition (NER) for sub-processors, and computes the final risk tier (Low to Critical).

---

## 🛠️ Technology Stack

* **Data Engineering:** `pandas`, `numpy`, `openpyxl`
* **OSINT / Scraping:** `requests`, `beautifulsoup4`, `urllib3`
* **Machine Learning:** `torch`, `transformers`, `accelerate` (Hugging Face)
* **Frontend / UI:** `streamlit`
* **Testing:** `pytest`

---

## 🚀 Installation & Setup

**Hardware Note:** Running the NLP engine (Stage 4) on a standard CPU is possible but slow. A dedicated GPU or Google Colab T4 environment is highly recommended.

1. **Clone the repository:**

   ```bash
   git clone https://github.com/dm0x23/TPRM-AI-Scanner.git
   cd TPRM-AI-Scanner
   ```

2. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare the Data Environment:**
   Ensure the required directory structure exists. Place the official input template (`Meridian_Vendor_Input_2.xlsx`) into the `data/input/` folder.

   ```text
   ├── data/
   │   ├── input/
   │   │   └── Meridian_Vendor_Input_2.xlsx
   │   └── output/
   ```

---

## 💻 Usage

The pipeline supports both a headless Command Line Interface (CLI) for automated runs and a Streamlit Dashboard for interactive analysis.

### Option A: Streamlit Web Dashboard (Recommended)

Launch the interactive graphical interface to upload files, run the pipeline, and visualize findings.

```bash
streamlit run app.py
```

*The dashboard will automatically open in your browser at `http://localhost:8501`.*

### Option B: Headless CLI (Batch Processing)

Execute the pipeline natively in the terminal. Output is automatically saved to `data/output/`.

```bash
python main.py
```

---

## 🧪 Testing

This project maintains a robust test suite to protect the actuarial math, scraping logic, and inter-module data handoffs.

**Run the full test suite (Unit & Integration):**

```bash
python -m pytest tests/ -v
```

**Run ONLY Module/Unit Tests (Skip Integration):**

```bash
python -m pytest tests/ --ignore=tests/test_integration.py -v
```

**Run ONLY Integration Tests:**

```bash
python -m pytest tests/test_integration.py -v
```

---

## 🤝 Team Collaboration

We strictly enforce modular code ownership to prevent merge conflicts and pipeline regressions.

* Do not modify `src/pipeline.py` without team consensus.
* **Never commit output `.xlsx` files** or cache directories to version control.
