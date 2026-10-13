import pandas as pd
import hashlib
import re
from datetime import datetime, timezone
from transformers import pipeline

# --- CONFIGURATION & MULTIPLIERS ---
class CFG:
    # Local models ensuring data never leaves the environment
    zero_shot_model = "facebook/bart-large-mnli"
    llm_model = "Qwen/Qwen2.5-0.5B-Instruct" 
    
    # Lexical Multipliers (RegEx Rules)
    positive_regex = re.compile(r"(now available|general availability|shipped|powered by|built on)", re.IGNORECASE)
    negative_regex = re.compile(r"(defends against|protects against|future release|roadmap|marketing)", re.IGNORECASE)

# Initialize models (loads locally into memory)
print("Loading NLP models (this may take a minute)...")
classifier = pipeline("zero-shot-classification", model=CFG.zero_shot_model)
summarizer = pipeline("text-generation", model=CFG.llm_model)

def generate_hash(text):
    """Generate SHA-256 cryptographic provenance for audit defensibility."""
    if pd.isna(text):
        return None
    return hashlib.sha256(str(text).encode('utf-8')).hexdigest()

def apply_lexical_multipliers(text, base_score):
    """Adjust confidence score based on RegEx rules to filter hype/defensive language."""
    text = str(text)
    if CFG.negative_regex.search(text):
        return base_score * 0.20  # Heavy penalty for defensive/hype language
    if CFG.positive_regex.search(text):
        return min(base_score * 1.20, 1.0) # Boost for verified shipped features
    return base_score

def summarize_capability(text):
    """Generate a crisp 1-2 sentence summary using the local Qwen model."""
    prompt = f"Summarize the AI capability in one sentence: {text}"
    # Truncate input to avoid model token limits on long paragraphs
    result = summarizer(prompt[:1000], max_new_tokens=30, do_sample=False)
    return result[0]['generated_text'].replace(prompt[:1000], "").strip()

def analyze(df):
    """
    Main entry point for Stage 3: NLP Detection Engine.
    Expects an in-memory Pandas DataFrame from the Data Flattener.
    """
    if df.empty:
        return df
        
    print(f"Running NLP analysis on {len(df)} evidence excerpts...")
    
    results = []
    labels = ["Embedded AI", "Marketing Hype", "Defensive Security", "Legacy Automation", "Not AI-Related"]

    for _, row in df.iterrows():
        # Retrieve columns passed from src.pipeline.py
        excerpt = str(row.get('excerpt', ''))
        context = str(row.get('context', ''))
        
        # 1. Hashing for Audit Provenance
        content_sha256 = generate_hash(context)
        timestamp = datetime.now(timezone.utc).isoformat()

        # 2. Dual-View Fusion & Zero-Shot Classification
        combined_text = f"{excerpt} {context}" 
        
        if len(combined_text.strip()) > 10:
            classification = classifier(combined_text, candidate_labels=labels)
            
            # Extract base probability for 'Embedded AI'
            ai_index = classification['labels'].index("Embedded AI")
            base_score = classification['scores'][ai_index]
            
            # 3. Apply Lexical Multipliers
            final_score = apply_lexical_multipliers(combined_text, base_score)
            
            # 4. Local Generative Summary
            summary = "N/A"
            if final_score > 0.50:
                summary = summarize_capability(combined_text)
                
        else:
            final_score = 0.0
            summary = "Insufficient Evidence"
            
        # Append enriched data for Stage 4 (Risk Matrix)
        results.append({
            "vendor_id": row.get("vendor_id"),
            "vendor_name": row.get("vendor_name"),
            "source_url": row.get("source_url"),
            "P_Embedded_AI": final_score,
            "Capability_Summary": summary,
            "content_sha256": content_sha256,
            "timestamp": timestamp
        })

    return pd.DataFrame(results)