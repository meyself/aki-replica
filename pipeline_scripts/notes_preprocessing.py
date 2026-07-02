import re
from transformers import AutoTokenizer

def get_tokenizer():
    try:
        return AutoTokenizer.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased"), True
    except Exception as e:
        print(f"Could not load tokenizer: {e}. Using fallback whitespace tokenizer.")
        return None, False

def extract_sections(text):
    extracted_text = ""
    # Case-insensitive robust regexes for various headers
    patterns = {
        "Chief Complaint": r"(?:Chief Complaint|CHIEF COMPLAINT|Chief complaint):\s*(.*?)(?=\n[A-Z][a-z ]+:|\n[A-Z]+:|\Z)",
        "History of Present Illness": r"(?:History of Present Illness|HISTORY OF PRESENT ILLNESS|HPI):\s*(.*?)(?=\n[A-Z][a-z ]+:|\n[A-Z]+:|\Z)",
        "Past Medical History": r"(?:Past Medical History|PAST MEDICAL HISTORY|PMH):\s*(.*?)(?=\n[A-Z][a-z ]+:|\n[A-Z]+:|\Z)",
        "Physical Exam": r"(?:Physical Exam|PHYSICAL EXAMINATION|Pertinent Results|Admission Physical Exam|EXAM):\s*(.*?)(?=\n[A-Z][a-z ]+:|\n[A-Z]+:|\Z)"
    }
    
    matches = {}
    for section, pattern in patterns.items():
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            content = match.group(1).strip()
            matches[section] = content
            extracted_text += content + " "
        else:
            matches[section] = None
            
    return extracted_text, matches

def debug_note_extraction(raw_text: str) -> dict:
    _, matches = extract_sections(raw_text)
    
    print("\n--- DEBUG NOTE EXTRACTION ---")
    summary = {}
    for section, content in matches.items():
        if content:
            word_count = len(content.split())
            print(f"[{section}] MATCHED ({word_count} words):\n  {content[:100]}...")
            summary[section] = {'matched': True, 'words': word_count}
        else:
            print(f"[{section}] FAILED TO MATCH")
            summary[section] = {'matched': False, 'words': 0}
    print("-----------------------------\n")
    return summary

def process_patient_notes(notes_df, tokenizer, has_real_tokenizer, max_len=512):
    if notes_df.empty:
        return []
        
    text = notes_df.iloc[0]['text']
    extracted_text, _ = extract_sections(text)
    
    cleaned_text = extracted_text.lower().replace('\n', ' ')
    cleaned_text = re.sub(r'_{3,}', '[DEIDENTIFIED]', cleaned_text)
    
    if has_real_tokenizer:
        tokens = tokenizer.encode(cleaned_text, add_special_tokens=True, truncation=False)
    else:
        tokens = cleaned_text.split()
        
    chunks = [tokens[i:i + max_len] for i in range(0, len(tokens), max_len)]
    return chunks

if __name__ == "__main__":
    sample_text = """Chief Complaint:
 patient presents with decreased urine output.

History of Present Illness:
 ___ year old male with history of hypertension who presents with oliguria for 2 days.

Past Medical History:
 Hypertension
 CKD stage 2

Physical Exam:
 Vitals stable. Alert and oriented. No edema."""
 
    res = debug_note_extraction(sample_text)
    assert res["Chief Complaint"]["matched"]
    assert res["History of Present Illness"]["matched"]
    assert res["Past Medical History"]["matched"]
    assert res["Physical Exam"]["matched"]
    print("All sample sections matched successfully!")
