#!/usr/bin/env python3
"""
Bengali TTS Dataset Optimizer

Optimizes and curates Bengali text data for high-quality TTS training.
Creates a fluent, phoneme-balanced dataset optimized for neural TTS models.

Key optimizations:
1. Phoneme coverage analysis - ensures all Bengali sounds are well represented
2. Sentence length optimization - balances short and long utterances
3. Prosody diversity - varied sentence types (statements, questions, etc.)
4. Text normalization - consistent formatting and cleanup
5. Deduplication - removes exact and near duplicates
6. Quality filtering - removes problematic or incomplete sentences

For RTX 5060 Ti 16GB:
- Recommended dataset: 5,000-10,000 sentences (6-12 hours of audio)
- Sentence length: 5-50 characters (0.5-5 seconds each)
- Balance between short phrases and longer sentences
"""

import os
import re
import json
import csv
import unicodedata
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Set, Optional
from dataclasses import dataclass, field
from pathlib import Path
import hashlib

# ============================================================================
# BENGALI PHONEME INVENTORY
# ============================================================================

# Vowels (স্বরবর্ণ)
VOWELS = ['অ', 'আ', 'ই', 'ঈ', 'উ', 'ঊ', 'ঋ', 'এ', 'ঐ', 'ও', 'ঔ']

# Consonants (ব্যঞ্জনবর্ণ)
CONSONANTS = [
    'ক', 'খ', 'গ', 'ঘ', 'ঙ',
    'চ', 'ছ', 'জ', 'ঝ', 'ঞ',
    'ট', 'ঠ', 'ড', 'ঢ', 'ণ',
    'ত', 'থ', 'দ', 'ধ', 'ন',
    'প', 'ফ', 'ব', 'ভ', 'ম',
    'য', 'র', 'ল',
    'শ', 'ষ', 'স', 'হ',
    'ড়', 'ঢ়', 'য়'
]

# Matras (dependent vowels)
MATRAS = ['া', 'ি', 'ী', 'ু', 'ূ', 'ৃ', 'ে', 'ৈ', 'ো', 'ৌ']

# Special characters
SPECIAL_CHARS = ['ং', 'ঃ', 'ঁ', '্', 'ৎ']

# Common conjuncts that should be well represented
IMPORTANT_CONJUNCTS = [
    'ক্ত', 'ক্ষ', 'ক্র', 'ক্ল', 'গ্র', 'ঙ্ক', 'ঙ্গ',
    'চ্ছ', 'জ্ঞ', 'ঞ্চ', 'ঞ্জ',
    'ট্ট', 'ণ্ড', 'ণ্ট', 'ন্ত', 'ন্দ', 'ন্ধ', 'ন্ন',
    'প্ত', 'প্র', 'ব্দ', 'ব্র', 'ভ্র', 'ম্ব', 'ম্প',
    'ল্ক', 'ল্প', 'ল্ল', 'শ্চ', 'শ্র', 'ষ্ট', 'ষ্ঠ',
    'স্ক', 'স্ত', 'স্থ', 'স্ন', 'স্প', 'স্ম', 'স্র',
    'হ্ন', 'হ্ম', 'হ্য', 'হ্র',
    'র্ক', 'র্ত', 'র্থ', 'র্ন', 'র্ম', 'র্য', 'র্স',
    'ত্র', 'দ্র', 'ন্ত্র', 'স্ত্র', 'ষ্ট্র'
]

# Sentences to ensure complete phoneme coverage
# These are added to guarantee all Bengali sounds are represented
COVERAGE_SENTENCES = [
    # Rare vowels
    "এই ঔষধ খেলে আপনি ভালো হয়ে যাবেন।",  # ঔ vowel
    "ঔষধালয় থেকে ঔষধ কিনে আনো।",  # ঔ vowel
    "ঔৎসুক্যের কারণে সে এগিয়ে গেল।",  # ঔ vowel  
    "সৌন্দর্যের প্রতি মানুষের আকর্ষণ স্বাভাবিক।",  # ৌ matra (ou sound)
    "গৌরবময় ইতিহাস আমাদের অনুপ্রেরণা দেয়।",  # ৌ matra
    # Important conjuncts for natural speech
    "ক্ষমা করবেন, আমি একটু দেরি করে ফেললাম।",
    "বিশ্ববিদ্যালয়ে পড়াশোনা করতে চাই আমি।",
    "প্রথম শ্রেণিতে পড়ে আমাদের ছেলে।",
    "সত্যিকারের বন্ধুত্ব অমূল্য সম্পদ।",
    "স্বপ্ন দেখতে ভালোবাসি আমি সবসময়।",
]

# All Bengali characters for validation
BENGALI_RANGE = set(chr(i) for i in range(0x0980, 0x09FF + 1))
ALLOWED_CHARS = BENGALI_RANGE | set(' ।,!?\'"-():;০১২৩৪৫৬৭৮৯')

# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class Sentence:
    """Represents a processed sentence with metadata."""
    id: str
    text: str
    original_text: str
    source: str
    char_count: int = 0
    word_count: int = 0
    phoneme_coverage: Set[str] = field(default_factory=set)
    has_conjuncts: List[str] = field(default_factory=list)
    sentence_type: str = 'statement'  # statement, question, exclamation
    style: str = ''  # call_center, conversational, literary, etc.
    quality_score: float = 0.0
    
    def __post_init__(self):
        self.char_count = len(self.text)
        self.word_count = len(self.text.split())
        self.analyze_phonemes()
        self.detect_sentence_type()
    
    def analyze_phonemes(self):
        """Analyze phoneme coverage in this sentence."""
        self.phoneme_coverage = set()
        text = self.text
        
        # Find vowels
        for v in VOWELS:
            if v in text:
                self.phoneme_coverage.add(f'vowel:{v}')
        
        # Find consonants
        for c in CONSONANTS:
            if c in text:
                self.phoneme_coverage.add(f'consonant:{c}')
        
        # Find matras
        for m in MATRAS:
            if m in text:
                self.phoneme_coverage.add(f'matra:{m}')
        
        # Find special chars
        for s in SPECIAL_CHARS:
            if s in text:
                self.phoneme_coverage.add(f'special:{s}')
        
        # Find conjuncts
        self.has_conjuncts = []
        for conj in IMPORTANT_CONJUNCTS:
            if conj in text:
                self.has_conjuncts.append(conj)
                self.phoneme_coverage.add(f'conjunct:{conj}')
    
    def detect_sentence_type(self):
        """Detect if sentence is question, exclamation, or statement."""
        if '?' in self.text or self.text.endswith('?'):
            self.sentence_type = 'question'
        elif '!' in self.text or self.text.endswith('!'):
            self.sentence_type = 'exclamation'
        else:
            self.sentence_type = 'statement'
    
    def __hash__(self):
        return hash(self.text)
    
    def __eq__(self, other):
        return self.text == other.text


@dataclass
class DatasetStats:
    """Statistics about the dataset."""
    total_sentences: int = 0
    total_characters: int = 0
    total_words: int = 0
    avg_sentence_length: float = 0.0
    phoneme_coverage: Dict[str, int] = field(default_factory=dict)
    conjunct_coverage: Dict[str, int] = field(default_factory=dict)
    sentence_types: Dict[str, int] = field(default_factory=dict)
    length_distribution: Dict[str, int] = field(default_factory=dict)


# ============================================================================
# TEXT NORMALIZATION FUNCTIONS
# ============================================================================

def normalize_unicode(text: str) -> str:
    """Normalize Unicode to NFC form."""
    text = unicodedata.normalize('NFC', text)
    
    # Remove zero-width characters
    text = text.replace('\u200c', '')  # ZWNJ
    text = text.replace('\u200b', '')  # ZWS
    text = text.replace('\ufeff', '')  # BOM
    text = text.replace('\u00a0', ' ')  # NBSP
    
    return text


def normalize_text(text: str) -> str:
    """
    Normalize Bengali text for TTS training.
    """
    # Unicode normalization
    text = normalize_unicode(text)
    
    # Remove line numbers at start (e.g., "123,")
    text = re.sub(r'^\d+[,.]?\s*', '', text)
    
    # Remove leading/trailing whitespace
    text = text.strip()
    
    # Collapse multiple spaces
    text = re.sub(r'\s+', ' ', text)
    
    # Normalize quotes
    text = re.sub(r'["""]', '"', text)
    text = re.sub(r"[''']", "'", text)
    
    # Normalize dashes
    text = re.sub(r'[–—]', '-', text)
    
    # Fix double punctuation
    text = re.sub(r'।।+', '।', text)
    text = re.sub(r'\.\.+', '।', text)
    text = re.sub(r'\?\?+', '?', text)
    text = re.sub(r'!!+', '!', text)
    
    # Remove trailing punctuation duplicates
    text = re.sub(r'([।.!?])[।.!?]+$', r'\1', text)
    
    return text


def is_valid_bengali(text: str) -> bool:
    """
    Check if text contains primarily valid Bengali characters.
    Returns True if at least 70% of characters are Bengali.
    """
    if not text:
        return False
    
    bengali_count = sum(1 for c in text if c in BENGALI_RANGE or c in ' ।,!?\'"-():;০১২৩৪৫৬৭৮৯\n')
    total = len(text)
    
    return bengali_count / total >= 0.7 if total > 0 else False


def remove_english(text: str) -> str:
    """Remove English words but keep the sentence structure."""
    # Remove English words (keeping Bengali)
    text = re.sub(r'[a-zA-Z]+', '', text)
    # Clean up extra spaces
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def get_text_hash(text: str) -> str:
    """Get a hash for deduplication."""
    # Normalize for comparison
    normalized = re.sub(r'\s+', '', text.lower())
    return hashlib.md5(normalized.encode('utf-8')).hexdigest()[:12]


# ============================================================================
# QUALITY SCORING
# ============================================================================

def calculate_quality_score(sentence: Sentence) -> float:
    """
    Calculate a quality score for ranking sentences.
    Higher scores are better.
    
    Factors:
    - Optimal length (15-40 chars preferred)
    - Phoneme diversity
    - Contains conjuncts (bonus)
    - Complete sentence structure
    - Variety of sentence types
    """
    score = 0.0
    text = sentence.text
    char_count = sentence.char_count
    
    # 1. Length scoring (max 30 points)
    # Optimal range: 15-50 characters for TTS
    if 15 <= char_count <= 50:
        score += 30
    elif 10 <= char_count <= 60:
        score += 20
    elif 5 <= char_count <= 80:
        score += 10
    else:
        score += 5
    
    # 2. Phoneme diversity (max 30 points)
    phoneme_count = len(sentence.phoneme_coverage)
    score += min(phoneme_count * 1.5, 30)
    
    # 3. Conjunct bonus (max 15 points)
    conjunct_count = len(sentence.has_conjuncts)
    score += min(conjunct_count * 3, 15)
    
    # 4. Sentence completeness (max 15 points)
    # Ends with proper punctuation
    if text.endswith(('।', '.', '?', '!')):
        score += 10
    # Starts with capital Bengali char or valid start
    if text and text[0] in BENGALI_RANGE:
        score += 5
    
    # 5. Prosody variety bonus (max 10 points)
    if sentence.sentence_type == 'question':
        score += 10
    elif sentence.sentence_type == 'exclamation':
        score += 8
    else:
        score += 5
    
    return score


# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

def load_12k_txt(filepath: str) -> List[Sentence]:
    """Load sentences from 12k.txt format (number,text)."""
    sentences = []
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                # Parse line (format: number,text)
                match = re.match(r'^\d+[,.]?\s*(.+)$', line)
                if match:
                    text = match.group(1).strip()
                else:
                    text = line
                
                # Normalize
                text = normalize_text(text)
                
                # Validate
                if text and is_valid_bengali(text) and len(text) >= 3:
                    sent = Sentence(
                        id=f"12k_{line_num:05d}",
                        text=text,
                        original_text=line,
                        source="12k.txt",
                        style="literary"
                    )
                    sentences.append(sent)
    
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
    
    return sentences


def load_jsonl_metadata(filepath: str) -> List[Sentence]:
    """Load sentences from JSONL metadata format."""
    sentences = []
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    data = json.loads(line)
                    text = data.get('sentence', '').strip()
                    style = data.get('style', 'unknown')
                    category = data.get('category', 'unknown')
                    orig_id = data.get('id', str(line_num))
                    
                    # Normalize
                    text = normalize_text(text)
                    
                    # Validate
                    if text and is_valid_bengali(text) and len(text) >= 3:
                        sent = Sentence(
                            id=f"meta_{orig_id}",
                            text=text,
                            original_text=line,
                            source="bengali_sentences_6000_metadata.jsonl",
                            style=style
                        )
                        sentences.append(sent)
                except json.JSONDecodeError:
                    continue
    
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
    
    return sentences


def load_prompts_csv(filepath: str) -> List[Sentence]:
    """Load sentences from prompts.csv format."""
    sentences = []
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                text = row.get('text', '').strip()
                prompt_type = row.get('type', 'unknown')
                prompt_id = row.get('prompt_id', '')
                
                # Normalize
                text = normalize_text(text)
                
                # Validate - prompts can be short (single chars)
                if text and is_valid_bengali(text):
                    sent = Sentence(
                        id=f"prompt_{prompt_id}",
                        text=text,
                        original_text=row.get('text', ''),
                        source="prompts.csv",
                        style=f"phoneme_{prompt_type}"
                    )
                    sentences.append(sent)
    
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
    
    return sentences


# ============================================================================
# DATASET OPTIMIZATION
# ============================================================================

def deduplicate_sentences(sentences: List[Sentence]) -> List[Sentence]:
    """Remove duplicate sentences."""
    seen_hashes = set()
    unique_sentences = []
    
    for sent in sentences:
        text_hash = get_text_hash(sent.text)
        if text_hash not in seen_hashes:
            seen_hashes.add(text_hash)
            unique_sentences.append(sent)
    
    return unique_sentences


def filter_by_quality(sentences: List[Sentence], min_score: float = 20.0) -> List[Sentence]:
    """Filter sentences by minimum quality score."""
    return [s for s in sentences if s.quality_score >= min_score]


def filter_by_length(sentences: List[Sentence], 
                     min_chars: int = 3, 
                     max_chars: int = 150) -> List[Sentence]:
    """Filter sentences by character count."""
    return [s for s in sentences if min_chars <= s.char_count <= max_chars]


def ensure_phoneme_coverage(sentences: List[Sentence], 
                           target_coverage: int = 10) -> List[Sentence]:
    """
    Select sentences to ensure good phoneme coverage.
    Returns sentences sorted by their contribution to overall coverage.
    """
    # Track what's covered
    covered_phonemes = Counter()
    selected = []
    remaining = sentences.copy()
    
    # First pass: select sentences that cover rare phonemes
    while remaining:
        best_sent = None
        best_new_coverage = 0
        
        for sent in remaining:
            new_phonemes = len([p for p in sent.phoneme_coverage 
                              if covered_phonemes[p] < target_coverage])
            if new_phonemes > best_new_coverage:
                best_new_coverage = new_phonemes
                best_sent = sent
        
        if best_sent is None or best_new_coverage == 0:
            # No more new coverage, add remaining by score
            selected.extend(sorted(remaining, key=lambda x: -x.quality_score))
            break
        
        selected.append(best_sent)
        remaining.remove(best_sent)
        for p in best_sent.phoneme_coverage:
            covered_phonemes[p] += 1
    
    return selected


def balance_sentence_types(sentences: List[Sentence]) -> List[Sentence]:
    """Balance the distribution of sentence types."""
    by_type = defaultdict(list)
    
    for sent in sentences:
        by_type[sent.sentence_type].append(sent)
    
    # Calculate target counts
    total = len(sentences)
    target_statement = int(total * 0.60)  # 60% statements
    target_question = int(total * 0.25)   # 25% questions
    target_exclamation = int(total * 0.15)  # 15% exclamations
    
    result = []
    
    # Add statements
    statements = sorted(by_type['statement'], key=lambda x: -x.quality_score)
    result.extend(statements[:target_statement])
    
    # Add questions
    questions = sorted(by_type['question'], key=lambda x: -x.quality_score)
    result.extend(questions[:target_question])
    
    # Add exclamations
    exclamations = sorted(by_type['exclamation'], key=lambda x: -x.quality_score)
    result.extend(exclamations[:target_exclamation])
    
    # Fill remaining with highest scored from any type
    remaining = (statements[target_statement:] + 
                questions[target_question:] + 
                exclamations[target_exclamation:])
    remaining_sorted = sorted(remaining, key=lambda x: -x.quality_score)
    result.extend(remaining_sorted[:total - len(result)])
    
    return result


def balance_sentence_lengths(sentences: List[Sentence]) -> List[Sentence]:
    """Balance the distribution of sentence lengths."""
    # Categorize by length
    short = [s for s in sentences if s.char_count <= 20]      # 1-20 chars
    medium = [s for s in sentences if 20 < s.char_count <= 50]  # 21-50 chars
    long = [s for s in sentences if s.char_count > 50]         # 50+ chars
    
    total = len(sentences)
    
    # Target distribution (TTS works best with varied lengths)
    target_short = int(total * 0.25)   # 25% short
    target_medium = int(total * 0.50)  # 50% medium (preferred)
    target_long = int(total * 0.25)    # 25% long
    
    result = []
    
    # Sort each category by score
    short = sorted(short, key=lambda x: -x.quality_score)
    medium = sorted(medium, key=lambda x: -x.quality_score)
    long = sorted(long, key=lambda x: -x.quality_score)
    
    # Add from each category
    result.extend(short[:target_short])
    result.extend(medium[:target_medium])
    result.extend(long[:target_long])
    
    # Fill remaining
    remaining = short[target_short:] + medium[target_medium:] + long[target_long:]
    result.extend(sorted(remaining, key=lambda x: -x.quality_score)[:total - len(result)])
    
    return result


def select_optimal_dataset(sentences: List[Sentence], 
                          target_count: int = 8000) -> List[Sentence]:
    """
    Select optimal subset for TTS training.
    
    For RTX 5060 Ti 16GB:
    - Recommended: 5,000-10,000 sentences
    - Good quality: 8,000 sentences (~8-10 hours of audio)
    """
    print(f"\n🎯 Selecting optimal {target_count} sentences...")
    
    # Step 1: Remove very short or very long sentences
    filtered = filter_by_length(sentences, min_chars=5, max_chars=120)
    print(f"   After length filter: {len(filtered)}")
    
    # Step 2: Calculate quality scores
    for sent in filtered:
        sent.quality_score = calculate_quality_score(sent)
    
    # Step 3: Filter by quality score
    quality_filtered = filter_by_quality(filtered, min_score=25.0)
    print(f"   After quality filter: {len(quality_filtered)}")
    
    # Step 4: Ensure phoneme coverage
    phoneme_covered = ensure_phoneme_coverage(quality_filtered, target_coverage=15)
    print(f"   After phoneme coverage: {len(phoneme_covered)}")
    
    # Step 5: Balance sentence types
    type_balanced = balance_sentence_types(phoneme_covered)
    
    # Step 6: Balance lengths
    length_balanced = balance_sentence_lengths(type_balanced)
    
    # Step 7: Final selection
    final = sorted(length_balanced, key=lambda x: -x.quality_score)[:target_count]
    
    print(f"   Final selection: {len(final)}")
    
    return final


# ============================================================================
# STATISTICS AND REPORTING
# ============================================================================

def calculate_dataset_stats(sentences: List[Sentence]) -> DatasetStats:
    """Calculate comprehensive statistics for the dataset."""
    stats = DatasetStats()
    
    stats.total_sentences = len(sentences)
    stats.total_characters = sum(s.char_count for s in sentences)
    stats.total_words = sum(s.word_count for s in sentences)
    stats.avg_sentence_length = stats.total_characters / stats.total_sentences if sentences else 0
    
    # Phoneme coverage
    for sent in sentences:
        for phoneme in sent.phoneme_coverage:
            stats.phoneme_coverage[phoneme] = stats.phoneme_coverage.get(phoneme, 0) + 1
    
    # Conjunct coverage
    for sent in sentences:
        for conj in sent.has_conjuncts:
            stats.conjunct_coverage[conj] = stats.conjunct_coverage.get(conj, 0) + 1
    
    # Sentence types
    for sent in sentences:
        stats.sentence_types[sent.sentence_type] = stats.sentence_types.get(sent.sentence_type, 0) + 1
    
    # Length distribution
    for sent in sentences:
        if sent.char_count <= 10:
            bucket = '1-10'
        elif sent.char_count <= 20:
            bucket = '11-20'
        elif sent.char_count <= 30:
            bucket = '21-30'
        elif sent.char_count <= 50:
            bucket = '31-50'
        elif sent.char_count <= 80:
            bucket = '51-80'
        else:
            bucket = '80+'
        stats.length_distribution[bucket] = stats.length_distribution.get(bucket, 0) + 1
    
    return stats


def print_stats(stats: DatasetStats, title: str = "Dataset Statistics"):
    """Print formatted statistics."""
    print(f"\n{'='*60}")
    print(f"📊 {title}")
    print(f"{'='*60}")
    
    print(f"\n📝 Overview:")
    print(f"   Total sentences: {stats.total_sentences:,}")
    print(f"   Total characters: {stats.total_characters:,}")
    print(f"   Total words: {stats.total_words:,}")
    print(f"   Average sentence length: {stats.avg_sentence_length:.1f} chars")
    
    # Estimate audio duration (avg 5 chars/sec for Bengali)
    est_duration_sec = stats.total_characters / 5
    est_hours = est_duration_sec / 3600
    print(f"   Estimated audio duration: {est_hours:.1f} hours")
    
    print(f"\n📍 Sentence Types:")
    for stype, count in sorted(stats.sentence_types.items()):
        pct = (count / stats.total_sentences * 100) if stats.total_sentences > 0 else 0
        print(f"   {stype}: {count:,} ({pct:.1f}%)")
    
    print(f"\n📏 Length Distribution:")
    for bucket in ['1-10', '11-20', '21-30', '31-50', '51-80', '80+']:
        count = stats.length_distribution.get(bucket, 0)
        pct = (count / stats.total_sentences * 100) if stats.total_sentences > 0 else 0
        bar = '█' * int(pct / 2)
        print(f"   {bucket:>6} chars: {count:>5,} ({pct:>5.1f}%) {bar}")
    
    # Phoneme coverage summary
    vowel_coverage = len([p for p in stats.phoneme_coverage if p.startswith('vowel:')])
    consonant_coverage = len([p for p in stats.phoneme_coverage if p.startswith('consonant:')])
    matra_coverage = len([p for p in stats.phoneme_coverage if p.startswith('matra:')])
    conjunct_coverage = len([p for p in stats.phoneme_coverage if p.startswith('conjunct:')])
    
    print(f"\n🔤 Phoneme Coverage:")
    print(f"   Vowels: {vowel_coverage}/{len(VOWELS)} ({vowel_coverage/len(VOWELS)*100:.0f}%)")
    print(f"   Consonants: {consonant_coverage}/{len(CONSONANTS)} ({consonant_coverage/len(CONSONANTS)*100:.0f}%)")
    print(f"   Matras: {matra_coverage}/{len(MATRAS)} ({matra_coverage/len(MATRAS)*100:.0f}%)")
    print(f"   Important Conjuncts: {conjunct_coverage}/{len(IMPORTANT_CONJUNCTS)}")
    
    # Show missing phonemes
    missing_vowels = [v for v in VOWELS if f'vowel:{v}' not in stats.phoneme_coverage]
    missing_consonants = [c for c in CONSONANTS if f'consonant:{c}' not in stats.phoneme_coverage]
    
    if missing_vowels:
        print(f"\n   ⚠️  Missing vowels: {', '.join(missing_vowels)}")
    if missing_consonants:
        print(f"   ⚠️  Missing consonants: {', '.join(missing_consonants)}")
    
    print(f"\n{'='*60}")


# ============================================================================
# OUTPUT FUNCTIONS
# ============================================================================

def save_optimized_dataset(sentences: List[Sentence], output_dir: str):
    """Save optimized dataset in multiple formats."""
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Save as simple text file (one sentence per line)
    txt_path = os.path.join(output_dir, 'optimized_sentences.txt')
    with open(txt_path, 'w', encoding='utf-8') as f:
        for sent in sentences:
            f.write(f"{sent.text}\n")
    print(f"   ✅ Saved: {txt_path}")
    
    # 2. Save as CSV with metadata
    csv_path = os.path.join(output_dir, 'optimized_metadata.csv')
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'text', 'char_count', 'word_count', 
                        'sentence_type', 'style', 'quality_score', 'source'])
        for i, sent in enumerate(sentences, 1):
            writer.writerow([
                f'{i:05d}',
                sent.text,
                sent.char_count,
                sent.word_count,
                sent.sentence_type,
                sent.style,
                f'{sent.quality_score:.1f}',
                sent.source
            ])
    print(f"   ✅ Saved: {csv_path}")
    
    # 3. Save as JSONL for training
    jsonl_path = os.path.join(output_dir, 'optimized_training.jsonl')
    with open(jsonl_path, 'w', encoding='utf-8') as f:
        for i, sent in enumerate(sentences, 1):
            record = {
                'id': f'{i:05d}',
                'sentence': sent.text,
                'char_count': sent.char_count,
                'word_count': sent.word_count,
                'sentence_type': sent.sentence_type,
                'style': sent.style,
                'quality_score': sent.quality_score,
                'has_conjuncts': sent.has_conjuncts
            }
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    print(f"   ✅ Saved: {jsonl_path}")
    
    # 4. Save recording prompts (LJSpeech format)
    prompts_path = os.path.join(output_dir, 'recording_prompts.csv')
    with open(prompts_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['prompt_id', 'text', 'status'])
        for i, sent in enumerate(sentences, 1):
            writer.writerow([f'{i:05d}', sent.text, 'pending'])
    print(f"   ✅ Saved: {prompts_path}")
    
    # 5. Save statistics report
    stats = calculate_dataset_stats(sentences)
    report_path = os.path.join(output_dir, 'dataset_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("Bengali TTS Optimized Dataset Report\n")
        f.write("="*60 + "\n\n")
        
        f.write("OVERVIEW\n")
        f.write("-"*30 + "\n")
        f.write(f"Total sentences: {stats.total_sentences:,}\n")
        f.write(f"Total characters: {stats.total_characters:,}\n")
        f.write(f"Total words: {stats.total_words:,}\n")
        f.write(f"Average sentence length: {stats.avg_sentence_length:.1f} chars\n")
        
        est_hours = stats.total_characters / 5 / 3600
        f.write(f"Estimated audio duration: {est_hours:.1f} hours\n\n")
        
        f.write("SENTENCE TYPES\n")
        f.write("-"*30 + "\n")
        for stype, count in sorted(stats.sentence_types.items()):
            pct = count / stats.total_sentences * 100
            f.write(f"{stype}: {count:,} ({pct:.1f}%)\n")
        
        f.write("\nLENGTH DISTRIBUTION\n")
        f.write("-"*30 + "\n")
        for bucket in ['1-10', '11-20', '21-30', '31-50', '51-80', '80+']:
            count = stats.length_distribution.get(bucket, 0)
            pct = count / stats.total_sentences * 100
            f.write(f"{bucket} chars: {count:,} ({pct:.1f}%)\n")
        
        f.write("\nCONJUNCT COVERAGE\n")
        f.write("-"*30 + "\n")
        for conj, count in sorted(stats.conjunct_coverage.items(), key=lambda x: -x[1])[:20]:
            f.write(f"{conj}: {count}\n")
    
    print(f"   ✅ Saved: {report_path}")


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """Main optimization pipeline."""
    print("="*60)
    print("🚀 Bengali TTS Dataset Optimizer")
    print("="*60)
    
    # Paths
    base_dir = Path(__file__).parent
    text_set_dir = base_dir / 'text set'
    output_dir = base_dir / 'optimized_dataset'
    
    # Load all data sources
    all_sentences = []
    
    print("\n📥 Loading data sources...")
    
    # Load 12k.txt
    path_12k = text_set_dir / '12k.txt'
    if path_12k.exists():
        sentences_12k = load_12k_txt(str(path_12k))
        print(f"   12k.txt: {len(sentences_12k):,} sentences loaded")
        all_sentences.extend(sentences_12k)
    
    # Load bengali_sentences_6000_metadata.jsonl
    path_jsonl = text_set_dir / 'bengali_sentences_6000_metadata.jsonl'
    if path_jsonl.exists():
        sentences_jsonl = load_jsonl_metadata(str(path_jsonl))
        print(f"   bengali_sentences_6000_metadata.jsonl: {len(sentences_jsonl):,} sentences loaded")
        all_sentences.extend(sentences_jsonl)
    
    # Load prompts.csv
    path_prompts = text_set_dir / 'prompts.csv'
    if path_prompts.exists():
        sentences_prompts = load_prompts_csv(str(path_prompts))
        print(f"   prompts.csv: {len(sentences_prompts):,} prompts loaded")
        all_sentences.extend(sentences_prompts)
    
    # Add coverage sentences for rare phonemes
    print(f"   Adding {len(COVERAGE_SENTENCES)} coverage sentences for rare phonemes...")
    for i, text in enumerate(COVERAGE_SENTENCES, 1):
        sent = Sentence(
            id=f"coverage_{i:03d}",
            text=text,
            original_text=text,
            source="coverage_sentences",
            style="balanced"
        )
        all_sentences.append(sent)
    
    print(f"\n📊 Total loaded: {len(all_sentences):,} entries")
    
    # Calculate initial stats
    initial_stats = calculate_dataset_stats(all_sentences)
    print_stats(initial_stats, "Initial Dataset Statistics")
    
    # Deduplicate
    print("\n🔄 Deduplicating...")
    unique_sentences = deduplicate_sentences(all_sentences)
    print(f"   Unique sentences: {len(unique_sentences):,}")
    print(f"   Duplicates removed: {len(all_sentences) - len(unique_sentences):,}")
    
    # Select optimal dataset
    # For RTX 5060 Ti 16GB, recommend 8000 sentences (~8-10 hours)
    optimal_sentences = select_optimal_dataset(unique_sentences, target_count=8000)
    
    # Calculate final stats
    final_stats = calculate_dataset_stats(optimal_sentences)
    print_stats(final_stats, "Optimized Dataset Statistics")
    
    # Save outputs
    print("\n💾 Saving optimized dataset...")
    save_optimized_dataset(optimal_sentences, str(output_dir))
    
    # Summary
    print("\n" + "="*60)
    print("✅ Dataset Optimization Complete!")
    print("="*60)
    print(f"\n📁 Output directory: {output_dir}")
    print(f"📝 Total optimized sentences: {len(optimal_sentences):,}")
    print(f"⏱️  Estimated recording time: {final_stats.total_characters / 5 / 3600:.1f} hours")
    print("\n💡 Next steps:")
    print("   1. Review 'optimized_sentences.txt' for the final text")
    print("   2. Use 'recording_prompts.csv' with your recording app")
    print("   3. Check 'dataset_report.txt' for detailed statistics")
    print("   4. Run: python app.py to start recording")
    print("\n🎯 For RTX 5060 Ti 16GB:")
    print("   - This dataset size is optimized for ~8-10 hours of audio")
    print("   - Expected training time: 24-48 hours")
    print("   - Recommended batch size: 16 with gradient accumulation 4")


if __name__ == '__main__':
    main()
