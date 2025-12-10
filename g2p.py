#!/usr/bin/env python3
"""
Bengali Grapheme-to-Phoneme (G2P) Engine

Converts Bengali text to phoneme sequences for TTS training and inference.
Handles vowels, consonants, matras, conjuncts (juktakkhor), and special cases.
"""

import re
from typing import List, Dict, Tuple

# ============================================================================
# PHONEME MAPPINGS
# ============================================================================

# Vowels (স্বরবর্ণ) -> Phonemes
VOWEL_MAP: Dict[str, str] = {
    'অ': 'ɔ',      # o as in "hot"
    'আ': 'a',      # a as in "father"
    'ই': 'i',      # i as in "bit"
    'ঈ': 'iː',     # long i
    'উ': 'u',      # u as in "put"
    'ঊ': 'uː',     # long u
    'ঋ': 'ri',     # ri
    'এ': 'e',      # e as in "bed"
    'ঐ': 'oi',     # oi diphthong
    'ও': 'o',      # o as in "go"
    'ঔ': 'ou',     # ou diphthong
}

# Consonants (ব্যঞ্জনবর্ণ) -> Phonemes
CONSONANT_MAP: Dict[str, str] = {
    # Velars
    'ক': 'k',      # k
    'খ': 'kʰ',     # aspirated k
    'গ': 'g',      # g
    'ঘ': 'gʰ',     # aspirated g
    'ঙ': 'ŋ',      # ng
    
    # Palatals
    'চ': 'tʃ',     # ch
    'ছ': 'tʃʰ',    # aspirated ch
    'জ': 'dʒ',     # j
    'ঝ': 'dʒʰ',    # aspirated j
    'ঞ': 'n',      # palatal n (often just n)
    
    # Retroflexes
    'ট': 'ʈ',      # retroflex t
    'ঠ': 'ʈʰ',     # aspirated retroflex t
    'ড': 'ɖ',      # retroflex d
    'ঢ': 'ɖʰ',     # aspirated retroflex d
    'ণ': 'n',      # retroflex n (often just n)
    
    # Dentals
    'ত': 't',      # dental t
    'থ': 'tʰ',     # aspirated dental t
    'দ': 'd',      # dental d
    'ধ': 'dʰ',     # aspirated dental d
    'ন': 'n',      # dental n
    
    # Labials
    'প': 'p',      # p
    'ফ': 'pʰ',     # aspirated p (or f in loanwords)
    'ব': 'b',      # b
    'ভ': 'bʰ',     # aspirated b
    'ম': 'm',      # m
    
    # Semivowels and liquids
    'য': 'dʒ',     # j (when initial) / j (elsewhere)
    'র': 'r',      # r
    'ল': 'l',      # l
    
    # Sibilants
    'শ': 'ʃ',      # sh
    'ষ': 'ʃ',      # sh (same as শ in modern Bengali)
    'স': 's',      # s
    
    # Glottal
    'হ': 'h',      # h
    
    # Special consonants
    'ড়': 'ɽ',      # flap r
    'ঢ়': 'ɽʰ',     # aspirated flap
    'য়': 'e̯',      # semivowel y
    'ৎ': 't',      # final t (khanda ta)
}

# Matras (dependent vowel signs) -> Phonemes
MATRA_MAP: Dict[str, str] = {
    'া': 'a',      # aa-kar
    'ি': 'i',      # i-kar
    'ী': 'iː',     # dirgho i-kar
    'ু': 'u',      # u-kar
    'ূ': 'uː',     # dirgho u-kar
    'ৃ': 'ri',     # ri-kar
    'ে': 'e',      # e-kar
    'ৈ': 'oi',     # oi-kar
    'ো': 'o',      # o-kar
    'ৌ': 'ou',     # ou-kar
}

# Special characters
SPECIAL_MAP: Dict[str, str] = {
    'ং': 'ŋ',      # anusvara (nasal)
    'ঃ': 'h',      # visarga
    'ঁ': '̃',       # chandrabindu (nasalization)
    '্': '',       # hasanta/virama (no sound, suppresses inherent vowel)
}

# Common conjuncts (juktakkhor) with special pronunciations
CONJUNCT_MAP: Dict[str, str] = {
    'ক্ষ': 'kʰ',           # ksha -> kh (common pronunciation)
    'জ্ঞ': 'gːɔ',          # gya
    'ঞ্চ': 'ntʃ',          # nch
    'ঞ্জ': 'ndʒ',          # nj
    'ঙ্গ': 'ŋg',           # ng+g
    'ঙ্ক': 'ŋk',           # ng+k
    'ক্ক': 'kːɔ',          # geminate k
    'ত্ত': 'tːɔ',          # geminate t
    'দ্দ': 'dːɔ',          # geminate d
    'ন্ন': 'nːɔ',          # geminate n
    'প্প': 'pːɔ',          # geminate p
    'ম্ম': 'mːɔ',          # geminate m
    'ল্ল': 'lːɔ',          # geminate l
    'স্স': 'sːɔ',          # geminate s
}

# Inherent vowel (schwa)
INHERENT_VOWEL = 'ɔ'

# ============================================================================
# G2P FUNCTIONS
# ============================================================================

def normalize_text(text: str) -> str:
    """
    Normalize Bengali text before G2P conversion.
    - Remove punctuation
    - Normalize whitespace
    - Handle common variations
    """
    # Remove common punctuation (dash at end of character class to avoid range issues)
    text = re.sub(r'[।,;:!?"\'\(\)\[\]{}–—-]', ' ', text)
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def is_consonant(char: str) -> bool:
    """Check if character is a Bengali consonant."""
    return char in CONSONANT_MAP


def is_vowel(char: str) -> bool:
    """Check if character is a Bengali vowel."""
    return char in VOWEL_MAP


def is_matra(char: str) -> bool:
    """Check if character is a Bengali matra (dependent vowel)."""
    return char in MATRA_MAP


def is_hasanta(char: str) -> bool:
    """Check if character is hasanta (virama)."""
    return char == '্'


def bengali_g2p(text: str) -> List[str]:
    """
    Convert Bengali text to phoneme sequence.
    
    Args:
        text: Bengali text string
        
    Returns:
        List of phoneme strings
    """
    text = normalize_text(text)
    phonemes: List[str] = []
    i = 0
    
    while i < len(text):
        char = text[i]
        
        # Skip spaces, add word boundary
        if char == ' ':
            if phonemes and phonemes[-1] != ' ':
                phonemes.append(' ')
            i += 1
            continue
        
        # Check for conjuncts (2-3 character sequences)
        found_conjunct = False
        for length in [4, 3, 2]:
            if i + length <= len(text):
                seq = text[i:i+length]
                if seq in CONJUNCT_MAP:
                    phonemes.append(CONJUNCT_MAP[seq])
                    i += length
                    found_conjunct = True
                    break
        
        if found_conjunct:
            continue
        
        # Handle consonants
        if is_consonant(char):
            base_phoneme = CONSONANT_MAP[char]
            
            # Look ahead for matra or hasanta
            if i + 1 < len(text):
                next_char = text[i + 1]
                
                if is_matra(next_char):
                    # Consonant + matra
                    phonemes.append(base_phoneme)
                    phonemes.append(MATRA_MAP[next_char])
                    i += 2
                    continue
                    
                elif is_hasanta(next_char):
                    # Consonant + hasanta (no inherent vowel)
                    phonemes.append(base_phoneme)
                    i += 2
                    continue
            
            # Consonant with inherent vowel
            phonemes.append(base_phoneme)
            phonemes.append(INHERENT_VOWEL)
            i += 1
            continue
        
        # Handle independent vowels
        if is_vowel(char):
            phonemes.append(VOWEL_MAP[char])
            i += 1
            continue
        
        # Handle matras (shouldn't appear alone, but handle gracefully)
        if is_matra(char):
            phonemes.append(MATRA_MAP[char])
            i += 1
            continue
        
        # Handle special characters
        if char in SPECIAL_MAP:
            if SPECIAL_MAP[char]:
                phonemes.append(SPECIAL_MAP[char])
            i += 1
            continue
        
        # Unknown character - keep as is or skip
        if char.strip():
            phonemes.append(char)
        i += 1
    
    # Clean up: remove trailing spaces, merge consecutive spaces
    result = []
    for p in phonemes:
        if p == ' ':
            if result and result[-1] != ' ':
                result.append(p)
        else:
            result.append(p)
    
    if result and result[-1] == ' ':
        result.pop()
    
    return result


def phonemes_to_string(phonemes: List[str]) -> str:
    """Convert phoneme list to space-separated string."""
    return ' '.join(phonemes)


def g2p(text: str) -> str:
    """
    Main G2P function - converts Bengali text to phoneme string.
    
    Args:
        text: Bengali text
        
    Returns:
        Space-separated phoneme string
    """
    phonemes = bengali_g2p(text)
    return phonemes_to_string(phonemes)


# ============================================================================
# CLI & TESTING
# ============================================================================

def test_g2p():
    """Test G2P with example words."""
    test_cases = [
        ('ক', 'Single consonant'),
        ('কা', 'Consonant + aa-kar'),
        ('কি', 'Consonant + i-kar'),
        ('কী', 'Consonant + long i-kar'),
        ('কু', 'Consonant + u-kar'),
        ('কূ', 'Consonant + long u-kar'),
        ('কে', 'Consonant + e-kar'),
        ('কৈ', 'Consonant + oi-kar'),
        ('কো', 'Consonant + o-kar'),
        ('কৌ', 'Consonant + ou-kar'),
        ('অ', 'Vowel a'),
        ('আ', 'Vowel aa'),
        ('বাংলা', 'Word: Bangla'),
        ('আমি', 'Word: Ami (I)'),
        ('তুমি', 'Word: Tumi (You)'),
        ('ক্ষমা', 'Word with conjunct: Kshoma'),
        ('জ্ঞান', 'Word with conjunct: Gyan'),
        ('বিদ্যা', 'Word: Bidya'),
        ('আমার নাম কি', 'Sentence: Amar naam ki'),
    ]
    
    print("=" * 60)
    print("Bengali G2P Test Results")
    print("=" * 60)
    
    for text, description in test_cases:
        result = g2p(text)
        print(f"\n{description}")
        print(f"  Input:  {text}")
        print(f"  Output: {result}")
    
    print("\n" + "=" * 60)


if __name__ == '__main__':
    test_g2p()
