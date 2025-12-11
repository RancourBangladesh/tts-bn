#!/usr/bin/env python3
"""
Bengali Grapheme-to-Phoneme (G2P) Engine

Converts Bengali text to phoneme sequences for TTS training and inference.
Handles:
- All vowels (স্বরবর্ণ)
- All consonants (ব্যঞ্জনবর্ণ) with matras
- 150+ conjuncts (juktakkhor)
- Nasalization (chandrabindu)
- Reph (র-ফলা), Rafala, Jo-fola (য-ফলা)
- Number expansion (dates, years, currency, phone numbers)
- English→Bengali pronunciation fallback
- Text normalization (Unicode cleanup, punctuation)
"""

import re
import unicodedata
from typing import List, Dict, Tuple, Optional

# ============================================================================
# PHONEME MAPPINGS - COMPLETE BANGLADESHI BENGALI
# ============================================================================

# Vowels (স্বরবর্ণ) -> Phonemes - 11 vowels
VOWEL_MAP: Dict[str, str] = {
    'অ': 'ɔ',      # o as in "hot" (inherent vowel sound)
    'আ': 'a',      # a as in "father"
    'ই': 'i',      # i as in "bit"
    'ঈ': 'iː',     # long i
    'উ': 'u',      # u as in "put"
    'ঊ': 'uː',     # long u
    'ঋ': 'ri',     # ri (vocalic r)
    'ৠ': 'riː',    # long ri (rare)
    'ঌ': 'li',     # vocalic l (rare)
    'এ': 'e',      # e as in "bed"
    'ঐ': 'oi',     # oi diphthong
    'ও': 'o',      # o as in "go"
    'ঔ': 'ou',     # ou diphthong
}

# Consonants (ব্যঞ্জনবর্ণ) -> Phonemes - 39 consonants
CONSONANT_MAP: Dict[str, str] = {
    # Velars (কণ্ঠ্য)
    'ক': 'k',      # k (voiceless velar stop)
    'খ': 'kʰ',     # aspirated k
    'গ': 'g',      # g (voiced velar stop)
    'ঘ': 'gʰ',     # aspirated g
    'ঙ': 'ŋ',      # ng (velar nasal)
    
    # Palatals (তালব্য)
    'চ': 'tʃ',     # ch (voiceless palatal affricate)
    'ছ': 'tʃʰ',    # aspirated ch
    'জ': 'dʒ',     # j (voiced palatal affricate)
    'ঝ': 'dʒʰ',    # aspirated j
    'ঞ': 'ɲ',      # palatal nasal (often pronounced as n)
    
    # Retroflexes (মূর্ধন্য)
    'ট': 'ʈ',      # retroflex t
    'ঠ': 'ʈʰ',     # aspirated retroflex t
    'ড': 'ɖ',      # retroflex d
    'ঢ': 'ɖʰ',     # aspirated retroflex d
    'ণ': 'ɳ',      # retroflex n (often pronounced as n)
    
    # Dentals (দন্ত্য)
    'ত': 't',      # dental t
    'থ': 'tʰ',     # aspirated dental t
    'দ': 'd',      # dental d
    'ধ': 'dʰ',     # aspirated dental d
    'ন': 'n',      # dental n
    
    # Labials (ওষ্ঠ্য)
    'প': 'p',      # p
    'ফ': 'pʰ',     # aspirated p (or f in loanwords)
    'ব': 'b',      # b
    'ভ': 'bʰ',     # aspirated b
    'ম': 'm',      # m
    
    # Semivowels and approximants (অন্তঃস্থ)
    'য': 'dʒ',     # j sound (when initial)
    'র': 'r',      # r (alveolar trill/tap)
    'ল': 'l',      # l (lateral approximant)
    
    # Sibilants (উষ্ম)
    'শ': 'ʃ',      # sh (voiceless postalveolar fricative)
    'ষ': 'ʃ',      # sh (same as শ in modern Bengali)
    'স': 's',      # s (voiceless alveolar fricative)
    
    # Glottal
    'হ': 'h',      # h (voiceless glottal fricative)
    
    # Special consonants
    'ড়': 'ɽ',      # flap r (retroflex flap)
    'ঢ়': 'ɽʰ',     # aspirated retroflex flap
    'য়': 'j',      # semivowel y
    'ৎ': 't',      # khanda ta (final t)
}

# Matras (dependent vowel signs - কার) -> Phonemes - 10 matras
MATRA_MAP: Dict[str, str] = {
    'া': 'a',      # আ-কার (aa-kar)
    'ি': 'i',      # ই-কার (hrasva i-kar)
    'ী': 'iː',     # ঈ-কার (dirgho i-kar)
    'ু': 'u',      # উ-কার (hrasva u-kar)
    'ূ': 'uː',     # ঊ-কার (dirgho u-kar)
    'ৃ': 'ri',     # ঋ-কার (ri-kar)
    'ে': 'e',      # এ-কার (e-kar)
    'ৈ': 'oi',     # ঐ-কার (oi-kar)
    'ো': 'o',      # ও-কার (o-kar, composite)
    'ৌ': 'ou',     # ঔ-কার (ou-kar, composite)
}

# Special characters and modifiers
SPECIAL_MAP: Dict[str, str] = {
    'ং': 'ŋ',      # অনুস্বার (anusvara) - nasal
    'ঃ': 'h',      # বিসর্গ (visarga)
    'ঁ': '̃',       # চন্দ্রবিন্দু (chandrabindu) - nasalization marker
    '্': '',       # হসন্ত/বিরাম (hasanta/virama) - suppresses inherent vowel
    'ঽ': '',       # অবগ্রহ (avagraha) - elision marker
}

# Extended conjuncts (juktakkhor) - 150+ common combinations
CONJUNCT_MAP: Dict[str, str] = {
    # ক-conjuncts
    'ক্ক': 'kːɔ',      # geminate k
    'ক্ট': 'kʈɔ',
    'ক্ত': 'ktɔ',
    'ক্ন': 'knɔ',
    'ক্ম': 'kmɔ',
    'ক্য': 'kjɔ',      # k + ya-phala
    'ক্র': 'krɔ',      # k + ra-phala
    'ক্ল': 'klɔ',
    'ক্ষ': 'kʰɔ',      # ksha -> kh (common Bangladeshi pronunciation)
    'ক্ষ্ণ': 'kʰnɔ',
    'ক্ষ্ম': 'kʰmɔ',
    'ক্স': 'ksɔ',
    
    # খ-conjuncts
    'খ্য': 'kʰjɔ',
    'খ্র': 'kʰrɔ',
    
    # গ-conjuncts
    'গ্গ': 'gːɔ',
    'গ্ধ': 'gdʰɔ',
    'গ্ন': 'gnɔ',
    'গ্ম': 'gmɔ',
    'গ্য': 'gjɔ',
    'গ্র': 'grɔ',
    'গ্ল': 'glɔ',
    
    # ঘ-conjuncts
    'ঘ্ন': 'gʰnɔ',
    'ঘ্য': 'gʰjɔ',
    'ঘ্র': 'gʰrɔ',
    
    # ঙ-conjuncts
    'ঙ্ক': 'ŋkɔ',
    'ঙ্খ': 'ŋkʰɔ',
    'ঙ্গ': 'ŋgɔ',
    'ঙ্ঘ': 'ŋgʰɔ',
    'ঙ্ম': 'ŋmɔ',
    
    # চ-conjuncts
    'চ্চ': 'tʃːɔ',
    'চ্ছ': 'tʃʰːɔ',
    'চ্ঞ': 'tʃɲɔ',
    'চ্য': 'tʃjɔ',
    
    # ছ-conjuncts
    'ছ্য': 'tʃʰjɔ',
    'ছ্র': 'tʃʰrɔ',
    
    # জ-conjuncts
    'জ্জ': 'dʒːɔ',
    'জ্ঝ': 'dʒdʒʰɔ',
    'জ্ঞ': 'gːɔ',      # gya (special pronunciation)
    'জ্য': 'dʒjɔ',
    'জ্র': 'dʒrɔ',
    
    # ঝ-conjuncts
    'ঝ্য': 'dʒʰjɔ',
    
    # ঞ-conjuncts
    'ঞ্চ': 'ntʃɔ',
    'ঞ্ছ': 'ntʃʰɔ',
    'ঞ্জ': 'ndʒɔ',
    'ঞ্ঝ': 'ndʒʰɔ',
    
    # ট-conjuncts
    'ট্ট': 'ʈːɔ',
    'ট্য': 'ʈjɔ',
    'ট্র': 'ʈrɔ',
    
    # ঠ-conjuncts
    'ঠ্য': 'ʈʰjɔ',
    
    # ড-conjuncts
    'ড্ড': 'ɖːɔ',
    'ড্য': 'ɖjɔ',
    'ড্র': 'ɖrɔ',
    
    # ঢ-conjuncts
    'ঢ্য': 'ɖʰjɔ',
    
    # ণ-conjuncts
    'ণ্ট': 'ɳʈɔ',
    'ণ্ঠ': 'ɳʈʰɔ',
    'ণ্ড': 'ɳɖɔ',
    'ণ্ঢ': 'ɳɖʰɔ',
    'ণ্ণ': 'ɳːɔ',
    'ণ্ম': 'ɳmɔ',
    'ণ্য': 'ɳjɔ',
    
    # ত-conjuncts
    'ত্ত': 'tːɔ',
    'ত্থ': 'ttʰɔ',
    'ত্ন': 'tnɔ',
    'ত্ম': 'tmɔ',
    'ত্য': 'tjɔ',
    'ত্র': 'trɔ',
    
    # থ-conjuncts
    'থ্য': 'tʰjɔ',
    'থ্র': 'tʰrɔ',
    
    # দ-conjuncts
    'দ্গ': 'dgɔ',
    'দ্ঘ': 'dgʰɔ',
    'দ্দ': 'dːɔ',
    'দ্ধ': 'dːʰɔ',
    'দ্ব': 'dbɔ',
    'দ্ভ': 'dbʰɔ',
    'দ্ম': 'dmɔ',
    'দ্য': 'djɔ',
    'দ্র': 'drɔ',
    
    # ধ-conjuncts
    'ধ্ন': 'dʰnɔ',
    'ধ্ম': 'dʰmɔ',
    'ধ্য': 'dʰjɔ',
    'ধ্র': 'dʰrɔ',
    
    # ন-conjuncts
    'ন্ট': 'nʈɔ',
    'ন্ঠ': 'nʈʰɔ',
    'ন্ড': 'nɖɔ',
    'ন্ঢ': 'nɖʰɔ',
    'ন্ত': 'ntɔ',
    'ন্থ': 'ntʰɔ',
    'ন্দ': 'ndɔ',
    'ন্ধ': 'ndʰɔ',
    'ন্ন': 'nːɔ',
    'ন্ম': 'nmɔ',
    'ন্য': 'njɔ',
    'ন্র': 'nrɔ',
    'ন্ব': 'nbɔ',
    
    # প-conjuncts
    'প্ট': 'pʈɔ',
    'প্ত': 'ptɔ',
    'প্ন': 'pnɔ',
    'প্প': 'pːɔ',
    'প্ম': 'pmɔ',
    'প্য': 'pjɔ',
    'প্র': 'prɔ',
    'প্ল': 'plɔ',
    'প্স': 'psɔ',
    
    # ফ-conjuncts
    'ফ্য': 'pʰjɔ',
    'ফ্র': 'pʰrɔ',
    'ফ্ল': 'pʰlɔ',
    
    # ব-conjuncts
    'ব্জ': 'bdʒɔ',
    'ব্দ': 'bdɔ',
    'ব্ধ': 'bdʰɔ',
    'ব্ব': 'bːɔ',
    'ব্য': 'bjɔ',
    'ব্র': 'brɔ',
    'ব্ল': 'blɔ',
    
    # ভ-conjuncts
    'ভ্য': 'bʰjɔ',
    'ভ্র': 'bʰrɔ',
    
    # ম-conjuncts
    'ম্ন': 'mnɔ',
    'ম্প': 'mpɔ',
    'ম্ফ': 'mpʰɔ',
    'ম্ব': 'mbɔ',
    'ম্ভ': 'mbʰɔ',
    'ম্ম': 'mːɔ',
    'ম্য': 'mjɔ',
    'ম্র': 'mrɔ',
    'ম্ল': 'mlɔ',
    
    # য-conjuncts (when য appears first)
    'য্য': 'jːɔ',
    
    # র-conjuncts (reph and ra-phala)
    'র্ক': 'rkɔ',
    'র্খ': 'rkʰɔ',
    'র্গ': 'rgɔ',
    'র্ঘ': 'rgʰɔ',
    'র্চ': 'rtʃɔ',
    'র্ছ': 'rtʃʰɔ',
    'র্জ': 'rdʒɔ',
    'র্ঝ': 'rdʒʰɔ',
    'র্ট': 'rʈɔ',
    'র্ঠ': 'rʈʰɔ',
    'র্ড': 'rɖɔ',
    'র্ঢ': 'rɖʰɔ',
    'র্ণ': 'rɳɔ',
    'র্ত': 'rtɔ',
    'র্থ': 'rtʰɔ',
    'র্দ': 'rdɔ',
    'র্ধ': 'rdʰɔ',
    'র্ন': 'rnɔ',
    'র্প': 'rpɔ',
    'র্ফ': 'rpʰɔ',
    'র্ব': 'rbɔ',
    'র্ভ': 'rbʰɔ',
    'র্ম': 'rmɔ',
    'র্য': 'rjɔ',
    'র্ল': 'rlɔ',
    'র্শ': 'rʃɔ',
    'র্ষ': 'rʃɔ',
    'র্স': 'rsɔ',
    'র্হ': 'rhɔ',
    
    # ল-conjuncts
    'ল্ক': 'lkɔ',
    'ল্গ': 'lgɔ',
    'ল্ট': 'lʈɔ',
    'ল্ড': 'lɖɔ',
    'ল্প': 'lpɔ',
    'ল্ফ': 'lpʰɔ',
    'ল্ব': 'lbɔ',
    'ল্ম': 'lmɔ',
    'ল্য': 'ljɔ',
    'ল্ল': 'lːɔ',
    
    # শ-conjuncts
    'শ্চ': 'ʃtʃɔ',
    'শ্ছ': 'ʃtʃʰɔ',
    'শ্ন': 'ʃnɔ',
    'শ্ব': 'ʃbɔ',
    'শ্ম': 'ʃmɔ',
    'শ্য': 'ʃjɔ',
    'শ্র': 'ʃrɔ',
    'শ্ল': 'ʃlɔ',
    
    # ষ-conjuncts
    'ষ্ক': 'ʃkɔ',
    'ষ্ট': 'ʃʈɔ',
    'ষ্ঠ': 'ʃʈʰɔ',
    'ষ্ণ': 'ʃɳɔ',
    'ষ্প': 'ʃpɔ',
    'ষ্ফ': 'ʃpʰɔ',
    'ষ্ম': 'ʃmɔ',
    'ষ্য': 'ʃjɔ',
    
    # স-conjuncts
    'স্ক': 'skɔ',
    'স্খ': 'skʰɔ',
    'স্ট': 'sʈɔ',
    'স্ত': 'stɔ',
    'স্থ': 'stʰɔ',
    'স্ন': 'snɔ',
    'স্প': 'spɔ',
    'স্ফ': 'spʰɔ',
    'স্ব': 'sbɔ',
    'স্ম': 'smɔ',
    'স্য': 'sjɔ',
    'স্র': 'srɔ',
    'স্ল': 'slɔ',
    'স্স': 'sːɔ',
    
    # হ-conjuncts
    'হ্ণ': 'hɳɔ',
    'হ্ন': 'hnɔ',
    'হ্ম': 'hmɔ',
    'হ্য': 'hjɔ',
    'হ্র': 'hrɔ',
    'হ্ল': 'hlɔ',
    
    # Three-consonant clusters
    'ক্ত্র': 'ktrɔ',
    'ক্ষ্ণ': 'kʰnɔ',
    'ক্ষ্ম': 'kʰmɔ',
    'ক্ষ্য': 'kʰjɔ',
    'ঙ্ক্ষ': 'ŋkʰɔ',
    'ঙ্ক্য': 'ŋkjɔ',
    'ঞ্জ্য': 'ndʒjɔ',
    'ণ্ট্র': 'ɳʈrɔ',
    'ণ্ড্র': 'ɳɖrɔ',
    'ন্ত্র': 'ntrɔ',
    'ন্ত্য': 'ntjɔ',
    'ন্দ্র': 'ndrɔ',
    'ন্দ্য': 'ndjɔ',
    'ন্ধ্র': 'ndʰrɔ',
    'ন্ধ্য': 'ndʰjɔ',
    'ম্প্র': 'mprɔ',
    'ম্ব্র': 'mbrɔ',
    'ষ্ট্র': 'ʃʈrɔ',
    'ষ্ঠ্য': 'ʃʈʰjɔ',
    'স্ত্র': 'strɔ',
    'স্ত্য': 'stjɔ',
    'স্থ্য': 'stʰjɔ',
    'স্প্র': 'sprɔ',
    'স্ক্র': 'skrɔ',
}

# Inherent vowel (schwa) in Bangladeshi Bengali
INHERENT_VOWEL = 'ɔ'

# ============================================================================
# NUMBER EXPANSION - Bengali numbers
# ============================================================================

# Bengali digits
BENGALI_DIGITS = {
    '০': 0, '১': 1, '২': 2, '৩': 3, '৪': 4,
    '৫': 5, '৬': 6, '৭': 7, '৮': 8, '৯': 9
}

# Number words
NUMBER_WORDS = {
    0: 'শূন্য', 1: 'এক', 2: 'দুই', 3: 'তিন', 4: 'চার',
    5: 'পাঁচ', 6: 'ছয়', 7: 'সাত', 8: 'আট', 9: 'নয়',
    10: 'দশ', 11: 'এগারো', 12: 'বারো', 13: 'তেরো', 14: 'চৌদ্দ',
    15: 'পনেরো', 16: 'ষোলো', 17: 'সতেরো', 18: 'আঠারো', 19: 'উনিশ',
    20: 'বিশ', 21: 'একুশ', 22: 'বাইশ', 23: 'তেইশ', 24: 'চব্বিশ',
    25: 'পঁচিশ', 26: 'ছাব্বিশ', 27: 'সাতাশ', 28: 'আটাশ', 29: 'ঊনত্রিশ',
    30: 'ত্রিশ', 40: 'চল্লিশ', 50: 'পঞ্চাশ', 60: 'ষাট', 70: 'সত্তর',
    80: 'আশি', 90: 'নব্বই', 100: 'একশো', 1000: 'এক হাজার',
    100000: 'এক লাখ', 10000000: 'এক কোটি'
}

# Currency
CURRENCY_WORDS = {
    '৳': 'টাকা',
    'টাকা': 'টাকা',
    '$': 'ডলার',
    '₹': 'রুপি',
}

# ============================================================================
# ENGLISH TO BENGALI PRONUNCIATION FALLBACK
# ============================================================================

ENGLISH_TO_BENGALI: Dict[str, str] = {
    'a': 'এ', 'b': 'বি', 'c': 'সি', 'd': 'ডি', 'e': 'ই',
    'f': 'এফ', 'g': 'জি', 'h': 'এইচ', 'i': 'আই', 'j': 'জে',
    'k': 'কে', 'l': 'এল', 'm': 'এম', 'n': 'এন', 'o': 'ও',
    'p': 'পি', 'q': 'কিউ', 'r': 'আর', 's': 'এস', 't': 'টি',
    'u': 'ইউ', 'v': 'ভি', 'w': 'ডব্লিউ', 'x': 'এক্স', 'y': 'ওয়াই',
    'z': 'জেড',
}

# ============================================================================
# TEXT NORMALIZATION FUNCTIONS
# ============================================================================

def normalize_unicode(text: str) -> str:
    """
    Normalize Unicode to NFC form and fix weird variants.
    """
    # NFC normalization
    text = unicodedata.normalize('NFC', text)
    
    # Replace common Unicode variants
    replacements = {
        '\u200c': '',    # Zero-width non-joiner
        '\u200d': '',    # Zero-width joiner (keep for conjuncts if needed)
        '\u200b': '',    # Zero-width space
        '\ufeff': '',    # BOM
        '\u00a0': ' ',   # Non-breaking space -> regular space
        '।।': '।',       # Double danda -> single
    }
    
    for old, new in replacements.items():
        text = text.replace(old, new)
    
    return text


def normalize_punctuation(text: str) -> str:
    """
    Normalize and unify punctuation marks.
    """
    # Bengali danda to period (optional - keep Bengali style)
    # text = text.replace('।', '.')
    
    # Unify quotes
    text = re.sub(r'["""]', '"', text)
    text = re.sub(r"[''']", "'", text)
    
    # Unify dashes
    text = re.sub(r'[–—]', '-', text)
    
    # Remove multiple punctuation
    text = re.sub(r'[।,.!?]+', lambda m: m.group(0)[0], text)
    
    return text


def expand_number(num: int) -> str:
    """
    Convert number to Bengali words.
    """
    if num in NUMBER_WORDS:
        return NUMBER_WORDS[num]
    
    if num < 0:
        return 'ঋণাত্মক ' + expand_number(abs(num))
    
    if num < 100:
        tens = (num // 10) * 10
        ones = num % 10
        if tens in NUMBER_WORDS:
            if ones == 0:
                return NUMBER_WORDS[tens]
            return NUMBER_WORDS[tens] + ' ' + NUMBER_WORDS[ones]
        return NUMBER_WORDS.get(num, str(num))
    
    if num < 1000:
        hundreds = num // 100
        remainder = num % 100
        result = NUMBER_WORDS[hundreds] + 'শো' if hundreds > 1 else 'একশো'
        if remainder > 0:
            result += ' ' + expand_number(remainder)
        return result
    
    if num < 100000:  # Up to 99,999
        thousands = num // 1000
        remainder = num % 1000
        result = expand_number(thousands) + ' হাজার'
        if remainder > 0:
            result += ' ' + expand_number(remainder)
        return result
    
    if num < 10000000:  # Up to 99,99,999 (lakh)
        lakhs = num // 100000
        remainder = num % 100000
        result = expand_number(lakhs) + ' লাখ'
        if remainder > 0:
            result += ' ' + expand_number(remainder)
        return result
    
    # Crore and above
    crores = num // 10000000
    remainder = num % 10000000
    result = expand_number(crores) + ' কোটি'
    if remainder > 0:
        result += ' ' + expand_number(remainder)
    return result


def bengali_to_int(bengali_num: str) -> int:
    """Convert Bengali digit string to integer."""
    result = 0
    for char in bengali_num:
        if char in BENGALI_DIGITS:
            result = result * 10 + BENGALI_DIGITS[char]
        elif char.isdigit():
            result = result * 10 + int(char)
    return result


def expand_currency(text: str) -> str:
    """
    Expand currency expressions.
    E.g., ৳500 -> পাঁচশো টাকা
    """
    # Match Bengali currency
    pattern = r'৳\s*([০-৯\d]+(?:\.[০-৯\d]+)?)'
    
    def replace_taka(match):
        amount_str = match.group(1)
        # Convert Bengali digits
        amount_str = ''.join(str(BENGALI_DIGITS.get(c, c)) for c in amount_str)
        try:
            if '.' in amount_str:
                taka, paisa = amount_str.split('.')
                result = expand_number(int(taka)) + ' টাকা'
                if int(paisa) > 0:
                    result += ' ' + expand_number(int(paisa)) + ' পয়সা'
                return result
            else:
                return expand_number(int(amount_str)) + ' টাকা'
        except (ValueError, KeyError):
            return match.group(0)
    
    text = re.sub(pattern, replace_taka, text)
    return text


def expand_phone_number(text: str) -> str:
    """
    Expand phone numbers digit by digit.
    E.g., 01712345678 -> শূন্য এক সাত এক দুই...
    """
    # Match Bangladeshi phone numbers
    pattern = r'\b(০১[৩-৯][০-৯]{8}|01[3-9]\d{8})\b'
    
    def replace_phone(match):
        phone = match.group(1)
        digits = []
        for char in phone:
            if char in BENGALI_DIGITS:
                digits.append(NUMBER_WORDS[BENGALI_DIGITS[char]])
            elif char.isdigit():
                digits.append(NUMBER_WORDS[int(char)])
        return ' '.join(digits)
    
    text = re.sub(pattern, replace_phone, text)
    return text


def expand_date(text: str) -> str:
    """
    Expand date expressions.
    E.g., ২৫/১২/২০২৩ -> পঁচিশ ডিসেম্বর দুই হাজার তেইশ
    """
    months_bn = {
        1: 'জানুয়ারি', 2: 'ফেব্রুয়ারি', 3: 'মার্চ', 4: 'এপ্রিল',
        5: 'মে', 6: 'জুন', 7: 'জুলাই', 8: 'আগস্ট',
        9: 'সেপ্টেম্বর', 10: 'অক্টোবর', 11: 'নভেম্বর', 12: 'ডিসেম্বর'
    }
    
    # Match DD/MM/YYYY or DD-MM-YYYY
    pattern = r'([০-৯\d]{1,2})[/\-]([০-৯\d]{1,2})[/\-]([০-৯\d]{2,4})'
    
    def replace_date(match):
        day = bengali_to_int(match.group(1))
        month = bengali_to_int(match.group(2))
        year = bengali_to_int(match.group(3))
        
        if month < 1 or month > 12:
            return match.group(0)
        
        day_word = expand_number(day)
        month_word = months_bn.get(month, str(month))
        year_word = expand_number(year)
        
        return f"{day_word} {month_word} {year_word}"
    
    text = re.sub(pattern, replace_date, text)
    return text


def expand_year(text: str) -> str:
    """
    Expand year expressions like ২০২৩ সাল.
    """
    pattern = r'([০-৯\d]{4})\s*(?:সাল|সালে|সালের)?'
    
    def replace_year(match):
        year = bengali_to_int(match.group(1))
        return expand_number(year) + ' সাল'
    
    text = re.sub(pattern, replace_year, text)
    return text


def expand_numbers_in_text(text: str) -> str:
    """
    Find and expand all numbers in text.
    """
    # Match Bengali or Arabic digit sequences
    pattern = r'[০-৯\d]+(?:\.[০-৯\d]+)?'
    
    def replace_num(match):
        num_str = match.group(0)
        # Convert to integer/float
        num_str = ''.join(str(BENGALI_DIGITS.get(c, c)) for c in num_str)
        try:
            if '.' in num_str:
                integer_part, decimal_part = num_str.split('.')
                result = expand_number(int(integer_part)) + ' দশমিক '
                # Read decimal digits individually
                for d in decimal_part:
                    result += NUMBER_WORDS[int(d)] + ' '
                return result.strip()
            else:
                return expand_number(int(num_str))
        except (ValueError, KeyError):
            return match.group(0)
    
    text = re.sub(pattern, replace_num, text)
    return text


def transliterate_english(text: str) -> str:
    """
    Transliterate English words to Bengali pronunciation.
    """
    # Find English words
    pattern = r'\b[a-zA-Z]+\b'
    
    def replace_english(match):
        word = match.group(0).lower()
        result = ''
        for char in word:
            if char in ENGLISH_TO_BENGALI:
                result += ENGLISH_TO_BENGALI[char]
        return result if result else match.group(0)
    
    text = re.sub(pattern, replace_english, text)
    return text


def normalize_text(text: str, expand_nums: bool = True, 
                   transliterate_eng: bool = True) -> str:
    """
    Comprehensive text normalization for Bengali TTS.
    
    Args:
        text: Input text
        expand_nums: Whether to expand numbers to words
        transliterate_eng: Whether to transliterate English words
    
    Returns:
        Normalized text
    """
    # Unicode normalization
    text = normalize_unicode(text)
    
    # Punctuation normalization
    text = normalize_punctuation(text)
    
    # Expand currency
    text = expand_currency(text)
    
    # Expand phone numbers
    text = expand_phone_number(text)
    
    # Expand dates
    text = expand_date(text)
    
    # Expand remaining numbers
    if expand_nums:
        text = expand_numbers_in_text(text)
    
    # Transliterate English
    if transliterate_eng:
        text = transliterate_english(text)
    
    # Remove remaining punctuation for phoneme conversion
    text = re.sub(r'[।,;:!?"\'\(\)\[\]{}–—\-]', ' ', text)
    
    # Fix spacing between consonant clusters
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def clean_sentence(text: str, min_length: int = 2, max_length: int = 500) -> Optional[str]:
    """
    Clean and validate a sentence.
    
    Args:
        text: Input sentence
        min_length: Minimum character length
        max_length: Maximum character length
    
    Returns:
        Cleaned sentence or None if invalid
    """
    if not text:
        return None
    
    # Strip whitespace
    text = text.strip()
    
    # Check length
    if len(text) < min_length or len(text) > max_length:
        return None
    
    # Remove if too much non-Bengali content
    bengali_chars = sum(1 for c in text if '\u0980' <= c <= '\u09FF')
    if bengali_chars < len(text.replace(' ', '')) * 0.5:
        return None  # Less than 50% Bengali
    
    return text


# ============================================================================
# G2P FUNCTIONS
# ============================================================================

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


def is_chandrabindu(char: str) -> bool:
    """Check if character is chandrabindu (nasalization marker)."""
    return char == 'ঁ'


def bengali_g2p(text: str, preserve_nasalization: bool = True) -> List[str]:
    """
    Convert Bengali text to phoneme sequence.
    
    Args:
        text: Bengali text string
        preserve_nasalization: Whether to mark nasalized vowels
        
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
        
        # Check for conjuncts (up to 5 characters for three-consonant clusters)
        found_conjunct = False
        for length in [5, 4, 3, 2]:
            if i + length <= len(text):
                seq = text[i:i+length]
                if seq in CONJUNCT_MAP:
                    phonemes.append(CONJUNCT_MAP[seq])
                    i += length
                    found_conjunct = True
                    break
        
        if found_conjunct:
            continue
        
        # Handle chandrabindu (nasalization)
        if is_chandrabindu(char):
            # Add nasalization marker to previous phoneme
            if phonemes and preserve_nasalization:
                phonemes[-1] = phonemes[-1] + '̃'
            i += 1
            continue
        
        # Handle consonants
        if is_consonant(char):
            base_phoneme = CONSONANT_MAP[char]
            
            # Look ahead for matra, hasanta, or chandrabindu
            if i + 1 < len(text):
                next_char = text[i + 1]
                
                if is_matra(next_char):
                    # Consonant + matra
                    phonemes.append(base_phoneme)
                    matra_phoneme = MATRA_MAP[next_char]
                    
                    # Check for chandrabindu after matra
                    if i + 2 < len(text) and is_chandrabindu(text[i + 2]):
                        matra_phoneme += '̃'
                        i += 3
                    else:
                        i += 2
                    
                    phonemes.append(matra_phoneme)
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
            vowel_phoneme = VOWEL_MAP[char]
            
            # Check for chandrabindu
            if i + 1 < len(text) and is_chandrabindu(text[i + 1]):
                vowel_phoneme += '̃'
                i += 2
            else:
                i += 1
            
            phonemes.append(vowel_phoneme)
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
    
    return result


def phonemes_to_string(phonemes: List[str]) -> str:
    """Convert phoneme list to space-separated string."""
    return ' '.join(phonemes)


def g2p(text: str, expand_nums: bool = True, 
        transliterate_eng: bool = True) -> str:
    """
    Main G2P function - converts Bengali text to phoneme string.
    
    Args:
        text: Bengali text
        expand_nums: Whether to expand numbers to words
        transliterate_eng: Whether to transliterate English words
        
    Returns:
        Space-separated phoneme string
    """
    # Pre-normalize text (handles numbers, English, etc.)
    normalized = normalize_text(text, expand_nums, transliterate_eng)
    phonemes = bengali_g2p(normalized)
    return phonemes_to_string(phonemes)


def get_phoneme_inventory() -> Dict[str, List[str]]:
    """
    Get the complete phoneme inventory used by this G2P.
    Useful for building TTS symbol tables.
    """
    inventory = {
        'vowels': list(set(VOWEL_MAP.values())),
        'consonants': list(set(CONSONANT_MAP.values())),
        'matras': list(set(MATRA_MAP.values())),
        'special': [p for p in SPECIAL_MAP.values() if p],
        'conjuncts': list(set(CONJUNCT_MAP.values())),
    }
    
    # Add nasalized versions
    inventory['nasalized'] = [v + '̃' for v in inventory['vowels']]
    
    # Flatten for complete list
    all_phonemes = set()
    for category in inventory.values():
        all_phonemes.update(category)
    
    inventory['all'] = sorted(list(all_phonemes))
    
    return inventory


# ============================================================================
# CLI & TESTING
# ============================================================================

def test_g2p():
    """Test G2P with example words and sentences."""
    test_cases = [
        # Basic phonemes
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
        
        # Vowels
        ('অ', 'Vowel a'),
        ('আ', 'Vowel aa'),
        ('ই', 'Vowel i'),
        ('উ', 'Vowel u'),
        ('এ', 'Vowel e'),
        ('ও', 'Vowel o'),
        
        # Common words
        ('বাংলা', 'Word: Bangla'),
        ('আমি', 'Word: Ami (I)'),
        ('তুমি', 'Word: Tumi (You)'),
        ('বাংলাদেশ', 'Word: Bangladesh'),
        
        # Conjuncts
        ('ক্ষমা', 'Word with conjunct: Kshoma'),
        ('জ্ঞান', 'Word with conjunct: Gyan'),
        ('বিদ্যা', 'Word: Bidya'),
        ('প্রেম', 'Word with ra-phala: Prem'),
        ('ব্যক্তি', 'Word with ya-phala: Byakti'),
        
        # Nasalization
        ('আঁখি', 'Word with chandrabindu: Akhi'),
        ('চাঁদ', 'Word with chandrabindu: Chand'),
        
        # Sentences
        ('আমার নাম কি', 'Sentence: Amar naam ki'),
        ('তুমি কেমন আছো', 'Sentence: Tumi kemon achho'),
        ('বাংলাদেশ আমার দেশ', 'Sentence: Bangladesh amar desh'),
        
        # Numbers
        ('আমার বয়স ২৫', 'Sentence with number'),
        ('৳৫০০ টাকা', 'Currency amount'),
        
        # Dates
        ('২৫/১২/২০২৩', 'Date format'),
    ]
    
    print("=" * 70)
    print("Bengali G2P Test Results - Comprehensive")
    print("=" * 70)
    
    for text, description in test_cases:
        result = g2p(text)
        print(f"\n{description}")
        print(f"  Input:  {text}")
        print(f"  Output: {result}")
    
    print("\n" + "=" * 70)
    
    # Print phoneme inventory
    print("\nPhoneme Inventory Summary:")
    print("-" * 40)
    inventory = get_phoneme_inventory()
    print(f"  Vowels: {len(inventory['vowels'])}")
    print(f"  Consonants: {len(inventory['consonants'])}")
    print(f"  Conjuncts: {len(inventory['conjuncts'])}")
    print(f"  Total unique phonemes: {len(inventory['all'])}")
    
    print("\n" + "=" * 70)


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1:
        # Convert command line argument
        text = ' '.join(sys.argv[1:])
        result = g2p(text)
        print(f"Input:  {text}")
        print(f"Output: {result}")
    else:
        test_g2p()
