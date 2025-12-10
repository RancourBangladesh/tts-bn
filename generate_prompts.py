#!/usr/bin/env python3
"""
Bengali TTS Prompt Generator

Generates comprehensive recording prompts covering:
- All vowels (shoroborno)
- All consonants with matras
- Common conjuncts (juktakkhor)
- Carrier sentences
"""

import csv
import os
from typing import List, Tuple

# ============================================================================
# BENGALI CHARACTER SETS
# ============================================================================

# Vowels (স্বরবর্ণ) - 11
VOWELS = ['অ', 'আ', 'ই', 'ঈ', 'উ', 'ঊ', 'ঋ', 'এ', 'ঐ', 'ও', 'ঔ']

# Consonants (ব্যঞ্জনবর্ণ) - 35
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

# Matras (dependent vowels) - 10
MATRAS = ['া', 'ি', 'ী', 'ু', 'ূ', 'ৃ', 'ে', 'ৈ', 'ো', 'ৌ']

# Common high-frequency conjuncts (juktakkhor)
COMMON_CONJUNCTS = [
    # Consonant + য-ফলা (ya-phala)
    'ক্য', 'খ্য', 'গ্য', 'ঘ্য', 'চ্য', 'জ্য', 'ঝ্য',
    'ট্য', 'ড্য', 'ণ্য', 'ত্য', 'থ্য', 'দ্য', 'ধ্য', 'ন্য',
    'প্য', 'ব্য', 'ভ্য', 'ম্য', 'র্য', 'ল্য', 'শ্য', 'ষ্য', 'স্য', 'হ্য',
    
    # Consonant + র-ফলা (ra-phala)
    'ক্র', 'খ্র', 'গ্র', 'ঘ্র', 'চ্র', 'জ্র',
    'ট্র', 'ড্র', 'ত্র', 'থ্র', 'দ্র', 'ধ্র', 'ন্র',
    'প্র', 'ফ্র', 'ব্র', 'ভ্র', 'ম্র', 'শ্র', 'স্র', 'হ্র',
    
    # Geminate (doubled) consonants
    'ক্ক', 'গ্গ', 'চ্চ', 'জ্জ', 'ট্ট', 'ড্ড', 'ত্ত', 'দ্দ',
    'ন্ন', 'প্প', 'ব্ব', 'ম্ম', 'ল্ল', 'স্স',
    
    # Common combinations
    'ক্ত', 'ক্ন', 'ক্ম', 'ক্ষ', 'ক্স',
    'গ্ন', 'গ্ম', 'ঘ্ন',
    'ঙ্ক', 'ঙ্খ', 'ঙ্গ', 'ঙ্ঘ',
    'চ্ছ', 'জ্ঞ', 'ঞ্চ', 'ঞ্ছ', 'ঞ্জ', 'ঞ্ঝ',
    'ট্ঠ', 'ড্ঢ', 'ণ্ট', 'ণ্ঠ', 'ণ্ড', 'ণ্ঢ', 'ণ্ণ',
    'ত্থ', 'ত্ন', 'ত্ম', 'দ্ধ', 'দ্ব', 'দ্ম', 'দ্ভ',
    'ন্ত', 'ন্থ', 'ন্দ', 'ন্ধ', 'ন্ম', 'ন্ব',
    'প্ত', 'প্ন', 'প্ম', 'প্স',
    'ব্দ', 'ব্ধ', 'ব্ন',
    'ভ্ন',
    'ম্ন', 'ম্প', 'ম্ফ', 'ম্ব', 'ম্ভ',
    'ল্ক', 'ল্গ', 'ল্ট', 'ল্ড', 'ল্প', 'ল্ফ', 'ল্ব', 'ল্ম',
    'শ্চ', 'শ্ছ', 'শ্ন', 'শ্ব', 'শ্ম',
    'ষ্ক', 'ষ্ট', 'ষ্ঠ', 'ষ্ণ', 'ষ্প', 'ষ্ফ', 'ষ্ম',
    'স্ক', 'স্খ', 'স্ট', 'স্ত', 'স্থ', 'স্ন', 'স্প', 'স্ফ', 'স্ব', 'স্ম',
    'হ্ন', 'হ্ম', 'হ্য', 'হ্র', 'হ্ল',
    
    # Three-consonant clusters
    'ক্ত্র', 'ক্ষ্ণ', 'ক্ষ্ম',
    'ঙ্ক্ষ',
    'ন্ত্র', 'ন্ত্য', 'ন্দ্র', 'ন্দ্য', 'ন্ধ্র', 'ন্ধ্য',
    'ম্প্র', 'ম্ব্র',
    'ষ্ট্র', 'ষ্ঠ্য',
    'স্ত্র', 'স্ত্য', 'স্থ্য', 'স্প্র',
]

# Common Bengali words for minimal pairs and practice
COMMON_WORDS = [
    'আমি', 'তুমি', 'সে', 'আমরা', 'তোমরা', 'তারা',
    'এটা', 'ওটা', 'সেটা', 'কি', 'কে', 'কেন', 'কোথায়', 'কখন', 'কীভাবে',
    'হ্যাঁ', 'না', 'ঠিক', 'ভুল', 'ভালো', 'খারাপ',
    'বড়', 'ছোট', 'নতুন', 'পুরনো', 'সুন্দর',
    'জল', 'পানি', 'খাবার', 'ভাত', 'রুটি', 'মাছ', 'মাংস',
    'বাড়ি', 'ঘর', 'দরজা', 'জানালা', 'ছাদ', 'মাটি',
    'গাছ', 'ফুল', 'পাতা', 'ফল', 'বীজ',
    'মা', 'বাবা', 'ভাই', 'বোন', 'দাদা', 'দিদি',
    'বন্ধু', 'শিক্ষক', 'ছাত্র', 'ডাক্তার',
    'দিন', 'রাত', 'সকাল', 'বিকাল', 'সন্ধ্যা',
    'আজ', 'কাল', 'পরশু', 'গতকাল',
    'এক', 'দুই', 'তিন', 'চার', 'পাঁচ', 'ছয়', 'সাত', 'আট', 'নয়', 'দশ',
    'লাল', 'নীল', 'সবুজ', 'হলুদ', 'সাদা', 'কালো',
    'যাওয়া', 'আসা', 'খাওয়া', 'পড়া', 'লেখা', 'দেখা', 'শোনা', 'বলা',
    'করা', 'হওয়া', 'থাকা', 'দেওয়া', 'নেওয়া', 'পাওয়া',
]

# Carrier sentences for natural speech patterns
CARRIER_SENTENCES = [
    'এটি {}।',
    'আমি {} বলছি।',
    'এখানে {} আছে।',
    '{} দেখতে পাচ্ছি।',
    'আমার {} দরকার।',
    'তোমার {} কোথায়?',
    '{} খুব সুন্দর।',
    'আজ {} পড়ব।',
]

# Complete sentences for prosody
COMPLETE_SENTENCES = [
    'আমার নাম বাংলা।',
    'বাংলাদেশ আমার দেশ।',
    'আজ আবহাওয়া ভালো।',
    'তুমি কেমন আছো?',
    'আমি ভালো আছি।',
    'ধন্যবাদ তোমাকে।',
    'আবার দেখা হবে।',
    'শুভ সকাল।',
    'শুভ রাত্রি।',
    'খুব ভালো লাগছে।',
    'একটু দাঁড়াও।',
    'আমি আসছি।',
    'তুমি কোথায় যাচ্ছ?',
    'এটা কী?',
    'ওটা কার?',
    'আমি জানি না।',
    'তুমি জানো?',
    'হ্যাঁ, আমি জানি।',
    'না, আমি জানি না।',
    'দয়া করে বলুন।',
]


def generate_prompts() -> List[Tuple[int, str, str]]:
    """
    Generate all recording prompts.
    
    Returns:
        List of tuples: (prompt_id, type, text)
    """
    prompts = []
    prompt_id = 1
    
    # 1. Single vowels
    for vowel in VOWELS:
        prompts.append((prompt_id, 'vowel', vowel))
        prompt_id += 1
    
    # 2. Single consonants (with inherent vowel)
    for consonant in CONSONANTS:
        prompts.append((prompt_id, 'consonant', consonant))
        prompt_id += 1
    
    # 3. Consonants with each matra
    for consonant in CONSONANTS:
        for matra in MATRAS:
            text = consonant + matra
            prompts.append((prompt_id, 'consonant_matra', text))
            prompt_id += 1
    
    # 4. Common conjuncts
    for conjunct in COMMON_CONJUNCTS:
        prompts.append((prompt_id, 'conjunct', conjunct))
        prompt_id += 1
    
    # 5. Common words
    for word in COMMON_WORDS:
        prompts.append((prompt_id, 'word', word))
        prompt_id += 1
    
    # 6. Complete sentences
    for sentence in COMPLETE_SENTENCES:
        prompts.append((prompt_id, 'sentence', sentence))
        prompt_id += 1
    
    # 7. Carrier sentences with words
    for word in COMMON_WORDS[:20]:  # Use first 20 words
        for carrier in CARRIER_SENTENCES[:3]:  # Use first 3 carriers
            sentence = carrier.format(word)
            prompts.append((prompt_id, 'carrier', sentence))
            prompt_id += 1
    
    return prompts


def save_prompts_csv(prompts: List[Tuple[int, str, str]], output_path: str):
    """Save prompts to CSV file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['prompt_id', 'type', 'text', 'status'])
        for prompt_id, prompt_type, text in prompts:
            writer.writerow([f'{prompt_id:05d}', prompt_type, text, 'pending'])
    
    print(f"Saved {len(prompts)} prompts to {output_path}")


def print_stats(prompts: List[Tuple[int, str, str]]):
    """Print statistics about generated prompts."""
    type_counts = {}
    for _, prompt_type, _ in prompts:
        type_counts[prompt_type] = type_counts.get(prompt_type, 0) + 1
    
    print("\n" + "=" * 50)
    print("Prompt Generation Statistics")
    print("=" * 50)
    print(f"\nTotal prompts: {len(prompts)}")
    print("\nBy type:")
    for prompt_type, count in sorted(type_counts.items()):
        print(f"  {prompt_type}: {count}")
    print("=" * 50)


def main():
    """Main function to generate prompts."""
    output_dir = os.path.join(os.path.dirname(__file__), 'dataset', 'prompts')
    output_path = os.path.join(output_dir, 'prompts.csv')
    
    print("Generating Bengali TTS recording prompts...")
    prompts = generate_prompts()
    
    print_stats(prompts)
    save_prompts_csv(prompts, output_path)
    
    print(f"\nPrompts saved to: {output_path}")
    print("\nNext steps:")
    print("1. Run: python app.py")
    print("2. Open http://localhost:5000 in your browser")
    print("3. Start recording!")


if __name__ == '__main__':
    main()
