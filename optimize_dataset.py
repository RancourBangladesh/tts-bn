#!/usr/bin/env python3
"""
Bengali TTS Dataset Optimizer

Optimizes and curates Bengali text data for high-quality, studio-quality TTS training.
Creates a fluent, phoneme-balanced dataset optimized for neural TTS models.

Key optimizations:
1. Phoneme coverage analysis - ensures all Bengali sounds are well represented
2. Sentence length optimization - balances short and long utterances
3. Prosody diversity - varied sentence types (statements, questions, etc.)
4. Text normalization - consistent formatting and cleanup
5. Deduplication - removes exact and near duplicates
6. Quality filtering - removes problematic or incomplete sentences
7. Sentence generation - creates additional natural sentences from templates

For RTX 5060 Ti 16GB (Studio Quality):
- Recommended dataset: 20,000 sentences (25-30 hours of audio)
- Sentence length: 5-80 characters (0.5-8 seconds each)
- Balance between short phrases and longer sentences
- Higher repetition of phonemes for better model learning
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

# ============================================================================
# SENTENCE GENERATION TEMPLATES FOR STUDIO-QUALITY TTS
# ============================================================================
# These templates help generate natural, fluent Bengali sentences
# covering diverse phonemes, prosody patterns, and speaking styles

# Common Bengali sentence patterns (templates with {placeholders})
SENTENCE_TEMPLATES = [
    # Statements
    "আমি {noun} {verb}।",
    "সে {noun} {verb}।",
    "তারা {noun} {verb}।",
    "আমরা {adjective} {noun} {verb}।",
    "{noun} খুব {adjective}।",
    "এই {noun} অনেক {adjective}।",
    "{name} {place} যাচ্ছে।",
    "{name} {food} খাচ্ছে।",
    "{time} আমি {activity} করব।",
    "আজ {weather} আছে।",
    "{noun} কিনতে হবে আমাকে।",
    "এখন {activity} করার সময়।",
    "{adjective} দিন আজকে।",
    "তুমি {noun} {verb}।",
    "আপনি {noun} {verb}।",
    "ও {noun} {verb}।",
    "{name} {noun} {verb}।",
    "সবাই {noun} {verb}।",
    "কেউ {noun} {verb}।",
    "{adjective} {noun} পছন্দ করি।",
    "আমার {adjective} {noun} লাগছে।",
    "{noun} দরকার আছে।",
    "{noun} নেই এখানে।",
    "{noun} থাকলে ভালো হত।",
    "{time} {name} আসবে।",
    "{time} {activity} শুরু হবে।",
    "{place} থেকে {name} এসেছে।",
    "{name} {place} থাকে।",
    "{food} আমার পছন্দ।",
    "{food} স্বাস্থ্যের জন্য ভালো।",
    
    # Questions  
    "তুমি কি {noun} {verb}?",
    "আপনি কি {adjective} {noun} চান?",
    "{name} কোথায় গেছে?",
    "কখন {noun} আসবে?",
    "কেন {noun} {verb}?",
    "কীভাবে {noun} {verb}?",
    "কোন {noun} ভালো?",
    "আপনার {noun} কেমন?",
    "তোমার {noun} কোথায়?",
    "{noun} কি {adjective}?",
    "{name} কি {place} যাবে?",
    "{time} কি {activity} হবে?",
    "আপনি কি {food} খাবেন?",
    "এখানে {noun} পাওয়া যায়?",
    "{name} কি এসেছে?",
    "তুমি কি {adjective} আছো?",
    "{noun} কত দাম?",
    "{noun} কখন শেষ হবে?",
    
    # Exclamations
    "কী {adjective} {noun}!",
    "বাহ, কত {adjective}!",
    "অসাধারণ {noun}!",
    "চমৎকার {noun}!",
    "{adjective} {noun} দেখো!",
    "কী মজা, {noun}!",
    "আহা, কত {adjective}!",
    "ওরে বাবা, {noun}!",
    
    # Commands/Requests
    "দয়া করে {noun} দিন।",
    "অনুগ্রহ করে {verb}।",
    "{noun} নিয়ে আসো।",
    "একটু {verb} তো।",
    "{noun} দেখো।",
    "{verb} এখন।",
    "{place} যাও।",
    "{food} খাও।",
    "{adjective} হও।",
    "{activity} করো।",
    
    # Conversational
    "আচ্ছা, {noun} কেমন?",
    "বলুন, আপনার {noun} কী?",
    "জানেন, আজ {noun} হয়েছে।",
    "শুনুন, {noun} সম্পর্কে বলি।",
    "দেখুন, {noun} এখানে আছে।",
    "বুঝলাম, {noun} দরকার।",
    "ঠিক আছে, {verb} পরে।",
    "হ্যাঁ, {noun} জানি।",
    "না, {noun} চাই না।",
    "হতে পারে, {noun} হবে।",
    
    # Compound sentences
    "আমি {noun} করব এবং তুমি {noun2} করো।",
    "{noun} ভালো কিন্তু {noun2} আরও ভালো।",
    "যদি {noun} হয় তাহলে {noun2} হবে।",
    "{noun} করার পরে {noun2} করব।",
    "{noun} শেষ হলে {noun2} শুরু করব।",
    "{adjective} {noun} থাকলে {noun2} হবে।",
    "{name} {noun} করবে আর আমি {noun2} করব।",
    "{time} {noun} হবে তারপর {noun2} হবে।",
]

# Word lists for template filling
TEMPLATE_NOUNS = [
    "কাজ", "পড়াশোনা", "খাবার", "বই", "গান", "ছবি", "বাজার",
    "চা", "কফি", "জল", "ভাত", "রুটি", "মাছ", "মাংস", "সবজি",
    "ফল", "ফুল", "গাছ", "পাখি", "মাটি", "আকাশ", "সূর্য", "চাঁদ",
    "বাড়ি", "ঘর", "দরজা", "জানালা", "রাস্তা", "গাড়ি", "বাস",
    "স্কুল", "কলেজ", "অফিস", "হাসপাতাল", "দোকান", "মন্দির",
    "নদী", "সমুদ্র", "পাহাড়", "বন", "মাঠ", "বাগান", "পার্ক",
    "টাকা", "কাপড়", "জুতা", "ঘড়ি", "মোবাইল", "কম্পিউটার",
    "সময়", "দিন", "রাত", "সকাল", "বিকাল", "সন্ধ্যা",
    "বন্ধু", "পরিবার", "মা", "বাবা", "ভাই", "বোন", "সন্তান",
    "স্বপ্ন", "আশা", "ভালোবাসা", "সুখ", "দুঃখ", "আনন্দ",
    "সমস্যা", "সমাধান", "প্রশ্ন", "উত্তর", "কথা", "গল্প",
]

TEMPLATE_VERBS = [
    "করি", "করছি", "করব", "করেছি", "করতে চাই",
    "দেখি", "দেখছি", "দেখব", "দেখেছি", "দেখতে চাই",
    "শুনি", "শুনছি", "শুনব", "শুনেছি", "শুনতে চাই",
    "বলি", "বলছি", "বলব", "বলেছি", "বলতে চাই",
    "খাই", "খাচ্ছি", "খাব", "খেয়েছি", "খেতে চাই",
    "যাই", "যাচ্ছি", "যাব", "গেছি", "যেতে চাই",
    "আসি", "আসছি", "আসব", "এসেছি", "আসতে চাই",
    "পড়ি", "পড়ছি", "পড়ব", "পড়েছি", "পড়তে চাই",
    "লিখি", "লিখছি", "লিখব", "লিখেছি", "লিখতে চাই",
    "ভাবি", "ভাবছি", "ভাবব", "ভেবেছি", "ভাবতে চাই",
    "চাই", "চাইছি", "চাইব", "চেয়েছি", "চাইতে থাকি",
    "পারি", "পারছি", "পারব", "পেরেছি", "পারতে চাই",
    "নিই", "নিচ্ছি", "নেব", "নিয়েছি", "নিতে চাই",
    "দিই", "দিচ্ছি", "দেব", "দিয়েছি", "দিতে চাই",
]

TEMPLATE_ADJECTIVES = [
    "সুন্দর", "ভালো", "খারাপ", "বড়", "ছোট", "নতুন", "পুরনো",
    "সাদা", "কালো", "লাল", "নীল", "সবুজ", "হলুদ", "গোলাপি",
    "গরম", "ঠান্ডা", "শীতল", "উষ্ণ", "মিষ্টি", "তেতো", "নোনা",
    "কঠিন", "সহজ", "জটিল", "সরল", "দ্রুত", "ধীর", "শান্ত",
    "উজ্জ্বল", "অন্ধকার", "পরিষ্কার", "নোংরা", "সতেজ", "বাসি",
    "প্রিয়", "অপ্রিয়", "বিখ্যাত", "অজানা", "পরিচিত", "নিকট",
    "কঠোর", "নরম", "মজবুত", "দুর্বল", "সুস্থ", "অসুস্থ",
    "সুখী", "দুঃখী", "আনন্দিত", "চিন্তিত", "ক্লান্ত", "সজাগ",
]

TEMPLATE_NAMES = [
    "রহিম", "করিম", "জামাল", "কামাল", "সালমা", "ফাতেমা",
    "অর্জুন", "কৃষ্ণ", "রাম", "সীতা", "গীতা", "মীরা", "রবি",
    "সুমন", "সুমি", "রুমি", "টুম্পা", "রাকিব", "সাকিব",
    "মিতা", "রিতা", "নিতা", "প্রিয়া", "তানিয়া", "সানিয়া",
    "আমির", "জাহির", "নাসির", "বশির", "মুনির", "শফিক",
]

TEMPLATE_PLACES = [
    "ঢাকায়", "চট্টগ্রামে", "সিলেটে", "রাজশাহীতে", "খুলনায়",
    "বাড়িতে", "অফিসে", "স্কুলে", "কলেজে", "বিশ্ববিদ্যালয়ে",
    "হাসপাতালে", "দোকানে", "বাজারে", "মাঠে", "পার্কে",
    "নদীর ধারে", "সমুদ্রের কাছে", "পাহাড়ে", "বনে", "গ্রামে",
]

TEMPLATE_FOODS = [
    "ভাত", "রুটি", "পরোটা", "বিরিয়ানি", "খিচুড়ি", "পোলাও",
    "মাছ", "মাংস", "ডিম", "সবজি", "ডাল", "ভর্তা", "ভাজি",
    "মিষ্টি", "রসগোল্লা", "সন্দেশ", "জিলাপি", "পায়েস",
    "চা", "কফি", "জুস", "লাচ্ছি", "শরবত", "পানি",
]

TEMPLATE_TIMES = [
    "সকালে", "দুপুরে", "বিকালে", "সন্ধ্যায়", "রাতে",
    "আজ", "কাল", "পরশু", "গতকাল", "আগামীকাল",
    "এখন", "তখন", "পরে", "আগে", "শীঘ্রই",
]

TEMPLATE_WEATHER = [
    "রোদ", "বৃষ্টি", "মেঘলা", "ঝড়", "গরম", "ঠান্ডা",
    "আবহাওয়া ভালো", "আবহাওয়া খারাপ", "বাতাস বইছে",
]

TEMPLATE_ACTIVITIES = [
    "খেলা", "পড়াশোনা", "কাজ", "রান্না", "গান গাওয়া",
    "নাচ", "ছবি আঁকা", "বাগান করা", "হাঁটা", "দৌড়ানো",
    "সাঁতার কাটা", "ঘুমানো", "বিশ্রাম নেওয়া", "টিভি দেখা",
]

# Additional high-quality sentences for fluent TTS
# These are natural, diverse sentences covering various topics
ADDITIONAL_SENTENCES = [
    # Greetings and common phrases
    "নমস্কার, আপনি কেমন আছেন?",
    "সালাম, সব ঠিক আছে?",
    "শুভ সকাল, ভালো দিন কাটুক।",
    "শুভ বিকাল, কেমন কাটল দিন?",
    "শুভ সন্ধ্যা, পরিবার কেমন আছে?",
    "শুভ রাত্রি, ভালো ঘুম হোক।",
    "আবার দেখা হবে, যত্ন নেবেন।",
    "আসি তাহলে, আল্লাহ হাফেজ।",
    
    # Daily conversations
    "আজ আবহাওয়া বেশ ভালো লাগছে।",
    "বাইরে অনেক গরম পড়েছে আজ।",
    "বৃষ্টি হওয়ার সম্ভাবনা আছে।",
    "একটু চা খাবেন নাকি কফি?",
    "দুপুরে কী খেলেন আজ?",
    "রাতের খাবার তৈরি হয়ে গেছে।",
    "বাজার থেকে কিছু সবজি আনতে হবে।",
    "মোবাইলে কথা বলছিলাম একটু।",
    
    # Work and study
    "অফিসে আজ অনেক কাজ ছিল।",
    "মিটিং শেষ হতে দেরি হয়ে গেল।",
    "প্রজেক্ট জমা দেওয়ার তারিখ কবে?",
    "পরীক্ষার প্রস্তুতি কেমন চলছে?",
    "বইটা পড়া শেষ হয়নি এখনও।",
    "নতুন কিছু শিখতে চাই আমি।",
    
    # Shopping and services
    "দোকানে নতুন জিনিস এসেছে।",
    "দাম একটু বেশি মনে হচ্ছে।",
    "কিছু ছাড় দেওয়া যাবে কি?",
    "বিলটা কত হল সব মিলিয়ে?",
    "ক্রেডিট কার্ডে পেমেন্ট করব।",
    
    # Health and well-being
    "শরীর ভালো নেই কিছুদিন ধরে।",
    "ডাক্তারের কাছে যেতে হবে।",
    "ওষুধ খেতে ভুলবেন না।",
    "পর্যাপ্ত বিশ্রাম নিন।",
    "নিয়মিত ব্যায়াম করা উচিত।",
    
    # Travel and transportation
    "ট্রেনের টিকিট কেটে রেখেছি।",
    "বাসে ভিড় অনেক বেশি ছিল।",
    "গাড়িতে জ্যাম লেগে গেছে।",
    "বিমানের সময় কখন?",
    "হোটেল বুকিং করা হয়ে গেছে।",
    
    # Emotions and expressions
    "খুব খুশি হলাম শুনে।",
    "দুঃখিত, আমার ভুল হয়ে গেছে।",
    "চিন্তা করবেন না, সব ঠিক হবে।",
    "অভিনন্দন আপনাকে।",
    "ধন্যবাদ সাহায্যের জন্য।",
    "মাফ করবেন, একটু দেরি হয়ে গেল।",
    
    # Directions and locations
    "বাম দিকে ঘুরুন এখান থেকে।",
    "সোজা গেলেই পাবেন জায়গাটা।",
    "এখান থেকে কত দূর?",
    "ঠিকানাটা একটু বলবেন?",
    "গুগল ম্যাপে দেখে নিন।",
    
    # Technology and modern life
    "ইন্টারনেট সংযোগ নেই এখানে।",
    "পাসওয়ার্ড ভুলে গেছি।",
    "অ্যাপটা আপডেট করতে হবে।",
    "ভিডিও কল করতে পারবেন?",
    "মেসেজ পেয়েছেন কি?",
    
    # Numbers and counting
    "এক, দুই, তিন, চার, পাঁচ।",
    "ছয়, সাত, আট, নয়, দশ।",
    "প্রথম, দ্বিতীয়, তৃতীয়।",
    "একশো টাকা লাগবে।",
    "হাজার টাকার নোট আছে?",
    
    # Time expressions
    "ঘড়িতে কটা বাজে এখন?",
    "পাঁচটা বাজতে পাঁচ মিনিট বাকি।",
    "দশ মিনিটের মধ্যে আসছি।",
    "আধ ঘণ্টা অপেক্ষা করুন।",
    "দুই ঘণ্টা লাগবে পৌঁছাতে।",
    
    # Nature and environment
    "আকাশে সুন্দর রংধনু দেখা যাচ্ছে।",
    "পাখিরা গান গাইছে গাছে।",
    "ফুলের সুবাস ছড়িয়ে পড়েছে।",
    "নদীর জল বেশ পরিষ্কার।",
    "পাহাড়ের দৃশ্য অসাধারণ।",
    
    # Culture and traditions
    "পূজার ছুটি কবে শুরু হচ্ছে?",
    "ঈদের কেনাকাটা করতে হবে।",
    "বিয়ের দাওয়াত পেয়েছি।",
    "জন্মদিনে কী উপহার দেব?",
    "নববর্ষের শুভেচ্ছা জানাই।",
    
    # Extended conversational sentences for studio quality
    "আপনার সাথে কথা বলে ভালো লাগল।",
    "এই বিষয়ে আরও জানতে চাই।",
    "একটু সময় দিন, আমি দেখে নিচ্ছি।",
    "সেটা সম্ভব হবে কিনা জানি না।",
    "চেষ্টা করে দেখব অবশ্যই।",
    "আমার মনে হয় এটা ঠিক হবে।",
    "তোমার কথা শুনে অবাক হলাম।",
    "এত সুন্দর খবর শুনে খুশি হলাম।",
    "দুঃখিত, এটা আমার পক্ষে সম্ভব না।",
    "আবার চেষ্টা করুন পরে।",
    
    # Professional settings
    "মিটিং কয়টায় শুরু হবে?",
    "রিপোর্ট জমা দেওয়ার সময় শেষ।",
    "এই প্রজেক্ট খুব গুরুত্বপূর্ণ।",
    "টিম মেম্বারদের সাথে আলোচনা করি।",
    "ক্লায়েন্ট সন্তুষ্ট হয়েছেন।",
    "বাজেট বাড়ানো দরকার।",
    "ডেডলাইন মিস করা যাবে না।",
    "প্রেজেন্টেশন তৈরি করতে হবে।",
    
    # Family and relationships
    "মা আজ রান্না করবেন।",
    "বাবা অফিস থেকে ফিরেছেন।",
    "ছোট ভাই স্কুলে গেছে।",
    "দিদি কলেজে পড়ছে।",
    "দাদু অসুস্থ আছেন।",
    "নানি গল্প শোনাবেন।",
    "পরিবারের সবাই ভালো আছে।",
    "একসাথে খেতে বসব।",
    
    # Education
    "পরীক্ষা ভালো হয়েছে।",
    "রেজাল্ট বের হয়ে গেছে।",
    "নতুন বই কিনতে হবে।",
    "টিউশন ফি জমা দিতে হবে।",
    "হোমওয়ার্ক শেষ করতে হবে।",
    "স্যার ভালো পড়ান।",
    "ক্লাস মিস করা যাবে না।",
    "লাইব্রেরিতে পড়তে যাব।",
    
    # Weather and seasons
    "শীতকাল আসছে।",
    "গরমে কষ্ট হচ্ছে।",
    "বর্ষায় ভালো লাগে।",
    "বসন্তে ফুল ফোটে।",
    "শরতের আকাশ সুন্দর।",
    "হেমন্তে ধান কাটা হয়।",
    "গ্রীষ্মে আম পাকে।",
    "শীতে কুয়াশা পড়ে।",
    
    # Food and cooking
    "আজ বিরিয়ানি রান্না করব।",
    "মাছের ঝোল খুব সুস্বাদু।",
    "মিষ্টি দই খেতে ইচ্ছে করছে।",
    "চায়ের সাথে বিস্কুট খাব।",
    "ফল খাওয়া স্বাস্থ্যের জন্য ভালো।",
    "সবজি কাটা শেষ হয়ে গেছে।",
    "মশলা দিয়ে রান্না করলে স্বাদ বাড়ে।",
    "ঠান্ডা পানি খেতে দিন।",
    
    # Technology
    "ফোনের ব্যাটারি শেষ হয়ে গেছে।",
    "ল্যাপটপ চার্জ দিতে হবে।",
    "ইন্টারনেট স্পিড কম।",
    "অ্যাপটা ডাউনলোড হচ্ছে না।",
    "পাসওয়ার্ড চেঞ্জ করুন।",
    "সফটওয়্যার আপডেট করা দরকার।",
    "ক্যামেরার কোয়ালিটি ভালো।",
    "স্ক্রিন ভেঙে গেছে।",
    
    # Shopping
    "এই জামাটা কত দাম?",
    "কোন সাইজ লাগবে?",
    "অন্য রঙে আছে কি?",
    "ফিটিং রুম কোথায়?",
    "ক্যাশে দেব নাকি কার্ডে?",
    "ছাড় দেওয়া যাবে কি?",
    "প্যাকেটে ভরে দিন।",
    "বিল কাটুন প্লিজ।",
    
    # Entertainment
    "সিনেমা দেখতে যাব।",
    "নতুন গান শুনেছেন?",
    "বই পড়া শেষ হয়নি।",
    "খেলা দেখতে ভালো লাগে।",
    "নাটক শুরু হয়ে গেছে।",
    "কনসার্টের টিকিট পেয়েছি।",
    "গান গাইতে পছন্দ করি।",
    "ছুটিতে কোথায় যাবেন?",
    
    # News and current affairs
    "আজকের খবর কী?",
    "নির্বাচন কবে হবে?",
    "দাম বেড়ে গেছে।",
    "ট্রাফিক জ্যাম হয়েছে।",
    "স্কুল বন্ধ রইলো আজ।",
    "নতুন নিয়ম চালু হয়েছে।",
    "সরকার ঘোষণা করেছে।",
    "ব্যাংক ছুটি থাকবে।",
    
    # Formal expressions
    "আপনাকে স্বাগত জানাই।",
    "সভা শুরু করা যাক।",
    "ধন্যবাদ আপনাদের সবাইকে।",
    "বিষয়টি বিবেচনা করব।",
    "সিদ্ধান্ত পরে জানাব।",
    "নিয়ম মানা বাধ্যতামূলক।",
    "আবেদন জমা দিয়েছি।",
    "অনুমোদন পেয়ে গেছি।",
    
    # Casual expressions
    "চল যাই এখন।",
    "থাক, পরে হবে।",
    "মজা হল আজ।",
    "বোরিং লাগছে।",
    "ঘুম পাচ্ছে।",
    "ক্ষুধা লেগেছে।",
    "তৃষ্ণা পেয়েছে।",
    "ক্লান্ত হয়ে গেছি।",
    
    # Sports
    "ক্রিকেট খেলা হবে।",
    "ফুটবল ম্যাচ কখন?",
    "টিম জিতে গেছে।",
    "হেরে গেছে আমরা।",
    "খেলোয়াড় ভালো খেলেছে।",
    "গোল হয়ে গেছে।",
    "রান করতে হবে।",
    "আউট হয়ে গেছে।",
    
    # Instructions
    "প্রথমে এটা করুন।",
    "তারপর এদিকে আসুন।",
    "শেষে এটা দেখুন।",
    "সাবধানে চলবেন।",
    "মনোযোগ দিন।",
    "ভালো করে শুনুন।",
    "দেখে নিন একবার।",
    "পড়ে বুঝে নিন।",
    
    # Descriptions
    "জায়গাটা অনেক সুন্দর।",
    "মানুষটা ভালো।",
    "কাজটা কঠিন ছিল।",
    "রাস্তা অনেক লম্বা।",
    "ঘরটা পরিষ্কার আছে।",
    "বাগানে ফুল ফুটেছে।",
    "আকাশ পরিষ্কার আজ।",
    "রাত অনেক গভীর।",
    
    # Opinions
    "আমার মতে এটা ভালো।",
    "আমি মনে করি ঠিক আছে।",
    "এটা ঠিক মনে হয় না।",
    "সম্ভবত হবে।",
    "হয়তো পারব।",
    "নিশ্চিত নই।",
    "আশা করি হবে।",
    "সন্দেহ আছে।",
    
    # Medical/Health
    "জ্বর এসেছে।",
    "মাথা ব্যথা করছে।",
    "পেট খারাপ হয়েছে।",
    "সর্দি হয়ে গেছে।",
    "ওষুধ খেতে হবে।",
    "ডাক্তার দেখাতে হবে।",
    "বিশ্রাম নিতে হবে।",
    "সুস্থ হয়ে যাব।",
    
    # Banking/Finance
    "টাকা তুলতে হবে।",
    "জমা দিতে এসেছি।",
    "ব্যালেন্স চেক করুন।",
    "লোন নিতে চাই।",
    "সুদের হার কত?",
    "ইএমআই কত হবে?",
    "অ্যাকাউন্ট খুলতে চাই।",
    "পাসবুক আপডেট করুন।",
    
    # More natural expressions
    "কী করছেন এখন?",
    "কোথায় যাচ্ছেন?",
    "কখন ফিরবেন?",
    "কেমন লাগছে?",
    "কী খবর আপনার?",
    "সব ঠিক তো?",
    "কাজ শেষ হয়েছে?",
    "বাড়ি পৌঁছেছেন?",
    
    # Extended questions
    "এটা কি সত্যি?",
    "কেন এমন হল?",
    "কীভাবে সম্ভব?",
    "কবে থেকে শুরু?",
    "কার কাছে পাব?",
    "কোথায় পাওয়া যাবে?",
    "কোনটা বেছে নেব?",
    "কত দিন লাগবে?",
    
    # Exclamations
    "দারুণ হয়েছে!",
    "অসাধারণ কাজ!",
    "চমৎকার খবর!",
    "আশ্চর্য ব্যাপার!",
    "অবিশ্বাস্য!",
    "কী সুন্দর!",
    "কত সুখের খবর!",
    "বেশ মজার!",
    
    # Polite requests
    "একটু সাহায্য করবেন?",
    "জলটা দেবেন প্লিজ?",
    "বসতে পারি কি?",
    "যেতে পারি এখন?",
    "দেরি হবে একটু।",
    "অপেক্ষা করতে বলুন।",
    "পরে আসব আবার।",
    "কাল দেখা হবে।",
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


def generate_sentences_from_templates(count: int = 15000) -> List[Sentence]:
    """
    Generate additional natural Bengali sentences from templates.
    This helps reach the 20,000 sentence target for studio-quality TTS.
    """
    import random
    random.seed(42)  # Reproducible generation
    
    generated = []
    seen_texts = set()
    
    # Generate from templates - increase attempts for more variety
    template_count = 0
    max_attempts = count * 5  # More attempts to find unique combinations
    while len(generated) < count * 0.8 and template_count < max_attempts:
        template = random.choice(SENTENCE_TEMPLATES)
        template_count += 1
        
        try:
            # Fill template with random words
            text = template
            if '{noun}' in text:
                text = text.replace('{noun}', random.choice(TEMPLATE_NOUNS), 1)
            if '{noun}' in text:
                text = text.replace('{noun}', random.choice(TEMPLATE_NOUNS), 1)
            if '{noun2}' in text:
                text = text.replace('{noun2}', random.choice(TEMPLATE_NOUNS))
            if '{verb}' in text:
                text = text.replace('{verb}', random.choice(TEMPLATE_VERBS))
            if '{adjective}' in text:
                text = text.replace('{adjective}', random.choice(TEMPLATE_ADJECTIVES))
            if '{name}' in text:
                text = text.replace('{name}', random.choice(TEMPLATE_NAMES))
            if '{place}' in text:
                text = text.replace('{place}', random.choice(TEMPLATE_PLACES))
            if '{food}' in text:
                text = text.replace('{food}', random.choice(TEMPLATE_FOODS))
            if '{time}' in text:
                text = text.replace('{time}', random.choice(TEMPLATE_TIMES))
            if '{weather}' in text:
                text = text.replace('{weather}', random.choice(TEMPLATE_WEATHER))
            if '{activity}' in text:
                text = text.replace('{activity}', random.choice(TEMPLATE_ACTIVITIES))
            
            # Skip if already generated or has unfilled placeholders
            if '{' in text or text in seen_texts:
                continue
            
            seen_texts.add(text)
            sent = Sentence(
                id=f"gen_{len(generated)+1:05d}",
                text=text,
                original_text=text,
                source="generated_template",
                style="conversational"
            )
            generated.append(sent)
        except Exception:
            continue
    
    # Add additional pre-written sentences
    for i, text in enumerate(ADDITIONAL_SENTENCES, 1):
        if text not in seen_texts:
            seen_texts.add(text)
            sent = Sentence(
                id=f"additional_{i:04d}",
                text=text,
                original_text=text,
                source="additional_sentences",
                style="conversational"
            )
            generated.append(sent)
    
    # Add coverage sentences
    for i, text in enumerate(COVERAGE_SENTENCES, 1):
        if text not in seen_texts:
            seen_texts.add(text)
            sent = Sentence(
                id=f"coverage_{i:03d}",
                text=text,
                original_text=text,
                source="coverage_sentences",
                style="balanced"
            )
            generated.append(sent)
    
    return generated


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
                          target_count: int = 20000) -> List[Sentence]:
    """
    Select optimal subset for TTS training.
    
    For RTX 5060 Ti 16GB (Studio Quality):
    - Target: 20,000 sentences for realistic human-like TTS
    - Estimated: 25-30 hours of audio
    """
    print(f"\n🎯 Selecting optimal {target_count} sentences for studio quality...")
    
    # Step 1: Remove very short or very long sentences
    filtered = filter_by_length(sentences, min_chars=5, max_chars=150)
    print(f"   After length filter: {len(filtered)}")
    
    # Step 2: Calculate quality scores
    for sent in filtered:
        sent.quality_score = calculate_quality_score(sent)
    
    # Step 3: Filter by quality score (lower threshold for more sentences)
    # Use lower threshold to include more sentences for 20k target
    quality_filtered = filter_by_quality(filtered, min_score=15.0)
    print(f"   After quality filter: {len(quality_filtered)}")
    
    # Step 4: Ensure phoneme coverage
    phoneme_covered = ensure_phoneme_coverage(quality_filtered, target_coverage=20)
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
    """Main optimization pipeline for studio-quality Bengali TTS."""
    print("="*60)
    print("🚀 Bengali TTS Dataset Optimizer (Studio Quality)")
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
    
    # Generate additional sentences from templates for studio quality
    print("\n🔧 Generating additional sentences from templates...")
    generated_sentences = generate_sentences_from_templates(count=15000)
    print(f"   Generated {len(generated_sentences):,} additional sentences")
    all_sentences.extend(generated_sentences)
    
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
    # For RTX 5060 Ti 16GB - STUDIO QUALITY requires 20,000 sentences (~25-30 hours)
    TARGET_COUNT = 20000
    optimal_sentences = select_optimal_dataset(unique_sentences, target_count=TARGET_COUNT)
    
    # If we don't have enough, include all available sentences
    if len(optimal_sentences) < TARGET_COUNT:
        print(f"\n⚠️  Only {len(optimal_sentences):,} sentences available after filtering.")
        print(f"   Including all quality sentences to maximize dataset size...")
        # Lower quality threshold and include more
        for sent in unique_sentences:
            sent.quality_score = calculate_quality_score(sent)
        quality_filtered = [s for s in unique_sentences if s.quality_score >= 15.0]
        optimal_sentences = sorted(quality_filtered, key=lambda x: -x.quality_score)[:TARGET_COUNT]
    
    # Calculate final stats
    final_stats = calculate_dataset_stats(optimal_sentences)
    print_stats(final_stats, "Optimized Dataset Statistics (Studio Quality)")
    
    # Save outputs
    print("\n💾 Saving optimized dataset...")
    save_optimized_dataset(optimal_sentences, str(output_dir))
    
    # Summary
    print("\n" + "="*60)
    print("✅ Studio Quality Dataset Optimization Complete!")
    print("="*60)
    print(f"\n📁 Output directory: {output_dir}")
    print(f"📝 Total optimized sentences: {len(optimal_sentences):,}")
    print(f"⏱️  Estimated recording time: {final_stats.total_characters / 5 / 3600:.1f} hours")
    print("\n💡 Next steps:")
    print("   1. Review 'optimized_sentences.txt' for the final text")
    print("   2. Use 'recording_prompts.csv' with your recording app")
    print("   3. Check 'dataset_report.txt' for detailed statistics")
    print("   4. Run: python app.py to start recording")
    print("\n🎯 For RTX 5060 Ti 16GB (Studio Quality):")
    print("   - 20,000 sentences for realistic human-like TTS")
    print("   - Expected recording time: 25-30 hours")
    print("   - Expected training time: 48-72 hours")
    print("   - Recommended batch size: 16 with gradient accumulation 4")


if __name__ == '__main__':
    main()
