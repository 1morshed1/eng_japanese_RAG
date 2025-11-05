# app/utils/language_detector.py
"""
Language detection utilities for Healthcare RAG Assistant.

Provides robust language detection with fallback mechanisms for
English and Japanese text processing.
"""

import logging
from typing import Optional
from langdetect import detect, detect_langs, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException

# Initialize logger
logger = logging.getLogger(__name__)

# Set seed for consistent results
DetectorFactory.seed = 0

# Supported languages
SUPPORTED_LANGUAGES = {'en', 'ja'}
DEFAULT_LANGUAGE = 'en'

# Minimum confidence threshold for language detection
MIN_CONFIDENCE = 0.7


def detect_language(
    text: str,
    fallback: str = DEFAULT_LANGUAGE,
    min_confidence: float = MIN_CONFIDENCE
) -> str:
    """
    Detect the language of input text with robust fallback mechanisms.
    
    Attempts to detect language using langdetect library with confidence
    scoring. Falls back to character-based detection (checking for Japanese
    characters) if primary detection fails or confidence is too low.
    
    Args:
        text: Text to detect language from
        fallback: Language to return if detection fails (default: 'en')
        min_confidence: Minimum confidence threshold for langdetect (0.0-1.0)
        
    Returns:
        str: Language code ('en' or 'ja')
        
    Raises:
        ValueError: If text is None or empty
        
    Examples:
        >>> detect_language("Hello, how are you?")
        'en'
        
        >>> detect_language("こんにちは、元気ですか？")
        'ja'
        
        >>> detect_language("2型糖尿病の管理")
        'ja'
        
        >>> detect_language("Type 2 diabetes management")
        'en'
        
    Notes:
        - Prioritizes Japanese detection for medical terms
        - Uses character-based fallback for short texts
        - Logs warnings when fallback mechanisms are used
    """
    # Validate input
    if text is None:
        raise ValueError("Text cannot be None")
    
    if not text or not text.strip():
        logger.warning("Empty text provided for language detection")
        return fallback
    
    # Clean text for detection
    text_cleaned = text.strip()
    
    # For very short text, use character-based detection
    if len(text_cleaned) < 10:
        logger.debug(
            "Text too short for reliable detection, using character-based fallback",
            extra={"text_length": len(text_cleaned)}
        )
        return _detect_by_characters(text_cleaned, fallback)
    
    try:
        # Try detection with confidence scoring
        detected_langs = detect_langs(text_cleaned)
        
        if not detected_langs:
            logger.warning("No languages detected")
            return _detect_by_characters(text_cleaned, fallback)
        
        # Get most probable language
        top_lang = detected_langs[0]
        detected_lang = top_lang.lang
        confidence = top_lang.prob
        
        logger.debug(
            "Language detected",
            extra={
                "language": detected_lang,
                "confidence": round(confidence, 3),
                "text_preview": text_cleaned[:50]
            }
        )
        
        # Check if detected language is supported
        if detected_lang not in SUPPORTED_LANGUAGES:
            logger.warning(
                "Unsupported language detected, using fallback",
                extra={
                    "detected_language": detected_lang,
                    "confidence": round(confidence, 3)
                }
            )
            return _detect_by_characters(text_cleaned, fallback)
        
        # Check confidence threshold
        if confidence < min_confidence:
            logger.warning(
                "Low confidence detection, using character-based fallback",
                extra={
                    "language": detected_lang,
                    "confidence": round(confidence, 3),
                    "threshold": min_confidence
                }
            )
            return _detect_by_characters(text_cleaned, fallback)
        
        return detected_lang
        
    except LangDetectException as e:
        logger.warning(
            "Language detection exception, using fallback",
            extra={
                "error": str(e),
                "text_preview": text_cleaned[:50]
            }
        )
        return _detect_by_characters(text_cleaned, fallback)
    
    except Exception as e:
        logger.error(
            "Unexpected error in language detection",
            extra={
                "error": str(e),
                "error_type": type(e).__name__
            },
            exc_info=True
        )
        return fallback


def _detect_by_characters(text: str, fallback: str = DEFAULT_LANGUAGE) -> str:
    """
    Detect language based on character ranges (fallback method).
    
    Checks for presence of Japanese characters (hiragana, katakana, kanji).
    If found, returns 'ja', otherwise returns fallback language.
    
    Args:
        text: Text to analyze
        fallback: Language to return if no Japanese characters found
        
    Returns:
        str: 'ja' if Japanese characters detected, otherwise fallback
        
    Notes:
        This is a simple heuristic and may not work for mixed-language text.
    """
    if _contains_japanese_chars(text):
        logger.debug(
            "Japanese characters detected",
            extra={"text_preview": text[:50]}
        )
        return 'ja'
    
    logger.debug(
        "No Japanese characters found, using fallback",
        extra={"fallback_language": fallback}
    )
    return fallback


def _contains_japanese_chars(text: str) -> bool:
    """
    Check if text contains Japanese characters.
    
    Checks for:
    - Hiragana (U+3040 - U+309F)
    - Katakana (U+30A0 - U+30FF)
    - Kanji/CJK Unified Ideographs (U+4E00 - U+9FFF)
    - Katakana Phonetic Extensions (U+31F0 - U+31FF)
    
    Args:
        text: Text to check
        
    Returns:
        bool: True if Japanese characters found, False otherwise
        
    Examples:
        >>> _contains_japanese_chars("hello")
        False
        
        >>> _contains_japanese_chars("こんにちは")
        True
        
        >>> _contains_japanese_chars("糖尿病")
        True
    """
    for char in text:
        code = ord(char)
        
        # Hiragana
        if 0x3040 <= code <= 0x309F:
            return True
        
        # Katakana
        if 0x30A0 <= code <= 0x30FF:
            return True
        
        # Katakana Phonetic Extensions
        if 0x31F0 <= code <= 0x31FF:
            return True
        
        # Kanji (CJK Unified Ideographs)
        if 0x4E00 <= code <= 0x9FFF:
            return True
        
        # CJK Compatibility Ideographs
        if 0xF900 <= code <= 0xFAFF:
            return True
        
        # Half-width Katakana
        if 0xFF65 <= code <= 0xFF9F:
            return True
    
    return False


def is_japanese(text: str) -> bool:
    """
    Check if text is primarily Japanese.
    
    Convenience function that returns True if detected language is Japanese.
    
    Args:
        text: Text to check
        
    Returns:
        bool: True if Japanese, False otherwise
        
    Example:
        >>> is_japanese("こんにちは")
        True
        
        >>> is_japanese("Hello")
        False
    """
    return detect_language(text) == 'ja'


def is_english(text: str) -> bool:
    """
    Check if text is primarily English.
    
    Convenience function that returns True if detected language is English.
    
    Args:
        text: Text to check
        
    Returns:
        bool: True if English, False otherwise
        
    Example:
        >>> is_english("Hello")
        True
        
        >>> is_english("こんにちは")
        False
    """
    return detect_language(text) == 'en'


def get_language_name(lang_code: str) -> str:
    """
    Get full language name from code.
    
    Args:
        lang_code: Two-letter language code ('en' or 'ja')
        
    Returns:
        str: Full language name
        
    Example:
        >>> get_language_name('en')
        'English'
        
        >>> get_language_name('ja')
        'Japanese'
    """
    language_names = {
        'en': 'English',
        'ja': 'Japanese'
    }
    return language_names.get(lang_code, lang_code.upper())