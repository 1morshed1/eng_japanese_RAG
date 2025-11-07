# app/services/translation_service.py
"""
Translation service for English-Japanese bidirectional translation.

Provides reliable translation using Helsinki-NLP MarianMT models with
lazy loading, batch processing, and GPU support.
"""

import logging
import time
from typing import List, Optional, Dict, Any
from transformers import MarianMTModel, MarianTokenizer
import torch
from app.config import settings
from app.utils.language_detector import detect_language

logger = logging.getLogger(__name__)


class TranslationService:
    """
    Service for handling translation between English and Japanese.
    
    Features:
        - Lazy loading of translation models
        - Bidirectional EN↔JA translation
        - Batch processing for efficiency
        - GPU support with automatic detection
        - Error handling and fallbacks
    
    Example:
        >>> service = TranslationService()
        >>> result = service.translate("Hello", "en", "ja")
        >>> print(result)  # こんにちは
    """
    
    def __init__(self, device: Optional[str] = None):
        """
        Initialize translation service with lazy loading.
        
        Args:
            device: Device to use ('cuda', 'cpu', or None for auto-detect)
        """
        # Model names for Helsinki-NLP MarianMT models
        self.en_ja_model_name = "Helsinki-NLP/opus-mt-en-jap"
        self.ja_en_model_name = "Helsinki-NLP/opus-mt-jap-en"
        
        # Device configuration
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        # Lazy loading - models loaded on first use
        self.en_ja_model: Optional[MarianMTModel] = None
        self.en_ja_tokenizer: Optional[MarianTokenizer] = None
        self.ja_en_model: Optional[MarianMTModel] = None
        self.ja_en_tokenizer: Optional[MarianTokenizer] = None
        
        # Model loading flags
        self.en_ja_loaded = False
        self.ja_en_loaded = False
        
        # Translation statistics
        self.translation_count = 0
        self.error_count = 0
        
        logger.info(
            "TranslationService initialized",
            extra={
                "device": self.device,
                "en_ja_model": self.en_ja_model_name,
                "ja_en_model": self.ja_en_model_name
            }
        )

    def _load_en_ja_model(self):
        """Load English to Japanese translation model and tokenizer."""
        if self.en_ja_loaded:
            return
            
        try:
            logger.info(f"Loading English→Japanese model: {self.en_ja_model_name}")
            
            self.en_ja_tokenizer = MarianTokenizer.from_pretrained(
                self.en_ja_model_name
            )
            self.en_ja_model = MarianMTModel.from_pretrained(
                self.en_ja_model_name
            )
            
            # Set model to evaluation mode and move to device
            self.en_ja_model.eval()
            self.en_ja_model = self.en_ja_model.to(self.device)
            
            self.en_ja_loaded = True
            logger.info(f"English→Japanese model loaded on {self.device}")
            
        except Exception as e:
            logger.error(
                "Failed to load English→Japanese model",
                extra={"error": str(e)},
                exc_info=True
            )
            raise RuntimeError(f"Could not load translation model: {str(e)}")

    def _load_ja_en_model(self):
        """Load Japanese to English translation model and tokenizer."""
        if self.ja_en_loaded:
            return
            
        try:
            logger.info(f"Loading Japanese→English model: {self.ja_en_model_name}")
            
            self.ja_en_tokenizer = MarianTokenizer.from_pretrained(
                self.ja_en_model_name
            )
            self.ja_en_model = MarianMTModel.from_pretrained(
                self.ja_en_model_name
            )
            
            # Set model to evaluation mode and move to device
            self.ja_en_model.eval()
            self.ja_en_model = self.ja_en_model.to(self.device)
            
            self.ja_en_loaded = True
            logger.info(f"Japanese→English model loaded on {self.device}")
            
        except Exception as e:
            logger.error(
                "Failed to load Japanese→English model",
                extra={"error": str(e)},
                exc_info=True
            )
            raise RuntimeError(f"Could not load translation model: {str(e)}")

    def translate(
        self, 
        text: str, 
        source_lang: str, 
        target_lang: str,
        max_length: int = 512
    ) -> str:
        """
        Translate text between English and Japanese.
        
        Args:
            text: Text to translate
            source_lang: Source language ('en' or 'ja')
            target_lang: Target language ('en' or 'ja')
            max_length: Maximum length for translation
            
        Returns:
            Translated text
            
        Raises:
            ValueError: For unsupported language pairs or invalid input
            RuntimeError: If translation fails
            
        Example:
            >>> result = service.translate("Hello", "en", "ja")
            >>> print(result)  # こんにちは
        """
        # Validate inputs
        if not text or not text.strip():
            return text if text is not None else ""
        
        if source_lang == target_lang:
            return text.strip()
        
        # Validate language codes
        if source_lang not in ['en', 'ja'] or target_lang not in ['en', 'ja']:
            raise ValueError(
                f"Unsupported language pair: {source_lang} → {target_lang}. "
                "Only 'en' and 'ja' are supported."
            )
        
        # Check text length (rough estimate: 1 token ≈ 4 chars)
        max_chars = max_length * 4
        if len(text) > max_chars:
            logger.warning(
                "Text too long, truncating",
                extra={
                    "text_length": len(text),
                    "max_chars": max_chars,
                    "truncated": True
                }
            )
            text = text[:max_chars]
        
        start_time = time.time()

        try:
            if source_lang == 'en' and target_lang == 'ja':
                self._load_en_ja_model()
                result = self._translate_with_model(
                    text, self.en_ja_model, self.en_ja_tokenizer, max_length
                )
                
            elif source_lang == 'ja' and target_lang == 'en':
                self._load_ja_en_model()
                result = self._translate_with_model(
                    text, self.ja_en_model, self.ja_en_tokenizer, max_length
                )
                
            else:
                raise ValueError(f"Unsupported translation direction: {source_lang} → {target_lang}")

            self.translation_count += 1
            translation_time = time.time() - start_time
            
            logger.debug(
                "Translation completed",
                extra={
                    "source_lang": source_lang,
                    "target_lang": target_lang,
                    "text_length": len(text),
                    "translation_time": round(translation_time, 3),
                    "device": self.device
                }
            )
            
            return result
            
        except Exception as e:
            self.error_count += 1
            logger.error(
                "Translation failed",
                extra={
                    "source_lang": source_lang,
                    "target_lang": target_lang,
                    "text_length": len(text),
                    "error": str(e)
                },
                exc_info=True
            )
            raise RuntimeError(f"Translation failed: {str(e)}")

    def _translate_with_model(
        self, 
        text: str, 
        model: MarianMTModel, 
        tokenizer: MarianTokenizer,
        max_length: int
    ) -> str:
        """
        Perform translation using a loaded model and tokenizer.
        
        Args:
            text: Text to translate
            model: Loaded MarianMT model
            tokenizer: Loaded tokenizer
            max_length: Maximum sequence length
            
        Returns:
            Translated text
        """
        try:
            # Tokenize input text
            inputs = tokenizer(
                text, 
                return_tensors="pt", 
                padding=True, 
                truncation=True,
                max_length=max_length
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate translation
            with torch.no_grad():
                translated_tokens = model.generate(
                    **inputs,
                    max_length=max_length,
                    num_beams=4,  # Beam search for better quality
                    early_stopping=True
                )
            
            # Decode translation
            result = tokenizer.decode(
                translated_tokens[0], 
                skip_special_tokens=True
            )
            
            return result.strip()
            
        except Exception as e:
            logger.error(f"Model translation failed: {str(e)}", exc_info=True)
            raise

    def batch_translate(
        self,
        texts: List[str],
        source_lang: str,
        target_lang: str,
        max_length: int = 512,
        batch_size: int = 8
    ) -> List[str]:
        """
        Translate multiple texts in batches for efficiency.
        
        Args:
            texts: List of texts to translate
            source_lang: Source language ('en' or 'ja')
            target_lang: Target language ('en' or 'ja')
            max_length: Maximum sequence length per text
            batch_size: Number of texts to process in batch
            
        Returns:
            List of translated texts in same order as input
            
        Example:
            >>> texts = ["Hello", "How are you?", "Good morning"]
            >>> results = service.batch_translate(texts, "en", "ja")
        """
        if not texts:
            return []
        
        # Handle same language
        if source_lang == target_lang:
            return [text.strip() if text else "" for text in texts]
        
        # Validate language pair
        if source_lang not in ['en', 'ja'] or target_lang not in ['en', 'ja']:
            raise ValueError(f"Unsupported language pair: {source_lang} → {target_lang}")
        
        # Filter and track empty texts
        texts_to_translate = []
        empty_indices = set()
        for i, text in enumerate(texts):
            if text and text.strip():
                texts_to_translate.append(text.strip())
            else:
                empty_indices.add(i)
        
        if not texts_to_translate:
            return [""] * len(texts)
        
        # Load appropriate model
        if source_lang == 'en' and target_lang == 'ja':
            self._load_en_ja_model()
            model = self.en_ja_model
            tokenizer = self.en_ja_tokenizer
        elif source_lang == 'ja' and target_lang == 'en':
            self._load_ja_en_model()
            model = self.ja_en_model
            tokenizer = self.ja_en_tokenizer
        else:
            raise ValueError(f"Unsupported language pair: {source_lang} → {target_lang}")
        
        results = []
        
        try:
            # Process in batches
            for i in range(0, len(texts_to_translate), batch_size):
                batch = texts_to_translate[i:i + batch_size]
                
                # Tokenize entire batch at once
                inputs = tokenizer(
                    batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=max_length
                )
                
                # Move to device
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Generate translations
                with torch.no_grad():
                    translated_tokens = model.generate(
                        **inputs,
                        max_length=max_length,
                        num_beams=4,
                        early_stopping=True
                    )
                
                # Decode each translation
                batch_results = [
                    tokenizer.decode(tokens, skip_special_tokens=True).strip()
                    for tokens in translated_tokens
                ]
                
                results.extend(batch_results)
                self.translation_count += len(batch_results)
            
            logger.debug(
                "Batch translation completed",
                extra={
                    "source_lang": source_lang,
                    "target_lang": target_lang,
                    "text_count": len(texts_to_translate),
                    "batch_size": batch_size
                }
            )
            
            # Reconstruct full results with empty strings in original positions
            final_results = []
            result_idx = 0
            for i in range(len(texts)):
                if i in empty_indices:
                    final_results.append("")
                else:
                    final_results.append(results[result_idx])
                    result_idx += 1
            
            return final_results
            
        except Exception as e:
            self.error_count += len(texts_to_translate)
            logger.error(
                "Batch translation failed, falling back to individual",
                extra={"error": str(e)},
                exc_info=True
            )
            # Fallback: translate individually
            fallback_results = []
            for text in texts:
                try:
                    if text and text.strip():
                        fallback_results.append(
                            self.translate(text, source_lang, target_lang, max_length)
                        )
                    else:
                        fallback_results.append("")
                except Exception:
                    fallback_results.append(text if text else "")
            return fallback_results

    def detect_and_translate(self, text: str, target_lang: str) -> str:
        """
        Detect language and translate to target language if needed.
        
        Args:
            text: Text to potentially translate
            target_lang: Target language ('en' or 'ja')
            
        Returns:
            Translated text if language differs, otherwise original text
            
        Example:
            >>> result = service.detect_and_translate("Hello", "ja")
            >>> print(result)  # こんにちは
        """
        if not text or not text.strip():
            return text if text is not None else ""
        
        source_lang = detect_language(text)
        
        if source_lang == target_lang:
            return text
        
        try:
            return self.translate(text, source_lang, target_lang)
        except Exception as e:
            logger.warning(
                f"Auto-translation failed, returning original text: {e}",
                extra={"detected_lang": source_lang, "target_lang": target_lang}
            )
            return text

    def warmup(self):
        """
        Warm up translation models with dummy translations.
        
        Ensures models are fully loaded and compiled for better
        first-call performance.
        """
        logger.info("Warming up translation models...")
        
        try:
            # Warm up EN→JA
            self._load_en_ja_model()
            _ = self._translate_with_model(
                "test",
                self.en_ja_model,
                self.en_ja_tokenizer,
                50
            )
            
            # Warm up JA→EN
            self._load_ja_en_model()
            _ = self._translate_with_model(
                "テスト",
                self.ja_en_model,
                self.ja_en_tokenizer,
                50
            )
            
            logger.info("Translation models warmed up successfully")
            
        except Exception as e:
            logger.warning(f"Translation warmup failed: {e}")

    def unload_models(self):
        """
        Unload models from memory to free resources.
        
        Useful for testing or when switching configurations.
        """
        if self.en_ja_model is not None:
            del self.en_ja_model
            del self.en_ja_tokenizer
            self.en_ja_model = None
            self.en_ja_tokenizer = None
            self.en_ja_loaded = False
        
        if self.ja_en_model is not None:
            del self.ja_en_model
            del self.ja_en_tokenizer
            self.ja_en_model = None
            self.ja_en_tokenizer = None
            self.ja_en_loaded = False
        
        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("Translation models unloaded from memory")

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded translation models."""
        return {
            "en_ja_loaded": self.en_ja_loaded,
            "ja_en_loaded": self.ja_en_loaded,
            "en_ja_model": self.en_ja_model_name,
            "ja_en_model": self.ja_en_model_name,
            "translation_count": self.translation_count,
            "error_count": self.error_count,
            "device": self.device,
            "cuda_available": torch.cuda.is_available()
        }

    def health_check(self) -> Dict[str, Any]:
        """Perform health check on translation service."""
        try:
            # Test translation in both directions
            test_text_en = "Hello, how are you?"
            test_text_ja = "こんにちは、元気ですか？"
            
            # Test EN→JA
            self._load_en_ja_model()
            result_ja = self._translate_with_model(
                test_text_en, self.en_ja_model, self.en_ja_tokenizer, 100
            )
            
            # Test JA→EN
            self._load_ja_en_model()
            result_en = self._translate_with_model(
                test_text_ja, self.ja_en_model, self.ja_en_tokenizer, 100
            )
            
            return {
                "status": "healthy",
                "en_ja_loaded": self.en_ja_loaded,
                "ja_en_loaded": self.ja_en_loaded,
                "test_translation_works": True,
                "device": self.device,
                "translation_count": self.translation_count,
                "error_count": self.error_count
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "en_ja_loaded": self.en_ja_loaded,
                "ja_en_loaded": self.ja_en_loaded,
                "error": str(e)
            }

    def preload_models(self):
        """
        Preload both translation models (useful for warm startup).
        
        Alias for warmup() for backward compatibility.
        """
        logger.info("Preloading translation models...")
        self._load_en_ja_model()
        self._load_ja_en_model()
        logger.info("Translation models preloaded successfully")
        
        # Also do warmup
        self.warmup()


# Global instance for dependency injection
translation_service = TranslationService()


# For testing purposes
if __name__ == "__main__":
    print("Testing Translation Service...\n")
    
    service = TranslationService()
    
    print("=" * 60)
    print("TEST 1: English to Japanese")
    print("=" * 60)
    result_ja = service.translate("Hello, how are you today?", "en", "ja")
    print(f"EN→JA: {result_ja}\n")
    
    print("=" * 60)
    print("TEST 2: Japanese to English")
    print("=" * 60)
    result_en = service.translate("こんにちは、今日は元気ですか？", "ja", "en")
    print(f"JA→EN: {result_en}\n")
    
    print("=" * 60)
    print("TEST 3: Batch Translation")
    print("=" * 60)
    texts_en = ["Hello", "How are you?", "Good morning"]
    results_ja = service.batch_translate(texts_en, "en", "ja")
    print(f"Batch EN→JA:")
    for orig, trans in zip(texts_en, results_ja):
        print(f"  {orig} → {trans}")
    print()
    
    print("=" * 60)
    print("TEST 4: Auto-detect and Translate")
    print("=" * 60)
    result_auto = service.detect_and_translate("Hello", "ja")
    print(f"Auto (Hello → JA): {result_auto}\n")
    
    print("=" * 60)
    print("TEST 5: Model Info")
    print("=" * 60)
    info = service.get_model_info()
    print(f"Model info: {info}\n")
    
    print("=" * 60)
    print("TEST 6: Health Check")
    print("=" * 60)
    health = service.health_check()
    print(f"Health check: {health}\n")
    
    print("All tests completed!")