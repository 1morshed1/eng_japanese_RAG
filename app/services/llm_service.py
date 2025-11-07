# app/services/llm_service.py
"""
Mock LLM service for generating structured medical responses.

Provides template-based response generation with medical domain knowledge
for both English and Japanese languages.
"""

import logging
import re
import time  
from typing import List, Dict, Optional, Set
import nltk
from app.config import settings
from app.utils.language_detector import detect_language
from app.services.translation_service import translation_service

logger = logging.getLogger(__name__)

# Download NLTK data if not present
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)


class LLMService:
    """
    Service for generating structured medical responses using retrieved documents.
    
    Provides mock LLM functionality with template-based response generation,
    medical keyword extraction, and bilingual support.
    
    Features:
        - Template-based response generation
        - Medical domain knowledge
        - Bilingual support (English/Japanese)
        - Source tracking
        - Medical disclaimers
    
    Example:
        >>> service = LLMService()
        >>> response = service.generate_response(query, documents, "en")
        >>> print(response["response"])
    """
    
    def __init__(self):
        """Initialize LLM service with response templates and medical patterns."""
        
        self.response_templates = {
            "en": {
                "introduction": "Based on the retrieved medical guidelines regarding {topic}:",
                "findings_header": "Key Findings:",
                "recommendations_header": "Recommendations:",
                "disclaimer": "\n⚠️ Disclaimer: This information is AI-generated based on retrieved documents. Please consult healthcare professionals for medical advice.",
                "no_results": "I couldn't find specific information about '{topic}' in the available medical guidelines. Please consult a healthcare provider for personalized medical advice.",
                "summary_template": "The guidelines indicate that {summary}",
                "error": "I apologize, but an error occurred while generating the response. Please try again later."
            },
            "ja": {
                "introduction": "{topic}に関する医療ガイドラインによると：",
                "findings_header": "主な知見：",
                "recommendations_header": "推奨事項：",
                "disclaimer": "\n⚠️ 免責事項：この情報は取得した文書に基づくAI生成です。医療アドバイスについては医療専門家にご相談ください。",
                "no_results": "利用可能な医療ガイドラインに'{topic}'に関する具体的な情報が見つかりませんでした。個別の医療アドバイスについては医療提供者にご相談ください。",
                "summary_template": "ガイドラインによれば、{summary}",
                "error": "申し訳ございません。応答の生成中にエラーが発生しました。しばらくしてからもう一度お試しください。"
            }
        }
        
        # Medical term patterns for extraction
        self.medical_patterns = {
            "recommendation_keywords": {
                "en": ["recommend", "suggest", "should", "advise", "indicated", "preferable", "appropriate"],
                "ja": ["推奨", "勧告", "すべき", "提案", "適応", "望ましい"]
            },
            "treatment_keywords": {
                "en": ["treatment", "therapy", "medication", "dose", "management", "intervention"],
                "ja": ["治療", "療法", "薬物", "投与量", "管理", "介入"]
            },
            "diagnosis_keywords": {
                "en": ["diagnosis", "symptoms", "signs", "test", "evaluation", "assessment"],
                "ja": ["診断", "症状", "徴候", "検査", "評価", "判定"]
            }
        }
        
        # Medical topics with synonyms
        self.medical_topics = {
            "en": {
                "diabetes": ["diabetes", "diabetic", "blood sugar", "glucose", "insulin", "glycemic"],
                "hypertension": ["hypertension", "high blood pressure", "blood pressure", "bp"],
                "heart disease": ["heart disease", "cardiac", "cardiovascular", "coronary", "cvd"],
                "cancer": ["cancer", "tumor", "oncology", "malignant", "carcinoma"],
                "asthma": ["asthma", "bronchial", "respiratory"],
            },
            "ja": {
                "糖尿病": ["糖尿病", "血糖", "インスリン", "血糖値"],
                "高血圧": ["高血圧", "血圧"],
                "心臓病": ["心臓病", "心疾患", "心血管", "冠動脈"],
                "がん": ["がん", "癌", "腫瘍", "悪性"],
                "喘息": ["喘息", "気管支"],
            }
        }
        
        self.response_count = 0
        logger.info("LLMService initialized with template-based response generation")

    def generate_response(
        self, 
        query: str, 
        retrieved_docs: List[Dict], 
        language: str = "en"
    ) -> Dict:
        """
        Generate structured medical response from retrieved documents.
        
        Args:
            query: User's query string
            retrieved_docs: List of retrieved document chunks with metadata
            language: Output language ('en' or 'ja')
            
        Returns:
            Dictionary containing:
                - response: Generated response text
                - sources: List of source documents used
                - language: Output language
                - generation_time_seconds: Time taken to generate
                - documents_used: Number of documents used
                
        Raises:
            ValueError: If query is empty
            
        Example:
            >>> response = service.generate_response(
            ...     "What are diabetes guidelines?",
            ...     documents,
            ...     "en"
            ... )
        """
        start_time = time.time()
        
        try:
            # Validate inputs
            if not query or not query.strip():
                raise ValueError("Query cannot be empty")
            
            if retrieved_docs is None:
                retrieved_docs = []
            
            # Validate and normalize language
            if language not in ["en", "ja"]:
                logger.warning(
                    f"Unsupported language '{language}', defaulting to 'en'",
                    extra={"requested_language": language}
                )
                language = "en"
            
            if not retrieved_docs:
                return self._generate_no_results_response(query, language)
            
            # Extract main topic from query
            topic = self._extract_topic(query, language)
            
            # Build structured response
            response_parts = []
            
            # 1. Introduction
            introduction = self._generate_introduction(topic, language)
            response_parts.append(introduction)
            
            # 2. Key findings from documents
            findings = self._extract_key_findings(retrieved_docs, language)
            if findings:
                findings_header = self.response_templates[language]["findings_header"]
                response_parts.append(f"\n{findings_header}\n{findings}")
            
            # 3. Recommendations
            recommendations = self._extract_recommendations(retrieved_docs, language)
            if recommendations:
                rec_header = self.response_templates[language]["recommendations_header"]
                response_parts.append(f"\n{rec_header}\n{recommendations}")
            
            # 4. Summary if no specific findings/recommendations
            if not findings and not recommendations:
                summary = self._generate_summary(retrieved_docs, language)
                if summary:
                    response_parts.append(f"\n{summary}")
            
            # 5. Medical disclaimer
            disclaimer = self.response_templates[language]["disclaimer"]
            response_parts.append(disclaimer)
            
            # Combine all parts
            response_text = "\n".join(response_parts)
            
            # Prepare sources information
            sources = self._prepare_sources(retrieved_docs)
            
            self.response_count += 1
            generation_time = time.time() - start_time
            
            logger.info(
                "Response generated successfully",
                extra={
                    "query_preview": query[:50],
                    "language": language,
                    "documents_used": len(retrieved_docs),
                    "generation_time": round(generation_time, 3),
                    "response_length": len(response_text)
                }
            )
            
            return {
                "response": response_text,
                "sources": sources,
                "language": language,
                "generation_time_seconds": round(generation_time, 3),
                "documents_used": len(retrieved_docs)
            }
            
        except ValueError as e:
            logger.warning(f"Validation error: {str(e)}")
            raise
        except Exception as e:
            logger.error(
                "Response generation failed",
                extra={
                    "query_preview": query[:50] if query else None,
                    "error": str(e)
                },
                exc_info=True
            )
            return self._generate_error_response(query, language)

    def _extract_topic(self, query: str, language: str) -> str:
        """
        Extract main medical topic from query.
        
        Args:
            query: User query
            language: Query language
            
        Returns:
            Extracted topic string
        """
        query_lower = query.lower()
        topics_dict = self.medical_topics.get(language, self.medical_topics["en"])
        
        # Check for topic matches with synonyms
        for topic, synonyms in topics_dict.items():
            if any(syn.lower() in query_lower for syn in synonyms):
                return topic
        
        # Fallback: return first few words or entire short query
        words = query.split()
        if len(words) <= 5:
            return query
        else:
            return " ".join(words[:3]) + "..."

    def _generate_introduction(self, topic: str, language: str) -> str:
        """Generate introduction section of response."""
        template = self.response_templates[language]["introduction"]
        return template.format(topic=topic)

    def _extract_key_findings(self, documents: List[Dict], language: str) -> str:
        """
        Extract key findings from retrieved documents.
        
        Args:
            documents: List of document chunks
            language: Output language
            
        Returns:
            Formatted findings text
        """
        findings = []
        
        for i, doc in enumerate(documents[:3], 1): 
            text = doc.get('text', '')
            doc_lang = doc.get('language', 'en')
            
            if not text:
                continue
            
            # Translate document text if language mismatch
            if doc_lang != language:
                try:
                    logger.info(f"Translating document text from {doc_lang} to {language}")
                    # Translate first 400 chars
                    text_to_translate = text[:400]
                    translated_text = translation_service.translate(text_to_translate, doc_lang, language, max_length=300)
                    
                    # Use translation even if quality is uncertain
                    if translated_text and len(translated_text) > 5:
                        text = translated_text
                        logger.info(f"Translation completed: {len(text)} chars")
                    else:
                        logger.warning(f"Translation returned empty/short result, using original")
                        
                except Exception as e:
                    logger.warning(f"Translation failed: {e}, using original text")
                    # Use original if translation completely fails
            
            # Extract first 1-2 meaningful sentences

            sentences = self._split_into_sentences(text, language)
            meaningful_sentences = []
            
            for sentence in sentences[:2]:  
                if len(sentence.strip()) > 10:  
                    meaningful_sentences.append(sentence.strip())
            
            if meaningful_sentences:
                summary = ('。' if language == 'ja' else '. ').join(meaningful_sentences)
                if not summary.endswith('.') and not summary.endswith('。'):
                    summary += '。' if language == 'ja' else '.'
                findings.append(f"{i}. {summary}")
        
        return "\n".join(findings) if findings else ""

    def _extract_recommendations(self, documents: List[Dict], language: str) -> str:
        """
        Extract recommendation sentences from documents.
        
        Args:
            documents: List of document chunks
            language: Output language
            
        Returns:
            Formatted recommendations text
        """
        recommendations = []
        seen_sentences: Set[str] = set()  
        keywords = self.medical_patterns["recommendation_keywords"][language]
        
        for doc in documents[:5]:  
            text = doc.get('text', '')
            doc_lang = doc.get('language', 'en')
            
            if not text:
                continue
            
            # Translate document text if language mismatch
            if doc_lang != language:
                try:
                    logger.info(f"Translating recommendation from {doc_lang} to {language}")
                    # Translate first 400 chars
                    text_to_translate = text[:400]
                    translated_text = translation_service.translate(text_to_translate, doc_lang, language, max_length=300)
                    
                    # Use translation 
                    if translated_text and len(translated_text) > 5:
                        text = translated_text
                        logger.info(f"Translation completed: {len(text)} chars")
                    else:
                        logger.warning(f"Translation returned empty/short, using original")
                        
                except Exception as e:
                    logger.warning(f"Translation failed: {e}, using original")
            
            # Use target language for sentence splitting
            sentences = self._split_into_sentences(text, language)
            
            for sentence in sentences[:3]:  
                sentence_normalized = ' '.join(sentence.split()) 
                
                # Skip if already seen (avoid duplicates)
                if sentence_normalized.lower() in seen_sentences:
                    continue
                
                # Include sentence if it has keywords OR is substantial (>20 chars)
                has_keyword = any(keyword.lower() in sentence.lower() for keyword in keywords)
                is_substantial = len(sentence.strip()) > 20
                
                if has_keyword or is_substantial:
                    clean_sentence = sentence.strip()
                    recommendations.append(f"• {clean_sentence}")
                    seen_sentences.add(sentence_normalized.lower())
                    if len(recommendations) >= 5:  # Limit to 5 recommendations
                        break
            
            if len(recommendations) >= 5:
                break
        
        return "\n".join(recommendations[:5])

    def _generate_summary(self, documents: List[Dict], language: str) -> str:
        """
        Generate summary when no specific findings/recommendations are found.
        
        Args:
            documents: List of document chunks
            language: Output language
            
        Returns:
            Summary text
        """
        # Extract and translate summary from documents
        translated_texts = []
        
        for doc in documents[:3]:
            text = doc.get('text', '')
            doc_lang = doc.get('language', 'en')
            
            if not text:
                continue
            
            # Translate if language mismatch
            if doc_lang != language:
                try:
                    logger.info(f"Translating summary text from {doc_lang} to {language}")
                    text_to_translate = text[:300]
                    translated = translation_service.translate(text_to_translate, doc_lang, language, max_length=200)
                    
                    if translated and len(translated) > 5:
                        text = translated
                        logger.info(f"Summary translation successful")
                    else:
                        logger.warning(f"Translation short, using original")
                        
                except Exception as e:
                    logger.warning(f"Summary translation failed: {e}, using original")
            
            translated_texts.append(text)
        
        if not translated_texts:
            # Fallback if no text
            if language == "ja":
                fallback_text = "個別の症例については医療専門家にご相談ください"
            else:
                fallback_text = "consultation with healthcare professionals is recommended for specific cases"
            template = self.response_templates[language]["summary_template"]
            return template.format(summary=fallback_text)
        
        # Combine texts and split into sentences
        all_text = " ".join(translated_texts)
        sentences = self._split_into_sentences(all_text, language)
        
        # Use first meaningful sentence as summary
        for sentence in sentences:
            if self._is_meaningful_sentence(sentence, language):
                template = self.response_templates[language]["summary_template"]
                return template.format(summary=sentence.strip())
        
        # Fallback
        if language == "ja":
            fallback_text = "個別の症例については医療専門家にご相談ください"
        else:
            fallback_text = "consultation with healthcare professionals is recommended for specific cases"
        
        template = self.response_templates[language]["summary_template"]
        return template.format(summary=fallback_text)

    def _prepare_sources(self, documents: List[Dict]) -> List[Dict]:
        """Prepare source information for response."""
        sources = []
        for doc in documents:
            sources.append({
                "document_id": doc.get('doc_id', 'unknown'),
                "chunk_id": doc.get('chunk_id', 0),
                "similarity_score": round(doc.get('similarity_score', 0.0), 3),
                "language": doc.get('language', 'en')
            })
        return sources

    def _split_into_sentences(self, text: str, language: str) -> List[str]:
        """
        Split text into sentences using NLTK for English, regex for Japanese.
        
        Args:
            text: Input text
            language: Text language
            
        Returns:
            List of sentences
        """
        try:
            if language == "ja":
                # For Japanese, use regex (NLTK doesn't handle Japanese well)
                sentences = re.split(r'(?<=[。！？])\s*', text)
            else:
                # Use NLTK for English (better handling of abbreviations)
                from nltk.tokenize import sent_tokenize
                sentences = sent_tokenize(text)
            
            # Clean and filter sentences
            clean_sentences = []
            for sentence in sentences:
                sentence = sentence.strip()
                if sentence and len(sentence) > 10:  # Minimum length
                    clean_sentences.append(sentence)
            
            return clean_sentences
            
        except Exception as e:
            logger.debug(f"Sentence tokenization failed, using fallback: {e}")
            # Fallback to simple regex
            if language == "ja":
                sentences = re.split(r'[。！？]', text)
            else:
                sentences = re.split(r'[.!?]', text)
            return [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]

    def _is_meaningful_sentence(self, sentence: str, language: str) -> bool:
        """
        Check if sentence contains meaningful medical content.
        
        Args:
            sentence: Sentence to check
            language: Sentence language
            
        Returns:
            True if sentence is meaningful
        """
        if len(sentence) < 20:  # Too short
            return False
        
        # Check for medical keywords
        all_keywords = []
        for pattern in self.medical_patterns.values():
            all_keywords.extend(pattern.get(language, []))
        
        sentence_lower = sentence.lower()
        has_keyword = any(keyword in sentence_lower for keyword in all_keywords)
        
        # Also accept sentences with numbers (likely contain data/measurements)
        has_number = bool(re.search(r'\d', sentence))
        
        return has_keyword or has_number

    def _generate_no_results_response(self, query: str, language: str) -> Dict:
        """Generate response when no documents are retrieved."""
        topic = self._extract_topic(query, language)
        template = self.response_templates[language]["no_results"]
        response_text = template.format(topic=topic)
        
        logger.info(
            "Generated no-results response",
            extra={"query_preview": query[:50], "language": language}
        )
        
        return {
            "response": response_text,
            "sources": [],
            "language": language,
            "generation_time_seconds": 0.0,
            "documents_used": 0
        }

    def _generate_error_response(self, query: str, language: str) -> Dict:
        """Generate error response when generation fails."""
        response_text = self.response_templates[language]["error"]
        
        return {
            "response": response_text,
            "sources": [],
            "language": language,
            "generation_time_seconds": 0.0,
            "documents_used": 0
        }

    def get_service_info(self) -> Dict:
        """Get information about the LLM service."""
        return {
            "service_type": "template_based",
            "response_count": self.response_count,
            "supported_languages": ["en", "ja"],
            "templates_available": list(self.response_templates.keys()),
            "medical_topics_count": sum(len(topics) for topics in self.medical_topics.values())
        }

    def health_check(self) -> Dict:
        """Perform health check on LLM service."""
        try:
            # Test response generation
            test_query = "diabetes management"
            test_docs = [{
                "text": "Diabetes management involves regular monitoring of blood glucose levels and adherence to prescribed medications.",
                "doc_id": "test_doc",
                "chunk_id": 0,
                "similarity_score": 0.9,
                "language": "en"
            }]
            
            response = self.generate_response(test_query, test_docs, "en")
            
            return {
                "status": "healthy",
                "response_generation_works": True,
                "response_count": self.response_count,
                "service_type": "template_based",
                "test_response_length": len(response["response"])
            }
            
        except Exception as e:
            logger.error(f"Health check failed: {e}", exc_info=True)
            return {
                "status": "unhealthy",
                "error": str(e),
                "response_count": self.response_count
            }


# Global instance for dependency injection
llm_service = LLMService()


# For testing purposes
if __name__ == "__main__":
    print("Testing LLM Service...\n")
    
    service = LLMService()
    
    # Test data
    test_documents = [
        {
            "text": "Type 2 diabetes management requires comprehensive lifestyle modifications including dietary changes and physical activity. Regular monitoring of HbA1c levels is essential for glycemic control.",
            "doc_id": "doc_1",
            "chunk_id": 0,
            "similarity_score": 0.95,
            "language": "en"
        },
        {
            "text": "First-line pharmacological therapy for type 2 diabetes typically includes metformin. Patients should be advised on proper medication adherence and potential side effects.",
            "doc_id": "doc_1", 
            "chunk_id": 1,
            "similarity_score": 0.88,
            "language": "en"
        }
    ]
    
    # Test English response
    print("=" * 60)
    print("TEST 1: English Response")
    print("=" * 60)
    response_en = service.generate_response(
        "What are the guidelines for diabetes management?",
        test_documents,
        "en"
    )
    print(response_en["response"])
    print(f"\nSources: {response_en['sources']}")
    print(f"Generation time: {response_en['generation_time_seconds']:.3f}s\n")
    
    # Test Japanese response
    print("=" * 60)
    print("TEST 2: Japanese Response")
    print("=" * 60)
    response_ja = service.generate_response(
        "糖尿病管理のガイドラインは何ですか？",
        test_documents, 
        "ja"
    )
    print(response_ja["response"])
    print(f"\nSources: {response_ja['sources']}")
    print(f"Generation time: {response_ja['generation_time_seconds']:.3f}s\n")
    
    # Test no results
    print("=" * 60)
    print("TEST 3: No Results Response")
    print("=" * 60)
    response_no_results = service.generate_response(
        "What about quantum physics?",
        [],
        "en"
    )
    print(response_no_results["response"])
    print()
    
    # Test service info
    print("=" * 60)
    print("TEST 4: Service Info")
    print("=" * 60)
    info = service.get_service_info()
    print(f"Service info: {info}\n")
    
    # Test health check
    print("=" * 60)
    print("TEST 5: Health Check")
    print("=" * 60)
    health = service.health_check()
    print(f"Health check: {health}\n")
    
    print("All tests completed!")