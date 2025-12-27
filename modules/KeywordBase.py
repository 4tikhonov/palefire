"""
Pale Fire - Keyword Extraction Module using Gensim

This module provides keyword extraction capabilities using various Gensim algorithms
with configurable weights and parameters.
"""

import logging
import re
from typing import Dict, List, Any, Optional, Tuple
from collections import Counter

# Configure logger
logger = logging.getLogger(__name__)

# Try to import gensim
try:
    import gensim
    from gensim import corpora, models
    from gensim.utils import simple_preprocess
    GENSIM_AVAILABLE = True
    
    # Try to import preprocessing functions (may not be available in newer gensim versions)
    try:
        from gensim.parsing.preprocessing import STOPWORDS, preprocess_string, strip_tags, strip_punctuation, strip_numeric, remove_stopwords, strip_short, stem_text
        GENSIM_PREPROCESSING_AVAILABLE = True
    except ImportError:
        # Fallback: use basic stopwords and implement our own preprocessing
        GENSIM_PREPROCESSING_AVAILABLE = False
        STOPWORDS = set()
        logger.info("Using custom preprocessing (gensim.parsing.preprocessing not available)")
        
except ImportError:
    logger.warning("Gensim not installed. Install with: pip install gensim")
    GENSIM_AVAILABLE = False
    GENSIM_PREPROCESSING_AVAILABLE = False
    STOPWORDS = set()


# Custom preprocessing functions (fallback when gensim.parsing.preprocessing is not available)
def _strip_tags(text: str) -> str:
    """Remove HTML/XML tags."""
    import re
    return re.sub(r'<[^>]+>', '', text)


def _strip_punctuation(text: str) -> str:
    """Remove punctuation."""
    import string
    return text.translate(str.maketrans('', '', string.punctuation))


def _strip_numeric(text: str) -> str:
    """Remove numeric characters."""
    import re
    return re.sub(r'\d+', '', text)


def _remove_stopwords(text: str, stopwords: set) -> str:
    """Remove stopwords."""
    words = text.split()
    return ' '.join(word for word in words if word.lower() not in stopwords)


def _strip_short(text: str, minsize: int = 3) -> str:
    """Remove words shorter than minsize."""
    words = text.split()
    return ' '.join(word for word in words if len(word) >= minsize)


def _stem_text(text: str) -> str:
    """Stem text using Porter stemmer."""
    try:
        from nltk.stem import PorterStemmer
        stemmer = PorterStemmer()
        words = text.split()
        return ' '.join(stemmer.stem(word) for word in words)
    except ImportError:
        logger.warning("NLTK not available for stemming, skipping")
        return text


# Default English stopwords (basic set)
DEFAULT_STOPWORDS = {
    'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
    'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
    'to', 'was', 'will', 'with', 'the', 'this', 'but', 'they', 'have',
    'had', 'what', 'said', 'each', 'which', 'their', 'time', 'if',
    'up', 'out', 'many', 'then', 'them', 'these', 'so', 'some', 'her',
    'would', 'make', 'like', 'into', 'him', 'has', 'two', 'more',
    'very', 'after', 'words', 'long', 'than', 'first', 'been', 'call',
    'who', 'oil', 'sit', 'now', 'find', 'down', 'day', 'did', 'get',
    'come', 'made', 'may', 'part'
}


class KeywordExtractor:
    """
    Extract important keywords from text using Gensim algorithms.
    Supports multiple extraction methods with configurable weights.
    """
    
    def __init__(
        self,
        method: str = 'tfidf',
        num_keywords: int = 10,
        min_word_length: int = 3,
        max_word_length: int = 50,
        use_stemming: bool = False,
        custom_stopwords: Optional[List[str]] = None,
        tfidf_weight: float = 1.0,
        textrank_weight: float = 0.5,
        word_freq_weight: float = 0.3,
        position_weight: float = 0.2,
        title_weight: float = 2.0,
        first_sentence_weight: float = 1.5,
        enable_ngrams: bool = True,
        min_ngram: int = 2,
        max_ngram: int = 4,
        ngram_weight: float = 1.2,
    ):
        """
        Initialize the Keyword Extractor.
        
        Args:
            method: Extraction method ('tfidf', 'textrank', 'combined', 'word_freq')
            num_keywords: Number of keywords to extract
            min_word_length: Minimum word length to consider
            max_word_length: Maximum word length to consider
            use_stemming: Whether to use stemming
            custom_stopwords: Additional stopwords to filter
            tfidf_weight: Weight for TF-IDF scores (for combined method)
            textrank_weight: Weight for TextRank scores (for combined method)
            word_freq_weight: Weight for word frequency scores (for combined method)
            position_weight: Weight for position-based scoring
            title_weight: Weight multiplier for words in titles/headers
            first_sentence_weight: Weight multiplier for words in first sentence
            enable_ngrams: Whether to extract n-grams (2-4 word phrases)
            min_ngram: Minimum n-gram size (2, 3, or 4)
            max_ngram: Maximum n-gram size (2, 3, or 4)
            ngram_weight: Weight multiplier for n-grams (default: 1.2 to slightly favor phrases)
        """
        if not GENSIM_AVAILABLE:
            raise ImportError("Gensim is required. Install with: pip install gensim")
        
        self.method = method.lower()
        self.num_keywords = num_keywords
        self.min_word_length = min_word_length
        self.max_word_length = max_word_length
        self.use_stemming = use_stemming
        
        # Combine default stopwords with custom ones
        if GENSIM_PREPROCESSING_AVAILABLE and STOPWORDS:
            self.stopwords = set(STOPWORDS)
        else:
            self.stopwords = DEFAULT_STOPWORDS.copy()
        
        if custom_stopwords:
            self.stopwords.update(word.lower() for word in custom_stopwords)
        
        # Weights for combined scoring
        self.tfidf_weight = tfidf_weight
        self.textrank_weight = textrank_weight
        self.word_freq_weight = word_freq_weight
        self.position_weight = position_weight
        self.title_weight = title_weight
        self.first_sentence_weight = first_sentence_weight
        
        # Store whether to use gensim preprocessing or custom
        self.use_gensim_preprocessing = GENSIM_PREPROCESSING_AVAILABLE
        
        # N-gram settings
        self.enable_ngrams = enable_ngrams
        # Allow min_ngram=1 (means include unigrams), otherwise clamp between 2 and 4
        if min_ngram == 1:
            self.min_ngram = 1
        else:
            self.min_ngram = max(2, min(min_ngram, 4))  # Clamp between 2 and 4
        self.max_ngram = max(2, min(max_ngram, 4))  # Clamp between 2 and 4
        if self.min_ngram > self.max_ngram and self.min_ngram != 1:
            self.min_ngram, self.max_ngram = self.max_ngram, self.min_ngram
        self.ngram_weight = ngram_weight
    
    def preprocess_text(self, text: str) -> List[str]:
        """
        Preprocess text into tokens.
        
        Args:
            text: Input text
            
        Returns:
            List of preprocessed tokens
        """
        if self.use_gensim_preprocessing:
            # Use gensim preprocessing if available
            try:
                # Import here to ensure they're available
                from gensim.parsing.preprocessing import (
                    preprocess_string, strip_tags, strip_punctuation, 
                    strip_numeric, remove_stopwords, strip_short, stem_text
                )
                filters = [
                    lambda x: x.lower(),
                    strip_tags,
                    strip_punctuation,
                    strip_numeric,
                    remove_stopwords,
                    strip_short,
                ]
                if self.use_stemming:
                    filters.append(stem_text)
                processed = preprocess_string(text, filters)
            except (ImportError, NameError, Exception) as e:
                logger.warning(f"Gensim preprocessing failed: {e}, using custom preprocessing")
                processed = self._custom_preprocess(text)
        else:
            # Use custom preprocessing
            processed = self._custom_preprocess(text)
        
        # Filter by length
        filtered = [
            word for word in processed 
            if self.min_word_length <= len(word) <= self.max_word_length
        ]
        
        return filtered
    
    def _custom_preprocess(self, text: str) -> List[str]:
        """Custom preprocessing when gensim.parsing.preprocessing is not available."""
        # Convert to lowercase
        text = text.lower()
        
        # Remove HTML/XML tags
        text = _strip_tags(text)
        
        # Remove punctuation
        text = _strip_punctuation(text)
        
        # Remove numeric characters
        text = _strip_numeric(text)
        
        # Remove stopwords
        text = _remove_stopwords(text, self.stopwords)
        
        # Remove short words
        text = _strip_short(text, self.min_word_length)
        
        # Apply stemming if requested
        if self.use_stemming:
            text = _stem_text(text)
        
        # Split into words and filter
        words = text.split()
        
        return words
    
    def generate_ngrams(self, tokens: List[str], n: int, text: Optional[str] = None) -> List[str]:
        """
        Generate n-grams from a list of tokens.
        Filters out low-quality n-grams (with stopwords in middle, across sentence boundaries).
        
        Args:
            tokens: List of preprocessed tokens
            n: Size of n-gram (2, 3, or 4)
            text: Original text (optional, for better filtering)
            
        Returns:
            List of n-gram strings
        """
        if len(tokens) < n:
            return []
        
        ngrams = []
        for i in range(len(tokens) - n + 1):
            ngram_tokens = tokens[i:i+n]
            ngram = ' '.join(ngram_tokens)
            
            # Filter out n-grams that are too short or too long
            if len(ngram) < self.min_word_length * n or len(ngram) > self.max_word_length * n:
                continue
            
            # Filter out n-grams with stopwords in the middle (but allow at edges)
            # This improves quality by avoiding phrases like "is a common" or "the baseline"
            if n > 2:
                middle_tokens = ngram_tokens[1:-1]
                if any(token.lower() in self.stopwords for token in middle_tokens):
                    continue
            
            # For 2-grams, be more selective - avoid stopword-starting unless it's meaningful
            if n == 2:
                # Skip if both tokens are stopwords
                if all(token.lower() in self.stopwords for token in ngram_tokens):
                    continue
                # Skip if first token is a very common stopword (unless second is important)
                common_stopwords = {'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'can', 'should', 'will', 'would'}
                if ngram_tokens[0].lower() in common_stopwords:
                    # Only keep if second word is substantial (longer than 4 chars)
                    if len(ngram_tokens[1]) <= 4:
                        continue
                # Skip if second token is a very common stopword
                if ngram_tokens[1].lower() in common_stopwords:
                    continue
                # Skip if either token is very short (likely not meaningful)
                if len(ngram_tokens[0]) < 3 or len(ngram_tokens[1]) < 3:
                    continue
            
            ngrams.append(ngram)
        
        return ngrams
    
    def extract_ngrams_tfidf(self, text: str, documents: Optional[List[str]] = None) -> Dict[int, List[Tuple[str, float]]]:
        """
        Extract n-grams using TF-IDF for each n-gram size.
        
        Args:
            text: Input text
            documents: Optional list of documents for IDF calculation
            
        Returns:
            Dictionary mapping n-gram size to list of (ngram, score) tuples
        """
        if not self.enable_ngrams:
            return {}
        
        tokens = self.preprocess_text(text)
        # Skip if min_ngram is 1 (unigrams handled separately) or not enough tokens
        start_n = max(2, self.min_ngram) if self.min_ngram > 1 else 2
        if len(tokens) < start_n:
            return {}
        
        ngram_results = {}
        
        for n in range(start_n, self.max_ngram + 1):
            # Generate n-grams
            ngrams = self.generate_ngrams(tokens, n, text)
            if not ngrams:
                continue
            
            # Create corpus with n-grams
            if documents:
                doc_tokens = [self.preprocess_text(doc) for doc in documents]
                doc_ngrams = []
                for doc_token_list in doc_tokens:
                    doc_ngrams.append(self.generate_ngrams(doc_token_list, n))
                doc_ngrams.append(ngrams)  # Add current text n-grams
            else:
                doc_ngrams = [ngrams]
            
            # Create dictionary and corpus
            dictionary = corpora.Dictionary(doc_ngrams)
            corpus = [dictionary.doc2bow(doc) for doc in doc_ngrams]
            
            # Calculate TF-IDF
            tfidf = models.TfidfModel(corpus)
            
            # Get TF-IDF scores for the last document (our text)
            text_bow = dictionary.doc2bow(ngrams)
            text_tfidf = tfidf[text_bow]
            
            # Convert to word-score pairs
            id2word = dictionary.id2token
            keywords = [
                (id2word[word_id], score) 
                for word_id, score in text_tfidf
            ]
            
            # Sort by score (descending)
            keywords.sort(key=lambda x: x[1], reverse=True)
            ngram_results[n] = keywords
        
        return ngram_results
    
    def extract_ngrams_freq(self, text: str) -> Dict[int, List[Tuple[str, float]]]:
        """
        Extract n-grams using word frequency for each n-gram size.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary mapping n-gram size to list of (ngram, score) tuples
        """
        if not self.enable_ngrams:
            return {}
        
        tokens = self.preprocess_text(text)
        # Skip if min_ngram is 1 (unigrams handled separately) or not enough tokens
        start_n = max(2, self.min_ngram) if self.min_ngram > 1 else 2
        if len(tokens) < start_n:
            return {}
        
        ngram_results = {}
        
        for n in range(start_n, self.max_ngram + 1):
            # Generate n-grams
            ngrams = self.generate_ngrams(tokens, n, text)
            if not ngrams:
                continue
            
            # Count n-gram frequencies
            ngram_counts = Counter(ngrams)
            
            # Convert to (ngram, frequency) tuples
            keywords = [(ngram, float(count)) for ngram, count in ngram_counts.items()]
            
            # Sort by frequency (descending)
            keywords.sort(key=lambda x: x[1], reverse=True)
            ngram_results[n] = keywords
        
        return ngram_results
    
    def extract_tfidf_keywords(self, text: str, documents: Optional[List[str]] = None) -> List[Tuple[str, float]]:
        """
        Extract keywords using TF-IDF.
        
        Args:
            text: Input text
            documents: Optional list of documents for IDF calculation
            
        Returns:
            List of (keyword, score) tuples sorted by score
        """
        # Preprocess text
        tokens = self.preprocess_text(text)
        
        if not tokens:
            return []
        
        # Create corpus
        if documents:
            # Use provided documents for IDF calculation
            doc_tokens = [self.preprocess_text(doc) for doc in documents]
            doc_tokens.append(tokens)  # Add current text
        else:
            # Use single document (TF only, no IDF)
            doc_tokens = [tokens]
        
        # Create dictionary and corpus
        dictionary = corpora.Dictionary(doc_tokens)
        corpus = [dictionary.doc2bow(doc) for doc in doc_tokens]
        
        # Calculate TF-IDF
        tfidf = models.TfidfModel(corpus)
        
        # Get TF-IDF scores for the last document (our text)
        text_bow = dictionary.doc2bow(tokens)
        text_tfidf = tfidf[text_bow]
        
        # Convert to word-score pairs
        id2word = dictionary.id2token
        keywords = [
            (id2word[word_id], score) 
            for word_id, score in text_tfidf
        ]
        
        # Sort by score (descending)
        keywords.sort(key=lambda x: x[1], reverse=True)
        
        return keywords
    
    def extract_textrank_keywords(self, text: str) -> List[Tuple[str, float]]:
        """
        Extract keywords using TextRank algorithm.
        
        Args:
            text: Input text
            
        Returns:
            List of (keyword, score) tuples sorted by score
        """
        try:
            # TextRank summarization (which also extracts keywords)
            # Note: gensim.summarization.keywords is deprecated, but we'll use it
            # For newer gensim versions, we'll implement a simple version
            
            # Split text into sentences
            sentences = re.split(r'[.!?]+', text)
            sentences = [s.strip() for s in sentences if s.strip()]
            
            if len(sentences) < 2:
                # Fallback to word frequency if too few sentences
                return self.extract_word_freq_keywords(text)
            
            # Preprocess each sentence
            sentence_tokens = [self.preprocess_text(sent) for sent in sentences]
            
            # Build word co-occurrence graph
            word_scores = {}
            word_counts = Counter()
            
            for tokens in sentence_tokens:
                word_counts.update(tokens)
                # Build co-occurrence within sentence
                unique_tokens = list(set(tokens))
                for i, word1 in enumerate(unique_tokens):
                    if word1 not in word_scores:
                        word_scores[word1] = 0.0
                    for word2 in unique_tokens[i+1:]:
                        if word2 not in word_scores:
                            word_scores[word2] = 0.0
                        # Add co-occurrence score
                        word_scores[word1] += 1.0
                        word_scores[word2] += 1.0
            
            # Normalize by word frequency
            keywords = []
            for word, score in word_scores.items():
                if word in word_counts:
                    normalized_score = score / (word_counts[word] + 1)
                    keywords.append((word, normalized_score))
            
            # Sort by score
            keywords.sort(key=lambda x: x[1], reverse=True)
            
            return keywords
            
        except Exception as e:
            logger.warning(f"TextRank extraction failed: {e}, falling back to word frequency")
            return self.extract_word_freq_keywords(text)
    
    def extract_word_freq_keywords(self, text: str) -> List[Tuple[str, float]]:
        """
        Extract keywords using simple word frequency.
        
        Args:
            text: Input text
            
        Returns:
            List of (keyword, score) tuples sorted by score
        """
        tokens = self.preprocess_text(text)
        
        if not tokens:
            return []
        
        # Count word frequencies
        word_counts = Counter(tokens)
        
        # Convert to (word, frequency) tuples
        keywords = [(word, float(count)) for word, count in word_counts.items()]
        
        # Sort by frequency (descending)
        keywords.sort(key=lambda x: x[1], reverse=True)
        
        return keywords
    
    def calculate_position_scores(self, text: str, keywords: List[Tuple[str, float]]) -> Dict[str, float]:
        """
        Calculate position-based scores for keywords.
        
        Args:
            text: Original text
            keywords: List of (keyword, score) tuples
            
        Returns:
            Dictionary mapping keywords to position scores
        """
        position_scores = {}
        
        # Split text into sentences
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            return position_scores
        
        # First sentence gets higher weight
        first_sentence = sentences[0].lower()
        
        # Check for title/header patterns (all caps, short lines, etc.)
        lines = text.split('\n')
        title_lines = []
        for line in lines[:3]:  # Check first 3 lines
            line_stripped = line.strip()
            if line_stripped and len(line_stripped) < 100:
                # Check if it's mostly uppercase or has special formatting
                if line_stripped.isupper() or line_stripped.startswith('#'):
                    title_lines.append(line_stripped.lower())
        
        # Calculate position scores
        for keyword, _ in keywords:
            keyword_lower = keyword.lower()
            score = 1.0  # Base score
            
            # Check first sentence
            if keyword_lower in first_sentence:
                score *= self.first_sentence_weight
            
            # Check title lines
            for title_line in title_lines:
                if keyword_lower in title_line:
                    score *= self.title_weight
                    break
            
            position_scores[keyword] = score
        
        return position_scores
    
    def extract_combined_keywords(
        self, 
        text: str, 
        documents: Optional[List[str]] = None
    ) -> List[Tuple[str, float]]:
        """
        Extract keywords using combined scoring from multiple methods.
        
        Args:
            text: Input text
            documents: Optional list of documents for IDF calculation
            
        Returns:
            List of (keyword, score) tuples sorted by score
        """
        # Get keywords from different methods
        tfidf_keywords = dict(self.extract_tfidf_keywords(text, documents))
        textrank_keywords = dict(self.extract_textrank_keywords(text))
        freq_keywords = dict(self.extract_word_freq_keywords(text))
        
        # Get position scores
        all_keywords = set(tfidf_keywords.keys()) | set(textrank_keywords.keys()) | set(freq_keywords.keys())
        position_scores = self.calculate_position_scores(text, [(kw, 0) for kw in all_keywords])
        
        # Combine scores
        combined_scores = {}
        for keyword in all_keywords:
            score = 0.0
            
            # TF-IDF score
            if keyword in tfidf_keywords:
                score += self.tfidf_weight * tfidf_keywords[keyword]
            
            # TextRank score
            if keyword in textrank_keywords:
                score += self.textrank_weight * textrank_keywords[keyword]
            
            # Word frequency score (normalized)
            if keyword in freq_keywords:
                max_freq = max(freq_keywords.values()) if freq_keywords else 1.0
                normalized_freq = freq_keywords[keyword] / max_freq
                score += self.word_freq_weight * normalized_freq
            
            # Position score
            if keyword in position_scores:
                score *= (1.0 + self.position_weight * (position_scores[keyword] - 1.0))
            
            combined_scores[keyword] = score
        
        # Convert to sorted list
        keywords = [(kw, score) for kw, score in combined_scores.items() if score > 0]
        keywords.sort(key=lambda x: x[1], reverse=True)
        
        return keywords
    
    def extract(self, text: str, documents: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Extract keywords from text using the configured method.
        Includes both unigrams and n-grams if enabled.
        
        Args:
            text: Input text
            documents: Optional list of documents for IDF calculation (for TF-IDF)
            
        Returns:
            List of keyword dictionaries with 'keyword', 'score', and 'type' fields
        """
        if not text or not text.strip():
            return []
        
        # Always extract unigrams (min_ngram only controls n-gram range, not unigram extraction)
        unigrams = []
        if self.method == 'tfidf':
            unigrams = self.extract_tfidf_keywords(text, documents)
            # If TF-IDF returns empty (single document case), fallback to word frequency
            if not unigrams and not documents:
                logger.debug("TF-IDF returned no results (single document), falling back to word frequency")
                unigrams = self.extract_word_freq_keywords(text)
        elif self.method == 'textrank':
            unigrams = self.extract_textrank_keywords(text)
        elif self.method == 'word_freq':
            unigrams = self.extract_word_freq_keywords(text)
        elif self.method == 'combined':
            unigrams = self.extract_combined_keywords(text, documents)
            # If combined returns empty, fallback to word frequency
            if not unigrams and not documents:
                logger.debug("Combined method returned no results, falling back to word frequency")
                unigrams = self.extract_word_freq_keywords(text)
        else:
            logger.warning(f"Unknown method '{self.method}', using 'word_freq'")
            unigrams = self.extract_word_freq_keywords(text)
        
        # Extract n-grams if enabled
        ngrams_all = []
        if self.enable_ngrams:
            # Determine n-gram range
            start_n = max(2, self.min_ngram) if self.min_ngram > 1 else 2
            
            if self.method == 'tfidf':
                ngram_dict = self.extract_ngrams_tfidf(text, documents)
                # If TF-IDF returns empty or all n-gram lists are empty, fallback to frequency
                if not ngram_dict or (not documents and all(len(ngram_list) == 0 for ngram_list in ngram_dict.values())):
                    logger.debug("TF-IDF n-grams returned no results (single document), falling back to frequency")
                    ngram_dict = self.extract_ngrams_freq(text)
            elif self.method == 'word_freq':
                ngram_dict = self.extract_ngrams_freq(text)
            else:
                # For textrank and combined, use frequency-based n-grams
                ngram_dict = self.extract_ngrams_freq(text)
            
            # Combine n-grams from all sizes
            for n, ngram_list in ngram_dict.items():
                # Apply n-gram weight multiplier
                weighted_ngrams = [
                    (ngram, score * self.ngram_weight)
                    for ngram, score in ngram_list
                ]
                ngrams_all.extend(weighted_ngrams)
        
        # Combine unigrams and n-grams
        all_keywords = []
        
        # Add unigrams with type='unigram'
        for keyword, score in unigrams:
            all_keywords.append({
                'keyword': keyword,
                'score': float(score),
                'type': 'unigram'
            })
        
        # Add n-grams with type='ngram' and n-gram size
        for i, (ngram, score) in enumerate(ngrams_all):
            # Determine n-gram size
            ngram_size = len(ngram.split())
            all_keywords.append({
                'keyword': ngram,
                'score': float(score),
                'type': f'{ngram_size}-gram'
            })
        
        # Sort by score (descending)
        all_keywords.sort(key=lambda x: x['score'], reverse=True)
        
        # Limit to requested number
        all_keywords = all_keywords[:self.num_keywords]
        
        return all_keywords
    
    def extract_keywords_list(self, text: str, documents: Optional[List[str]] = None) -> List[str]:
        """
        Extract keywords as a simple list of strings.
        
        Args:
            text: Input text
            documents: Optional list of documents for IDF calculation
            
        Returns:
            List of keyword strings
        """
        keywords = self.extract(text, documents)
        return [kw['keyword'] for kw in keywords]

