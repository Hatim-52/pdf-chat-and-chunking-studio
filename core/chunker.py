import pymupdf
import re
import io
import numpy as np

class RecursiveCharacterTextSplitter:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200, separators: list = None):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", " ", ""]

    def _split_text(self, text: str, separators: list) -> list[str]:
        if len(text) <= self.chunk_size:
            return [text]
        
        if not separators:
            # Fallback to character splitting
            return [text[i:i + self.chunk_size] for i in range(0, len(text), self.chunk_size - self.chunk_overlap)]
        
        separator = separators[0]
        next_separators = separators[1:]
        
        if separator == "":
            splits = list(text)
        else:
            splits = text.split(separator)
            
        chunks = []
        current_chunk = []
        current_length = 0
        
        for split in splits:
            if len(split) > self.chunk_size:
                if current_chunk:
                    chunks.append(separator.join(current_chunk))
                    current_chunk = []
                    current_length = 0
                
                # Recursively split the oversized part
                recursive_splits = self._split_text(split, next_separators)
                chunks.extend(recursive_splits)
            else:
                added_len = len(split) + (len(separator) if current_length > 0 else 0)
                if current_length + added_len <= self.chunk_size:
                    current_chunk.append(split)
                    current_length += added_len
                else:
                    if current_chunk:
                        chunks.append(separator.join(current_chunk))
                    
                    # Backtrack to implement overlap
                    overlap_chunk = []
                    overlap_len = 0
                    for prev_split in reversed(current_chunk):
                        prev_added_len = len(prev_split) + (len(separator) if overlap_len > 0 else 0)
                        if overlap_len + prev_added_len <= self.chunk_overlap:
                            overlap_chunk.insert(0, prev_split)
                            overlap_len += prev_added_len
                        else:
                            break
                    
                    current_chunk = overlap_chunk + [split]
                    current_length = sum(len(s) for s in current_chunk) + len(separator) * (len(current_chunk) - 1)
                    
        if current_chunk:
            chunks.append(separator.join(current_chunk))
            
        return chunks

    def split_text(self, text: str) -> list[str]:
        return self._split_text(text, self.separators)


class SemanticTextSplitter:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200, threshold_percentile: float = 80.0):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.threshold_percentile = threshold_percentile

    def split_text(self, text: str) -> list[str]:
        # Split by typical sentence boundaries
        sentence_regex = re.compile(r'(?<=[.!?])\s+')
        sentences = [s.strip() for s in sentence_regex.split(text) if s.strip()]
        if len(sentences) <= 1:
            return sentences

        # Tokenize sentences and build vocabulary
        tokenized_sentences = []
        vocab = {}
        vocab_idx = 0
        for sent in sentences:
            tokens = re.findall(r'\b\w+\b', sent.lower())
            tokenized_sentences.append(tokens)
            for token in tokens:
                if token not in vocab:
                    vocab[token] = vocab_idx
                    vocab_idx += 1

        if not vocab:
            return sentences

        # Build sentence vectors (bag of words)
        vectors = np.zeros((len(sentences), len(vocab)))
        for i, tokens in enumerate(tokenized_sentences):
            for token in tokens:
                vectors[i, vocab[token]] += 1
            
            # Normalize vectors (L2)
            norm = np.linalg.norm(vectors[i])
            if norm > 0:
                vectors[i] = vectors[i] / norm

        # Compute cosine similarities between consecutive sentences
        similarities = []
        for i in range(len(sentences) - 1):
            sim = np.dot(vectors[i], vectors[i+1])
            similarities.append(sim)

        # Distances
        distances = [1.0 - sim for sim in similarities]

        # Determine threshold distance using percentile
        if distances:
            threshold = np.percentile(distances, self.threshold_percentile)
        else:
            threshold = 0.5

        # Group sentences into chunks based on threshold boundaries
        chunks = []
        current_chunk = [sentences[0]]
        current_len = len(sentences[0])

        for i in range(len(sentences) - 1):
            next_sent = sentences[i+1]
            dist = distances[i]

            # If boundary is detected or adding next sentence exceeds maximum chunk size
            if dist > threshold or (current_len + len(next_sent) + 1 > self.chunk_size):
                chunks.append(" ".join(current_chunk))
                current_chunk = [next_sent]
                current_len = len(next_sent)
            else:
                current_chunk.append(next_sent)
                current_len += len(next_sent) + 1

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        # Perform a pass to split any chunk that is still larger than chunk_size
        final_chunks = []
        for chunk in chunks:
            if len(chunk) > self.chunk_size:
                fallback_splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
                final_chunks.extend(fallback_splitter.split_text(chunk))
            else:
                final_chunks.append(chunk)

        return final_chunks


def extract_pages_from_pdf(pdf_file) -> list[dict]:
    """
    Extracts text from each page of the uploaded PDF file.
    Returns a list of dicts: [{'page_num': int, 'text': str}]
    """
    pages = []
    # If it is bytes, read it directly
    if isinstance(pdf_file, bytes):
        doc = pymupdf.open(stream=pdf_file, filetype="pdf")
    else:
        # Streamlit UploadedFile is file-like
        pdf_bytes = pdf_file.read()
        doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
        
    for i, page in enumerate(doc):
        text = page.get_text("text")
        pages.append({
            "page_num": i + 1,
            "text": text
        })
    return pages


def chunk_document(pages: list[dict], strategy: str, chunk_size: int, chunk_overlap: int, threshold_percentile: float = 80.0) -> list[dict]:
    """
    Chunks a list of pages based on the selected strategy, chunk_size, and chunk_overlap.
    Returns a list of chunk dicts:
    [{'id': int, 'text': str, 'page_num': int, 'char_count': int, 'word_count': int}]
    """
    chunks = []
    chunk_id = 1
    
    if strategy == "Recursive Character":
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        for page in pages:
            page_text = page["text"].strip()
            if not page_text:
                continue
            split_texts = splitter.split_text(page_text)
            for text in split_texts:
                cleaned_text = text.strip()
                if cleaned_text:
                    chunks.append({
                        "id": chunk_id,
                        "text": cleaned_text,
                        "page_num": page["page_num"],
                        "char_count": len(cleaned_text),
                        "word_count": len(cleaned_text.split())
                    })
                    chunk_id += 1
                    
    elif strategy == "Semantic":
        splitter = SemanticTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, threshold_percentile=threshold_percentile)
        for page in pages:
            page_text = page["text"].strip()
            if not page_text:
                continue
            split_texts = splitter.split_text(page_text)
            for text in split_texts:
                cleaned_text = text.strip()
                if cleaned_text:
                    chunks.append({
                        "id": chunk_id,
                        "text": cleaned_text,
                        "page_num": page["page_num"],
                        "char_count": len(cleaned_text),
                        "word_count": len(cleaned_text.split())
                    })
                    chunk_id += 1
                    
    elif strategy == "Sentence-based":
        # Split by typical sentence boundaries
        sentence_regex = re.compile(r'(?<=[.!?])\s+')
        for page in pages:
            page_text = page["text"].strip()
            if not page_text:
                continue
            sentences = sentence_regex.split(page_text)
            
            current_chunk = []
            current_len = 0
            
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                
                # If adding this sentence exceeds chunk size, save current and restart with overlap
                if current_len + len(sentence) > chunk_size and current_chunk:
                    chunk_text = " ".join(current_chunk)
                    chunks.append({
                        "id": chunk_id,
                        "text": chunk_text,
                        "page_num": page["page_num"],
                        "char_count": len(chunk_text),
                        "word_count": len(chunk_text.split())
                    })
                    chunk_id += 1
                    
                    # Implement overlap by keeping last N sentences that fit within overlap limit
                    overlap_chunk = []
                    overlap_len = 0
                    for s in reversed(current_chunk):
                        if overlap_len + len(s) + 1 <= chunk_overlap:
                            overlap_chunk.insert(0, s)
                            overlap_len += len(s) + 1
                        else:
                            break
                    current_chunk = overlap_chunk
                    current_len = sum(len(s) for s in current_chunk) + len(current_chunk) - 1
                
                current_chunk.append(sentence)
                current_len += len(sentence) + (1 if current_len > 0 else 0)
                
            if current_chunk:
                chunk_text = " ".join(current_chunk)
                chunks.append({
                    "id": chunk_id,
                    "text": chunk_text,
                    "page_num": page["page_num"],
                    "char_count": len(chunk_text),
                    "word_count": len(chunk_text.split())
                })
                chunk_id += 1
                
    else:  # Fixed-size Character
        for page in pages:
            page_text = page["text"].strip()
            if not page_text:
                continue
            
            i = 0
            while i < len(page_text):
                # Grab a chunk of fixed size
                chunk_text = page_text[i:i + chunk_size].strip()
                if chunk_text:
                    chunks.append({
                        "id": chunk_id,
                        "text": chunk_text,
                        "page_num": page["page_num"],
                        "char_count": len(chunk_text),
                        "word_count": len(chunk_text.split())
                    })
                    chunk_id += 1
                i += (chunk_size - chunk_overlap) if chunk_size > chunk_overlap else chunk_size
                
    return chunks
