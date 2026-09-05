import os
import re
import math
from collections import Counter
import numpy as np
from openai import OpenAI
import ollama

class SimpleTFIDF:
    def __init__(self):
        self.vocab = {}
        self.idf = []
        self.doc_vectors = None
        self.chunks = []

    def fit_transform(self, documents: list[str]):
        if not documents:
            return
        
        # Clean and tokenize
        tokenized_docs = [self._tokenize(doc) for doc in documents]
        
        # Build vocab
        vocab_set = set()
        for doc in tokenized_docs:
            vocab_set.update(doc)
        
        self.vocab = {word: idx for idx, word in enumerate(sorted(list(vocab_set)))}
        num_docs = len(documents)
        
        # Calculate DF (document frequency) and IDF
        df = np.zeros(len(self.vocab))
        for doc in tokenized_docs:
            unique_words = set(doc)
            for word in unique_words:
                df[self.vocab[word]] += 1
        
        # IDF formula: log((1 + N)/(1 + df)) + 1
        self.idf = np.log((1 + num_docs) / (1 + df)) + 1
        
        # Build term frequency vectors
        self.doc_vectors = np.zeros((num_docs, len(self.vocab)))
        for i, doc in enumerate(tokenized_docs):
            counts = Counter(doc)
            for word, count in counts.items():
                if word in self.vocab:
                    self.doc_vectors[i, self.vocab[word]] = count
                    
        # Multiply TF * IDF
        self.doc_vectors = self.doc_vectors * self.idf
        
        # L2 Normalize the vectors for easy cosine similarity
        norms = np.linalg.norm(self.doc_vectors, axis=1, keepdims=True)
        # Avoid division by zero
        norms[norms == 0] = 1.0
        self.doc_vectors = self.doc_vectors / norms

    def _tokenize(self, text: str) -> list[str]:
        return re.findall(r'\b\w+\b', text.lower())

    def transform(self, query: str) -> np.ndarray:
        tokens = self._tokenize(query)
        vector = np.zeros(len(self.vocab))
        if not self.vocab:
            return vector
        counts = Counter(tokens)
        for word, count in counts.items():
            if word in self.vocab:
                vector[self.vocab[word]] = count
        vector = vector * self.idf
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        return vector

    def search(self, query: str, top_k: int = 3) -> list[tuple[int, float]]:
        if not self.vocab or self.doc_vectors is None or len(self.doc_vectors) == 0:
            return []
        q_vec = self.transform(query)
        # Cosine similarity is dot product of normalized vectors
        scores = np.dot(self.doc_vectors, q_vec)
        
        # Sort in descending order
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(int(idx), float(scores[idx])) for idx in top_indices]


class RAGEngine:
    def __init__(self, provider="Local (TF-IDF)", api_key=None, ollama_url="http://localhost:11434", ollama_model="llama3"):
        self.provider = provider
        self.api_key = api_key
        self.ollama_url = ollama_url
        self.ollama_model = ollama_model
        self.chunks = []
        self.tfidf_model = None
        self.embeddings = None
        
    def index_chunks(self, chunks: list[dict]):
        self.chunks = chunks
        if not chunks:
            return
        
        texts = [c["text"] for c in chunks]
        
        if self.provider == "Local (TF-IDF)":
            self.tfidf_model = SimpleTFIDF()
            self.tfidf_model.fit_transform(texts)
            
        elif self.provider == "OpenAI":
            if not self.api_key:
                raise ValueError("OpenAI API key is required.")
            client = OpenAI(api_key=self.api_key)
            
            # Embed chunks in batches to prevent API rate limits or issues
            batch_size = 100
            embeddings_list = []
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                response = client.embeddings.create(
                    model="text-embedding-3-small",
                    input=batch
                )
                embeddings_list.extend([emb.embedding for emb in response.data])
            
            self.embeddings = np.array(embeddings_list)
            # Normalize for cosine similarity
            norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            self.embeddings = self.embeddings / norms
            
        elif self.provider == "Ollama":
            # We can use the ollama library to generate embeddings
            embeddings_list = []
            for text in texts:
                try:
                    response = ollama.embeddings(model=self.ollama_model, prompt=text)
                    embeddings_list.append(response["embedding"])
                except Exception as e:
                    # Try using nomic-embed-text if model doesn't support embeddings or fails
                    try:
                        response = ollama.embeddings(model="nomic-embed-text", prompt=text)
                        embeddings_list.append(response["embedding"])
                    except Exception as sub_e:
                        raise RuntimeError(f"Ollama embedding failed. Ensure the model is running and supports embeddings. Error: {e}")
            
            self.embeddings = np.array(embeddings_list)
            norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            self.embeddings = self.embeddings / norms

    def retrieve(self, query: str, top_k: int = 3, similarity_threshold: float = 0.0) -> list[dict]:
        if not self.chunks:
            return []
        
        results = []
        if self.provider == "Local (TF-IDF)":
            if not self.tfidf_model:
                return []
            matches = self.tfidf_model.search(query, top_k=top_k)
            for idx, score in matches:
                if score < similarity_threshold:
                    continue
                # Add score to chunk info
                chunk_info = self.chunks[idx].copy()
                chunk_info["score"] = score
                results.append(chunk_info)
                
        elif self.provider in ["OpenAI", "Ollama"]:
            if self.embeddings is None or len(self.embeddings) == 0:
                return []
            
            # Generate query embedding
            if self.provider == "OpenAI":
                client = OpenAI(api_key=self.api_key)
                response = client.embeddings.create(
                    model="text-embedding-3-small",
                    input=[query]
                )
                q_emb = np.array(response.data[0].embedding)
            else:
                try:
                    response = ollama.embeddings(model=self.ollama_model, prompt=query)
                    q_emb = np.array(response["embedding"])
                except Exception:
                    response = ollama.embeddings(model="nomic-embed-text", prompt=query)
                    q_emb = np.array(response["embedding"])
                    
            # Normalize query vector
            norm = np.linalg.norm(q_emb)
            if norm > 0:
                q_emb = q_emb / norm
                
            # Compute cosine similarity
            scores = np.dot(self.embeddings, q_emb)
            top_indices = np.argsort(scores)[::-1]
            
            for idx in top_indices:
                score = float(scores[idx])
                if score < similarity_threshold:
                    continue
                chunk_info = self.chunks[idx].copy()
                chunk_info["score"] = score
                results.append(chunk_info)
                if len(results) >= top_k:
                    break
                
        return results

    def generate_response(self, query: str, retrieved_chunks: list[dict], chat_history: list[dict] = None, system_prompt_template: str = None, openai_model: str = "gpt-4o-mini") -> str:
        """
        Generates a response to the query using the retrieved chunks.
        Returns a generator of strings to support streaming.
        """
        if not retrieved_chunks:
            yield "No context is available. Please upload and process a PDF first."
            return
            
        context_str = "\n\n".join([f"[Source {i+1} - Page {c['page_num']}]: {c['text']}" for i, c in enumerate(retrieved_chunks)])
        
        if system_prompt_template:
            if "{context}" in system_prompt_template:
                system_prompt = system_prompt_template.replace("{context}", context_str)
            else:
                system_prompt = f"{system_prompt_template}\n\n--- CONTEXT ---\n{context_str}\n----------------"
        else:
            system_prompt = (
                "You are a helpful AI assistant. Answer the user's question based strictly on the provided context. "
                "If the answer cannot be found in the context, state that you do not know. "
                "Refer to the sources by their number (e.g., [Source 1], [Source 2]) when citing facts.\n\n"
                f"--- CONTEXT ---\n{context_str}\n----------------"
            )
        
        messages = [{"role": "system", "content": system_prompt}]
        
        # Include history if available
        if chat_history:
            # Add last 6 messages to avoid context window explosion
            for msg in chat_history[-6:]:
                messages.append({"role": msg["role"], "content": msg["content"]})
                
        messages.append({"role": "user", "content": query})
        
        if self.provider == "Local (TF-IDF)":
            # Synthesize a simple extractive local summary
            yield "*(Running in Local Offline Mode. Generating local extractive answer)*\n\n"
            
            # Simple summarization: extract sentences containing words from query or key sentences
            query_words = set(re.findall(r'\b\w+\b', query.lower()))
            best_sentences = []
            for chunk in retrieved_chunks:
                # split into sentences
                sentences = re.split(r'(?<=[.!?])\s+', chunk["text"])
                for s in sentences:
                    s_words = set(re.findall(r'\b\w+\b', s.lower()))
                    overlap = len(query_words.intersection(s_words))
                    if overlap > 0:
                        best_sentences.append((overlap, s.strip(), chunk["page_num"]))
            
            # Sort sentences by overlap score
            best_sentences.sort(key=lambda x: x[0], reverse=True)
            
            if best_sentences:
                seen = set()
                yield "Based on the matching sections of the document:\n\n"
                count = 0
                for score, sent, pagenum in best_sentences:
                    if sent.lower() not in seen and len(sent) > 10:
                        seen.add(sent.lower())
                        yield f"- \"{sent}\" **(Page {pagenum})**\n\n"
                        count += 1
                        if count >= 3:
                            break
            else:
                yield "No exact sentence matches found for your query. Here are the top matched passages:\n\n"
                for i, c in enumerate(retrieved_chunks):
                    yield f"**From Page {c['page_num']} (Match score: {c['score']:.2f}):**\n>{c['text']}\n\n"
                    
        elif self.provider == "OpenAI":
            client = OpenAI(api_key=self.api_key)
            response = client.chat.completions.create(
                model=openai_model,
                messages=messages,
                temperature=0.3,
                stream=True
            )
            for chunk in response:
                content = chunk.choices[0].delta.content
                if content is not None:
                    yield content
                    
        elif self.provider == "Ollama":
            try:
                response = ollama.chat(
                    model=self.ollama_model,
                    messages=messages,
                    stream=True
                )
                for chunk in response:
                    content = chunk['message']['content']
                    if content:
                        yield content
            except Exception as e:
                yield f"Error calling local Ollama instance: {str(e)}. Please check that Ollama is running (`ollama serve`) and the model `{self.ollama_model}` is pulled (`ollama pull {self.ollama_model}`)."
