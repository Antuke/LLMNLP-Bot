import ollama
from typing import List, Tuple, Dict
import numpy as np
from pathlib import Path
from dataclasses import dataclass

@dataclass
class ctx:
    filename: str
    scores: dict

class IndexRag:
    def __init__(self, build_vectorstore=False):
        self.embedding_model = "nomic-embed-text"
        self.index = "./data/index.txt"
        self.embeddings = {}  # Store embeddings for document samples
        self.keywords = {}    # Store keywords for keyword matching
        self.keyword_weights = {}  # Store keyword importance weights
        
        if build_vectorstore:
            self.build_store()
    
    def split_chunks(self, separator="$$$") -> List[Tuple[str, str, str]]:
        """
        Split the index file into chunks based on separator
        Returns list of tuples (filename, keywords, samples)
        """
        with open(self.index, 'r') as f:
            content = f.read()
        
        chunks = content.strip().split(separator)
        parsed_chunks = []
        
        for chunk in chunks:
            chunk = chunk.strip()
            if chunk:
                # Split by pipe for new format
                parts = [part.strip() for part in chunk.split('|')]
                if len(parts) == 3:
                    filename, keywords, samples = parts
                else:
                    print("File is not formatted properl")
                parsed_chunks.append((filename.strip(), keywords.strip(), samples.strip()))
                
        return parsed_chunks
    
    def retrieve_doc_content(self, query: str, top_k: int = 2, semantic_weight: float = 0.7, threshold: float = 0.4) -> List[Tuple[str, float]]:
        """Retrieves the content of the document if queries requires it"""
        retrieved = self.retrieve(query=query,top_k=top_k,semantic_weight=semantic_weight) 
        contents = []

        if len(retrieved) != 0:
            print("")

        for rtrv in retrieved:
            if rtrv[1]['combined_score'] < threshold:
                return contents
            print(f"Retrieving file {rtrv[0]}, with a relevance score of: {rtrv[1]['combined_score']}")
            contents.append(self.load_document(rtrv[0]))

        return contents

    def calculate_keyword_weights(self):
        """
        Calculate IDF weights for keywords
        """
        keyword_doc_freq = {}
        total_docs = len(self.keywords)
        
        # count how many times a keyword appears in the index 
        for keywords in self.keywords.values():
            keyword_list = keywords.split() # go from string to list
            for keyword in keyword_list:
                keyword_doc_freq[keyword] = keyword_doc_freq.get(keyword, 0) + 1
        
        for filename, keywords in self.keywords.items():
            weights = {}
            keyword_list = keywords.split()
            
            for keyword in keyword_list:
                idf = np.log(total_docs / keyword_doc_freq[keyword])
                # in our case, a keyword appears only once so no need to calculate the term frequency
                # tf = keyword_list.count(keyword) / len(keyword_list)
                tf = 1
                weights[keyword.lower()] = tf * idf
                
            self.keyword_weights[filename] = weights
    
    def build_store(self):
        """
        Build the vector store by creating embeddings for sample questions
        """
        chunks = self.split_chunks()
        prefix = "search_document:"
        print("Building embeddings for documents...")
        
        for filename, keywords, samples in chunks:
            self.keywords[filename] = keywords
            #self.samples[filename] = samples
            
            # Use sample questions for embedding if available, otherwise use keywords
            embedding_text = samples if samples else keywords
            prefixed_text = f"{prefix}{embedding_text}"

            
            try:
                response = ollama.embed(
                    model=self.embedding_model,
                    input=prefixed_text
                )
                
                if 'embeddings' in response and response['embeddings']:
                    self.embeddings[filename] = response['embeddings'][0]
                else:
                    print(f"Warning: No embedding generated for {filename}")
                    
            except Exception as e:
                print(f"Error embedding document {filename}: {str(e)}")
        
        self.calculate_keyword_weights()
    
    def semantic_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """
        Calculate cosine similarity between two vectors
        """
        vec1 = np.array(vec1).flatten()
        vec2 = np.array(vec2).flatten()
        
        norm_product = np.linalg.norm(vec1) * np.linalg.norm(vec2)
        if norm_product == 0:
            return 0
            
        return np.dot(vec1, vec2) / norm_product
    
    def keyword_similarity(self, query: str, filename: str) -> float:
        """
        Calculate keyword-based similarity score
        """
        query_terms = set(query.lower().split())
        # query_terms = query.lower().split()
        doc_weights = self.keyword_weights[filename]
        
        score = 0.0
        for term in query_terms:
            if term in doc_weights:
                score += doc_weights[term]
            else:
                for doc_term, weight in doc_weights.items():
                    if (term in doc_term or doc_term in term) and len(term) >= 4 and len(doc_term) >= 4:
                        score += weight * 0.5  # half-score for substrings
        
        return score
    
    
    
    def retrieve(self, query: str, top_k: int = 1, semantic_weight: float = 0.7) -> List[Tuple[str, Dict]]:
        """
        Hybrid retrieval combining semantic and keyword matching
        """
        keyword_weight = 1 - semantic_weight
        prefix = "search_query:"

        try:
            prefixed_query = f"{prefix}{query}"
            response = ollama.embed(
                model=self.embedding_model,
                input=prefixed_query
            )
            
            if 'embeddings' not in response or not response['embeddings']:
                raise ValueError("No embedding generated for query")
                
            query_embedding = response['embeddings'][0]
            
            combined_scores = []
            for filename, doc_embedding in self.embeddings.items():
                semantic_score = self.semantic_similarity(query_embedding, doc_embedding)
                keyword_score = self.keyword_similarity(query, filename)
                
                combined_score = (semantic_score * semantic_weight) + (keyword_score * keyword_weight)
                #matching_sample = self.get_matching_sample(filename, query)
                
                combined_scores.append((
                    filename, 
                    {
                        'combined_score': combined_score,
                        'semantic_score': semantic_score,
                        'keyword_score': keyword_score,
                    }
                ))
            
            combined_scores.sort(key=lambda x: x[1]['combined_score'], reverse=True)
            return combined_scores[:top_k]
            
        except Exception as e:
            print(f"Error during retrieval: {str(e)}")
            return []
        

    def load_document(self,filename):
        """
        Load the actual document content given the filename
        """
        file_path = Path("./data") / filename
        try:
            with open(file_path, 'r') as f:
                return f.read()
        except FileNotFoundError:
                return f"Document {filename} not found"





#Run RAG directly, to test the retrieval function of the chatbot
def main():
    print("Initializing RAG system and building vector store...")
    rag = IndexRag(build_vectorstore=True)
    print("System ready! Enter your queries (type 'exit' to quit)")
    
    while True:
        print("\n" + "="*50)
        query = input("Enter your query: ").strip()
        
        if query.lower() == 'exit':
            print("Goodbye!")
            break
        
        if not query:
            print("Please enter a valid query")
            continue
        
        try:
            results = rag.retrieve(query, top_k=3, semantic_weight=0.7)
            
            if not results:
                print("No matching documents found")
                continue
                
            print("\nTop matching documents:")
            for i, (filename, scores) in enumerate(results, 1):
                print(f"\n{i}. {filename}")
                print(f"   Combined Score: {scores['combined_score']:.4f}")
                print(f"   Semantic Score: {scores['semantic_score']:.4f}")
                print(f"   Keyword Score: {scores['keyword_score']:.4f}")
                print(f"   Keywords: {rag.keywords[filename]}")

        
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            continue

if __name__ == "__main__":
    main()