import chromadb
from chromadb.config import Settings
import openai
from sentence_transformers import SentenceTransformer
import requests
from typing import List, Dict, Any
import os
import logging

class RAGAgent:
    def __init__(self):
        self.chroma_client = None
        self.collection = None
        self.embedding_model = None
        self.openai_client = None
        self.tavily_api_key = os.getenv("TAVILY_API_KEY")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        
    async def initialize(self):
        """Initialize the RAG agent components"""
        # Initialize ChromaDB
        self.chroma_client = chromadb.Client(Settings(anonymized_telemetry=False))
        
        # Create or get collection
        try:
            self.collection = self.chroma_client.get_collection("knowledge_base")
        except:
            self.collection = self.chroma_client.create_collection(
                name="knowledge_base",
                metadata={"hnsw:space": "cosine"}
            )
        
        # Initialize embedding model
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize OpenAI client
        if self.openai_api_key:
            self.openai_client = openai.OpenAI(api_key=self.openai_api_key)
        
        print("RAG Agent initialized successfully")
    
    async def add_document(self, text: str, filename: str):
        """Add document to ChromaDB vector store"""
        # Split text into chunks
        chunks = self.split_text(text, chunk_size=500, overlap=50)
        
        # Generate embeddings
        embeddings = self.embedding_model.encode(chunks).tolist()
        
        # Create document IDs
        doc_ids = [f"{filename}_{i}" for i in range(len(chunks))]
        
        # Add to ChromaDB
        self.collection.add(
            embeddings=embeddings,
            documents=chunks,
            ids=doc_ids,
            metadatas=[{"filename": filename, "chunk_id": i} for i in range(len(chunks))]
        )
        
        print(f"Added {len(chunks)} chunks from {filename} to vector store")
    
    def split_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """Split text into overlapping chunks"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            chunks.append(chunk)
            
            if i + chunk_size >= len(words):
                break
                
        return chunks
    
    async def search_vector_store(self, query: str, n_results: int = 3) -> List[Dict]:
        """Search ChromaDB vector store"""
        query_embedding = self.embedding_model.encode([query]).tolist()
        
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=n_results
        )
        
        return [
            {
                "content": doc,
                "metadata": metadata,
                "distance": distance
            }
            for doc, metadata, distance in zip(
                results['documents'][0],
                results['metadatas'][0],
                results['distances'][0]
            )
        ]
    
    async def web_search(self, query: str) -> List[Dict]:
        """Perform web search using Tavily"""
        if not self.tavily_api_key:
            return []
        
        try:
            url = "https://api.tavily.com/search"
            payload = {
                "api_key": self.tavily_api_key,
                "query": query,
                "search_depth": "advanced",
                "max_results": 3
            }
            
            response = requests.post(url, json=payload)
            if response.status_code == 200:
                data = response.json()
                return [
                    {
                        "content": result.get("content", ""),
                        "url": result.get("url", ""),
                        "title": result.get("title", "")
                    }
                    for result in data.get("results", [])
                ]
        except Exception as e:
            print(f"Web search error: {e}")
        
        return []
    
    async def query(self, user_query: str) -> Dict[str, Any]:
        """Main query method that coordinates the RAG pipeline"""
        # Step 1: Search vector store
        vector_results = await self.search_vector_store(user_query)
        
        # Step 2: Determine if web search is needed
        needs_web_search = self.should_use_web_search(user_query, vector_results)
        
        web_results = []
        if needs_web_search:
            web_results = await self.web_search(user_query)
        
        # Step 3: Generate response using retrieved context
        response = await self.generate_response(user_query, vector_results, web_results)
        
        return {
            "answer": response,
            "sources": self.extract_sources(vector_results, web_results)
        }
    
    def should_use_web_search(self, query: str, vector_results: List[Dict]) -> bool:
        """Determine if web search is needed based on query and vector results"""
        # Simple heuristics - in production, you'd use more sophisticated logic
        web_keywords = ["current", "latest", "recent", "today", "news", "2024", "2025"]
        query_lower = query.lower()
        
        # If vector results are poor quality or query seems time-sensitive
        has_web_keywords = any(keyword in query_lower for keyword in web_keywords)
        poor_vector_results = not vector_results or (vector_results and vector_results[0]["distance"] > 0.7)
        
        return has_web_keywords or poor_vector_results
    
    async def generate_response(self, query: str, vector_results: List[Dict], web_results: List[Dict]) -> str:
        """Generate response using LLM with retrieved context"""
        if not self.openai_client:
            return self.generate_simple_response(query, vector_results, web_results)
        
        # Prepare context
        context = ""
        
        if vector_results:
            context += "Knowledge Base Information:\n"
            for i, result in enumerate(vector_results[:3]):
                context += f"{i+1}. {result['content']}\n\n"
        
        if web_results:
            context += "Recent Web Information:\n"
            for i, result in enumerate(web_results):
                context += f"{i+1}. {result['title']}: {result['content']}\n\n"
        
        # Generate response
        try:
            messages = [
                {
                    "role": "system",
                    "content": """You are a helpful AI assistant with access to a knowledge base and web search.
                    Use the provided context to answer questions accurately. If the context doesn't contain
                    enough information, say so clearly. Always cite your sources when possible."""
                },
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion: {query}\n\nPlease provide a comprehensive answer based on the available context."
                }
            ]
            
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages,
                max_tokens=500,
                temperature=0.7
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"OpenAI error: {e}")
            return self.generate_simple_response(query, vector_results, web_results)
    
    def generate_simple_response(self, query: str, vector_results: List[Dict], web_results: List[Dict]) -> str:
        """Fallback response generation without OpenAI"""
        response = f"Based on your query: '{query}'\n\n"
        
        if vector_results:
            response += "From Knowledge Base:\n"
            for result in vector_results[:2]:
                response += f"• {result['content'][:200]}...\n"
        
        if web_results:
            response += "\nFrom Web Search:\n"
            for result in web_results[:2]:
                response += f"• {result['title']}: {result['content'][:200]}...\n"
        
        if not vector_results and not web_results:
            response += "I couldn't find relevant information in my knowledge base or through web search. Please try rephrasing your question."
        
        return response
    
    def extract_sources(self, vector_results: List[Dict], web_results: List[Dict]) -> List[str]:
        """Extract source information"""
        sources = []
        
        # Add vector store sources
        for result in vector_results:
            if result["metadata"].get("filename"):
                sources.append(f"Knowledge Base: {result['metadata']['filename']}")
        
        # Add web sources
        for result in web_results:
            if result.get("url"):
                sources.append(f"Web: {result['title']} - {result['url']}")
        
        return list(set(sources))  # Remove duplicates


class AdvancedRAGAgent(RAGAgent):
    """Enhanced RAG Agent with better decision-making logic"""
    
    def __init__(self):
        super().__init__()
        self.confidence_threshold = 0.6
    
    async def analyze_query_intent(self, query: str) -> Dict[str, Any]:
        """Analyze query to determine the best retrieval strategy"""
        query_lower = query.lower()
        
        # Time-sensitive indicators
        time_indicators = ["current", "latest", "recent", "today", "now", "2024", "2025"]
        is_time_sensitive = any(indicator in query_lower for indicator in time_indicators)
        
        # Factual vs analytical indicators
        factual_indicators = ["what is", "who is", "when did", "where is", "how many"]
        is_factual = any(indicator in query_lower for indicator in factual_indicators)
        
        # Complex reasoning indicators
        analytical_indicators = ["analyze", "compare", "explain why", "what are the implications"]
        needs_analysis = any(indicator in query_lower for indicator in analytical_indicators)
        
        return {
            "is_time_sensitive": is_time_sensitive,
            "is_factual": is_factual,
            "needs_analysis": needs_analysis,
            "priority_web": is_time_sensitive,
            "priority_vector": not is_time_sensitive
        }
    
    async def query(self, user_query: str) -> Dict[str, Any]:
        """Enhanced query method with intent analysis"""
        # Analyze query intent
        intent = await self.analyze_query_intent(user_query)
        
        vector_results = []
        web_results = []
        
        # Retrieve based on intent
        if intent["priority_vector"] or not intent["priority_web"]:
            vector_results = await self.search_vector_store(user_query, n_results=5)
        
        if intent["priority_web"] or (vector_results and self.is_low_confidence(vector_results)):
            web_results = await self.web_search(user_query)
        
        # If still no good results and haven't tried the other source, try it
        if self.is_low_confidence(vector_results) and not web_results and not intent["priority_web"]:
            web_results = await self.web_search(user_query)
        elif not vector_results and web_results and not intent["priority_vector"]:
            vector_results = await self.search_vector_store(user_query, n_results=3)
        
        # Generate response
        response = await self.generate_enhanced_response(
            user_query, vector_results, web_results, intent
        )
        
        return {
            "answer": response,
            "sources": self.extract_sources(vector_results, web_results),
            "search_strategy": intent
        }
    
    def is_low_confidence(self, results: List[Dict]) -> bool:
        """Check if vector search results are low confidence"""
        if not results:
            return True
        return results[0]["distance"] > self.confidence_threshold
    
    async def generate_enhanced_response(
        self, query: str, vector_results: List[Dict], 
        web_results: List[Dict], intent: Dict[str, Any]
    ) -> str:
        """Generate enhanced response with better context handling"""
        if not self.openai_client:
            return self.generate_simple_response(query, vector_results, web_results)
        
        # Build context with source prioritization
        context_parts = []
        
        if vector_results and (not intent["priority_web"] or not web_results):
            context_parts.append("Knowledge Base Documents:")
            for i, result in enumerate(vector_results[:3]):
                confidence = 1 - result["distance"]
                context_parts.append(f"{i+1}. [Confidence: {confidence:.2f}] {result['content']}")
        
        if web_results:
            context_parts.append("\nRecent Web Information:")
            for i, result in enumerate(web_results):
                context_parts.append(f"{i+1}. {result['title']}: {result['content']}")
        
        context = "\n".join(context_parts)
        
        # Enhanced system prompt
        system_prompt = f"""You are an intelligent RAG assistant. Use the provided context to answer questions.

Query Analysis:
- Time-sensitive: {intent['is_time_sensitive']}
- Factual query: {intent['is_factual']}
- Needs analysis: {intent['needs_analysis']}

Instructions:
1. Prioritize more recent/relevant sources
2. Clearly distinguish between knowledge base and web information
3. If information is insufficient, state what additional info would be helpful
4. For analytical queries, synthesize information from multiple sources
5. Always mention your confidence level in the answer
"""
        
        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
            ]
            
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages,
                max_tokens=600,
                temperature=0.3 if intent["is_factual"] else 0.7
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"Enhanced response generation error: {e}")
            return self.generate_simple_response(query, vector_results, web_results)