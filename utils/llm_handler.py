from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain.chains import RetrievalQA
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, SystemMessage
from config import Config
import logging
from typing import Dict, Any, List, Optional

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMHandler:
    def __init__(self):
        self.config = Config()
        self.llm = None
        self.qa_chain = None
        self.initialized = False
        
    def load_llm(self) -> bool:
        """Initialize the LLM with HuggingFace
        
        Returns:
            bool: True if initialization was successful, False otherwise
        """
        try:
            if not self.config.HF_TOKEN:
                logger.error("HuggingFace token not configured")
                return False

            logger.info(f"Initializing LLM with model: {self.config.HUGGINGFACE_REPO_ID}")
            
            # Option 1: Try with ChatHuggingFace for conversational models
            try:
                base_llm = HuggingFaceEndpoint(
                    repo_id=self.config.HUGGINGFACE_REPO_ID,
                    temperature=self.config.TEMPERATURE,
                    huggingfacehub_api_token=self.config.HF_TOKEN,
                    timeout=60,
                    task="conversational"  # Specify the correct task
                )
                
                self.llm = ChatHuggingFace(llm=base_llm)
                
                # Test the connection
                test_response = self.llm.invoke([HumanMessage(content="Hello")])
                if not test_response:
                    raise ValueError("Empty response from LLM")
                
                logger.info("LLM initialized successfully with ChatHuggingFace")
                return True
                
            except Exception as e:
                logger.warning(f"ChatHuggingFace initialization failed: {str(e)}")
                
            # Option 2: Try with a text-generation compatible model
            text_gen_models = [
                "microsoft/DialoGPT-medium",
                "google/flan-t5-base",
                "HuggingFaceH4/zephyr-7b-beta"
            ]
            
            for model in text_gen_models:
                try:
                    logger.info(f"Trying alternative model: {model}")
                    self.llm = HuggingFaceEndpoint(
                        repo_id=model,
                        temperature=self.config.TEMPERATURE,
                        huggingfacehub_api_token=self.config.HF_TOKEN,
                        timeout=60,
                        max_new_tokens=self.config.MAX_LENGTH
                    )
                    
                    # Test the connection
                    test_response = self.llm.invoke("Hello")
                    if not test_response:
                        raise ValueError("Empty response from LLM")
                    
                    logger.info(f"LLM initialized successfully with model: {model}")
                    # Update config to remember working model
                    self.config.HUGGINGFACE_REPO_ID = model
                    return True
                    
                except Exception as e:
                    logger.warning(f"Model {model} failed: {str(e)}")
                    continue

            logger.error("All initialization attempts failed")
            return False
            
        except Exception as e:
            logger.error(f"Error loading LLM: {str(e)}")
            return False
    
    def create_custom_prompt(self) -> PromptTemplate:
        """Create custom prompt template for medical QA
        
        Returns:
            PromptTemplate: Configured prompt template
        """
        template = """
        You are a medical assistant chatbot. Use the pieces of information provided in the context to answer the user's medical question.
        
        Important guidelines:
        - Only answer based on the provided context
        - If you don't know the answer from the context, say "I don't have enough information to answer this question based on the available medical documents."
        - Provide accurate medical information but always recommend consulting with healthcare professionals
        - Be helpful but cautious with medical advice
        - If the question is not medical-related, politely decline to answer
        - Format your response clearly with bullet points or numbered lists when appropriate
        
        Context: {context}
        Question: {question}
        
        Answer:
        """
        
        return PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
    
    def create_qa_chain(self, vectorstore: Any) -> bool:
        """Create QA chain with retriever
        
        Args:
            vectorstore: Initialized vector store for document retrieval
            
        Returns:
            bool: True if chain creation was successful, False otherwise
        """
        try:
            if not self.load_llm():
                raise Exception("Failed to load LLM")

            if not vectorstore:
                raise Exception("Vector store not provided")

            retriever = vectorstore.as_retriever(
                search_kwargs={
                    'k': self.config.TOP_K_RESULTS
                }
            )
            
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=retriever,
                return_source_documents=True,
                chain_type_kwargs={
                    'prompt': self.create_custom_prompt(),
                    'document_prompt': PromptTemplate(
                        input_variables=["page_content"],
                        template="{page_content}"
                    )
                }
            )
            
            logger.info("QA chain created successfully")
            self.initialized = True
            return True
            
        except Exception as e:
            logger.error(f"Error creating QA chain: {str(e)}")
            self.initialized = False
            return False
    
    def get_response(self, query: str) -> Dict[str, Any]:
        """Get response for a query with enhanced error handling
        
        Args:
            query: User's question
            
        Returns:
            dict: Response containing answer, sources, and error information
        """
        try:
            if not self.initialized or not self.qa_chain:
                return {
                    "error": "QA chain not initialized. Please try again later.",
                    "answer": None,
                    "sources": []
                }
            
            if not query or not isinstance(query, str):
                return {
                    "error": "Invalid query format",
                    "answer": None,
                    "sources": []
                }

            logger.info(f"Processing query: {query[:50]}...")
            
            response = self.qa_chain.invoke({'query': query})
            
            # Process source documents
            sources = []
            if response.get("source_documents"):
                for doc in response["source_documents"]:
                    source_info = {
                        "content": doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content,
                        "metadata": doc.metadata
                    }
                    sources.append(source_info)

            return {
                "error": None,
                "answer": response.get("result", "No answer generated"),
                "sources": sources,
                "query": query
            }
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return {
                "error": f"Error generating response: {str(e)}",
                "answer": None,
                "sources": []
            }

    def is_initialized(self) -> bool:
        """Check if the handler is ready to process queries
        
        Returns:
            bool: True if initialized and ready, False otherwise
        """
        return self.initialized