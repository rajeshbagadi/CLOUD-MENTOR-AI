import re
from typing import List, Set, Dict, Any, Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from src.rag.retriever import SemanticRetriever, RetrievedChunk
from src.model.gemini_client import GeminiClientManager
from src.core.logging_config import setup_logger

logger = setup_logger("recommendation_engine")


class AWSRecommendationEngine:
    """Specialized RAG engine that identifies system design requests and generates AWS architecture blueprints."""

    def __init__(self, retriever: SemanticRetriever, gemini_manager: GeminiClientManager):
        """Initializes the AWSRecommendationEngine.
        
        Args:
            retriever (SemanticRetriever): Ingestion retriever instance.
            gemini_manager (GeminiClientManager): Gemini client interface.
        """
        self.retriever = retriever
        self.gemini_manager = gemini_manager

    def is_recommendation_query(self, query: str) -> bool:
        """Heuristically checks if the user is asking for architectural recommendations or designs.
        
        Args:
            query (str): User query.
            
        Returns:
            bool: True if query seeks system design recommendations, False otherwise.
        """
        design_keywords = [
            r"\barchitecture\b", 
            r"\brecommend\b", 
            r"\bdesign\b", 
            r"\bhow to build\b", 
            r"\bhow to host\b", 
            r"\bhow to deploy\b", 
            r"\bblueprint\b",
            r"\bsetup\b",
            r"\binfra\b",
            r"\binfrastructure\b",
            r"\bworkflow\b"
        ]
        query_lower = query.lower()
        return any(re.search(pattern, query_lower) for pattern in design_keywords)

    def execute_recommendation(self, query: str, k: int = 5) -> str:
        """Retrieves system architecture guidelines from vector store and generates a tailored AWS blueprint.
        
        Args:
            query (str): Original query.
            k (int): Number of background context chunks to retrieve.
            
        Returns:
            str: Markdown formatted architecture blueprint with Mermaid diagram.
        """
        logger.info(f"Executing AWS architecture recommendation engine for query: '{query}'")
        
        # 1. Retrieve documentation chunks related to the system architecture query
        retrieved_chunks = self.retriever.retrieve_chunks(query, k=k)
        
        if not retrieved_chunks:
            logger.warning("No context records found in the database to compile recommendations.")
            return (
                "I cannot find any documentation records in the vector database to build an architecture recommendation. "
                "Please upload relevant AWS architecture guides or PDF files."
            )

        # Build context payload block
        context_payload = []
        for idx, chunk in enumerate(retrieved_chunks):
            context_payload.append(
                f"--- DOCUMENT CHUNK {idx + 1} (Source: {chunk.source}, Page: {chunk.page}) ---\n"
                f"{chunk.content}"
            )
        context_str = "\n\n".join(context_payload)

        # Setup structured prompt instructions for standard solutions architect blueprinting
        system_instructions = (
            "You are CloudMentor AI, an expert Principal AWS Solutions Architect.\n"
            "Your task is to design a secure, reliable, and scalable AWS architecture based on the user's requirements "
            "and the provided documentation context.\n"
            "Generate your output using this exact Markdown template:\n\n"
            "### 🏗️ AWS Architecture Recommendation: [App/System Type]\n\n"
            "#### 1. Architecture Overview\n"
            "Provide a clear narrative explanation of the proposed architecture pattern (e.g. serverless, microservices, 3-tier web app) "
            "and why it fits the request.\n\n"
            "#### 2. Component Design (Mermaid Flowchart)\n"
            "Generate a Mermaid JS flowchart illustrating the components and data flow. "
            "Wrap it inside a standard ```mermaid code block. Start with 'graph TD' or 'graph LR'. "
            "Example:\n"
            "```mermaid\n"
            "graph TD\n"
            "  User[User Client] --> Route53[Amazon Route 53]\n"
            "  Route53 --> CF[Amazon CloudFront]\n"
            "```\n\n"
            "#### 3. Recommended AWS Services\n"
            "*   **Compute ([Service Name])**: Role of compute (e.g. AWS Lambda, Amazon EC2, Amazon ECS).\n"
            "*   **Database/Storage ([Service Name])**: Role of storage (e.g. Amazon RDS, DynamoDB, Amazon S3).\n"
            "*   **Networking/Delivery ([Service Name])**: Network configurations (e.g. Amazon VPC, Route 53, CloudFront).\n"
            "*   **Integration ([Service Name])**: Integration/Messaging layer (e.g. SQS, SNS, API Gateway).\n\n"
            "#### 4. Well-Architected Best Practices\n"
            "*   **Security & Compliance**: Detail IAM policies, security groups, database encryption, and certificate management.\n"
            "*   **Reliability & High Availability**: Describe multi-AZ setups, auto-scaling thresholds, and backups.\n"
            "*   **Operational Excellence / Cost Optimization**: Advise on resource monitoring, serverless execution limits, or instance savings plans.\n\n"
            "--- Rules:\n"
            "1. Grounding: Rely strictly on facts, service configurations, and limits detailed in the provided context. "
            "Do not introduce services or configurations not referenced in the context blocks.\n"
            "2. Citations: Every statement, architecture choice, or metric you output MUST refer to its source page. "
            "Cite sources inline using the exact format: [source: filename.pdf, page: X]."
        )

        prompt_template = ChatPromptTemplate.from_messages([
            ("system", system_instructions),
            (
                "user", 
                "AWS DOCUMENTATION CONTEXT:\n{context}\n\n"
                "RECOMMENDATION QUERY: {query}"
            )
        ])

        llm = self.gemini_manager.get_llm()
        generation_chain = prompt_template | llm | StrOutputParser()

        logger.info("Generating architecture recommendation blueprint via Gemini...")
        recommendation_result = generation_chain.invoke({
            "context": context_str,
            "query": query
        })
        
        return recommendation_result
