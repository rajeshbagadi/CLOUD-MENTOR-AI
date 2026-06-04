import re
from typing import List, Set, Dict, Any, Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from src.rag.retriever import SemanticRetriever, RetrievedChunk
from src.model.gemini_client import GeminiClientManager
from src.core.logging_config import setup_logger

logger = setup_logger("comparison_engine")


class AWSComparisonEngine:
    """Dedicated RAG engine that identifies service comparisons and executes targeted multi-query retrieval."""

    def __init__(self, retriever: SemanticRetriever, gemini_manager: GeminiClientManager):
        """Initializes the AWSComparisonEngine.
        
        Args:
            retriever (SemanticRetriever): Ingestion retriever instance.
            gemini_manager (GeminiClientManager): Gemini client interface.
        """
        self.retriever = retriever
        self.gemini_manager = gemini_manager

    def is_comparison_query(self, query: str) -> bool:
        """Determines if the query is requesting a service comparison.
        
        Args:
            query (str): The raw user query.
            
        Returns:
            bool: True if keywords indicate comparison, False otherwise.
        """
        comparison_regex = [
            r"\bcompare\b", 
            r"\bvs\b", 
            r"\bversus\b", 
            r"\bdifference\b", 
            r"\bdifferences\b", 
            r"\bor\b", 
            r"\balternative\b",
            r"\balternatives\b"
        ]
        query_lower = query.lower()
        return any(re.search(pattern, query_lower) for pattern in comparison_regex)

    def extract_services(self, query: str) -> List[str]:
        """Scans the query to extract known AWS service identifiers.
        
        Args:
            query (str): User's query.
            
        Returns:
            List[str]: List of recognized AWS service names in uppercase.
        """
        # Dictionary of standard service aliases to search for
        aws_service_glossary = [
            "ec2", "lambda", "s3", "rds", "dynamodb", "fargate", "ecs", "eks", 
            "sqs", "sns", "cloudfront", "apigateway", "ebs", "efs", "aurora", 
            "redshift", "route53", "vpc", "iam", "cognito", "kinesis", "glue"
        ]
        query_lower = query.lower()
        found_services = []
        for service in aws_service_glossary:
            if re.search(rf"\b{service}\b", query_lower):
                found_services.append(service.upper())
        return found_services

    def execute_comparison(self, query: str, services: List[str], k_per_service: int = 3) -> str:
        """Runs targeted multi-query retrieval for each service to build a balanced context block.
        
        Normal similarity searches suffer from semantic retrieval bias (e.g. searching "EC2 vs Lambda"
        might retrieve 4 chunks about Lambda and none about EC2). This method performs targeted searches
        for each service, combines the results, and structures a direct comparison response.
        
        Args:
            query (str): Original query.
            services (List[str]): Extracted target services.
            k_per_service (int): Number of document chunks to fetch per service.
            
        Returns:
            str: Generated structured Markdown comparison table and analysis.
        """
        logger.info(f"Executing AWS comparison engine for extracted services: {services}")
        
        combined_chunks: List[RetrievedChunk] = []
        retrieved_identifiers: Set[str] = set()
        
        # Perform targeted queries for each identified service to balance context representation
        search_terms = services if services else [query]
        
        for term in search_terms:
            logger.debug(f"Targeted comparison query retrieval for term: '{term}'")
            chunks = self.retriever.retrieve_chunks(term, k=k_per_service)
            for chunk in chunks:
                # Deduplicate chunks using filename and content hash
                doc_hash = hashlib_val = f"{chunk.source}_page_{chunk.page}_{hash(chunk.content)}"
                if doc_hash not in retrieved_identifiers:
                    retrieved_identifiers.add(doc_hash)
                    combined_chunks.append(chunk)

        if not combined_chunks:
            logger.warning("No context records found in the database to build service comparisons.")
            return (
                "I cannot find any documentation records in the vector database to perform this comparison. "
                "Please upload relevant AWS guides or documentation PDF files."
            )

        # Build context payload block
        context_payload = []
        for idx, chunk in enumerate(combined_chunks):
            context_payload.append(
                f"--- DOCUMENT CHUNK {idx + 1} (Source: {chunk.source}, Page: {chunk.page}) ---\n"
                f"{chunk.content}"
            )
        context_str = "\n\n".join(context_payload)

        # Setup strict markdown structure layout
        system_instructions = (
            "You are CloudMentor AI, an expert Senior AWS Solutions Architect.\n"
            "Your task is to compare the requested AWS services using ONLY the provided documentation context.\n"
            "Structure your output using this exact Markdown template:\n\n"
            "### 📊 AWS Service Comparison\n\n"
            "| Feature | [Service A] | [Service B] |\n"
            "| :--- | :--- | :--- |\n"
            "| **Core Concept** | ... | ... |\n"
            "| **Pricing Model** | ... | ... |\n"
            "| **Scaling Behavior** | ... | ... |\n"
            "| **Operational Overhead** | ... | ... |\n\n"
            "### 🔑 Key Differences\n"
            "- **Difference 1**: Describe a key architectural or feature difference.\n"
            "- **Difference 2**: Describe another key difference (pricing, limits, etc.).\n\n"
            "### 🎯 Recommended Use Cases\n"
            "* **Use [Service A] when**:\n"
            "  - Requirement detail 1.\n"
            "  - Requirement detail 2.\n"
            "* **Use [Service B] when**:\n"
            "  - Requirement detail 1.\n"
            "  - Requirement detail 2.\n\n"
            "### ⚡ Advantages & Limitations\n"
            "#### [Service A]\n"
            "* **Advantages**: Bullet list of benefits.\n"
            "* **Limitations**: Bullet list of constraints.\n\n"
            "#### [Service B]\n"
            "* **Advantages**: Bullet list of benefits.\n"
            "* **Limitations**: Bullet list of constraints.\n\n"
            "--- Rules:\n"
            "1. Grounding: Rely strictly on facts from the context. If details for a comparison cell are missing, "
            "write 'N/A from context'.\n"
            "2. Citations: Every statement, recommendation, or metric you write MUST refer to its source page. "
            "Cite sources inline using the exact format: [source: filename.pdf, page: X]."
        )

        prompt_template = ChatPromptTemplate.from_messages([
            ("system", system_instructions),
            (
                "user", 
                "AWS DOCUMENTATION CONTEXT:\n{context}\n\n"
                "COMPARISON QUERY: {query}"
            )
        ])

        llm = self.gemini_manager.get_llm()
        generation_chain = prompt_template | llm | StrOutputParser()

        logger.info("Generating comparison output via Gemini...")
        comparison_result = generation_chain.invoke({
            "context": context_str,
            "query": query
        })
        
        return comparison_result
