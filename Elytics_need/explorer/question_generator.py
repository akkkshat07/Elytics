"""
Question Generator for Explorer Metadata

Generates contextual sample questions based on client's database schema,
table descriptions, and column metadata.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, List, Dict, Optional
import os
# SECURITY: Use defusedxml for parsing (prevents XXE attacks)
import defusedxml.ElementTree as ET
from defusedxml.ElementTree import parse, fromstring
# Import Element creation classes from standard library (safe for creation, not parsing)
# Safe: Only used for element creation, not parsing untrusted input. All parsing uses defusedxml.
from xml.etree.ElementTree import Element, SubElement, ElementTree, indent, ParseError  # nosemgrep: python.lang.security.use-defused-xml.use-defused-xml
from glob import glob
from datetime import datetime

from util.llm_utils import LLMClient
from util.dataset_paths import resolve_xml_data_sources_dir

logger = logging.getLogger(__name__)


class QuestionGenerator:
    """Generate sample questions based on explorer metadata."""
    
    def __init__(self, client_id: str, db: Optional[Any] = None, dataset_id: Optional[str] = None):
        self.client_id = client_id
        self.db = db
        self.dataset_id = dataset_id
        self.base_dir = resolve_xml_data_sources_dir(client_id, dataset_id, for_write=True)
        self.meta_dir = self.base_dir / "meta_information"
        self.desc_dir = self.base_dir / "data_descriptions"
        self.questions_file = self.base_dir / "suggested_questions.xml"
        self.base_sap_dir = Path("xml_prompts/base_sap") / "data_sources"
        
        # Initialize LLMClient with explorer_agent
        try:
            self.llm_client = LLMClient(
                agent_name="explorer_agent",
                client_id=client_id,
                db=db
            )
            logger.info(f"Initialized LLMClient for explorer_agent with client_id={client_id}")
        except Exception as e:
            logger.error(f"Failed to initialize LLMClient: {e}")
            self.llm_client = None
        
    async def _load_metadata(self) -> Dict:
        """Load and parse explorer metadata from XML files."""
        # Determine db_type to know if we should use base_sap
        db_type = None
        
        # Try to get db if not provided
        db_to_use = self.db
        if db_to_use is None:
            try:
                from db_config.mongo_server import get_db
                db_to_use = await get_db()
            except Exception as e:
                logger.warning(f"Failed to get db connection for QuestionGenerator: {e}")
        
        if db_to_use is not None:
            try:
                from services.db_credentials_service import DBCredentialsService
                service = DBCredentialsService(db_to_use)
                credentials = await service.get_credentials(
                    self.client_id,
                    db_type=None,
                    decrypt_password=False,
                    dataset_id=self.dataset_id,
                )
                if credentials:
                    db_type = credentials.get("db_type")
            except Exception as e:
                logger.warning(f"Failed to load db_type for QuestionGenerator: {e}")
        
        # For sap_oracle and sap_sybase, always use base_sap (table_introductions.xml is used )
        if db_type in ("sap_oracle", "sap_sybase"):
            if self.base_sap_dir.exists():
                meta_dir = self.base_sap_dir / "meta_information"
                desc_dir = self.base_sap_dir / "data_descriptions"
                logger.info(
                    "QuestionGenerator: Using base_sap metadata for %s client %s",
                    db_type,
                    self.client_id,
                )
            else:
                raise FileNotFoundError(f"base_sap metadata not found for {db_type} client {self.client_id}")
        else:
            # For other databases (postgres, mysql, sap_hana, etc.), use client-specific metadata
            meta_dir = self.meta_dir
            desc_dir = self.desc_dir
            logger.info("QuestionGenerator: Using client-specific metadata for client %s (db_type=%s)", self.client_id, db_type)
                    
        if not meta_dir.exists():
            raise FileNotFoundError(f"Metadata directory not found for client {self.client_id}")
        
        intros_file = meta_dir / "table_introductions.xml"
        
        if not intros_file.exists():
            raise FileNotFoundError(f"table_introductions.xml not found for client {self.client_id}")
        
        # Parse table_introductions.xml to get table list
        intros_tree = parse(intros_file)
        intros_root = intros_tree.getroot()
        
        # Extract first 5 table names from table_introductions (matching _build_prompt limit)
        # OPTIMIZATION: Only parse description files for tables we'll actually use
        table_intro_nodes = intros_root.findall(".//table_introduction")[:5]
        tables_to_load = [node.get("table_name") for node in table_intro_nodes if node.get("table_name")]
        
        total_tables = len(intros_root.findall(".//table_introduction"))
        if total_tables > 5:
            logger.info(
                f"QuestionGenerator: Optimizing - only loading descriptions for first 5 tables "
            )
        
        # Parse table introductions
        intros_map = {
            elem.get("table_name"): (elem.text or "").strip()
            for elem in intros_root.findall(".//table_introduction")
        }
        
        # Parse column descriptions - OPTIMIZED: only for tables we'll actually use
        column_desc_map = {}
        column_type_map = {}
        if desc_dir.exists():
            # Only parse descriptions for first 5 tables (matching _build_prompt limit)
            # This significantly speeds up question generation for schemas with many tables
            for table_name in tables_to_load:
                if not table_name:
                    continue
                desc_file = desc_dir / f"{table_name}_description.xml"
                if desc_file.exists():
                    try:
                        tree = parse(desc_file)
                        root = tree.getroot()
                        table_descs = {}
                        table_types = {}
                        for col in root.findall(".//column"):
                            col_name = col.get("name")
                            data_type = col.get("data_type", "")
                            desc_node = col.find("description")
                            if col_name:
                                if desc_node is not None:
                                    table_descs[col_name] = (desc_node.text or "").strip()
                                if data_type:
                                    table_types[col_name] = data_type
                        column_desc_map[table_name] = table_descs
                        column_type_map[table_name] = table_types
                    except ET.ParseError as e:
                        logger.warning(f"Failed parsing {desc_file}: {e}")
        
        # Build structured metadata - OPTIMIZED: only for first 5 tables
        tables = []
        for table_name in tables_to_load:
            if not table_name:
                continue
            
            # Get column info from description files
            table_types = column_type_map.get(table_name, {})
            table_descs = column_desc_map.get(table_name, {})
            
            columns = []
            for col_name, data_type in table_types.items():
                # Infer sdtype from data_type (same logic as regenerate_base_sap_schema.py)
                normalized = (data_type or "").lower()
                if any(x in normalized for x in ["int", "integer", "numeric", "decimal"]):
                    sdtype = "numerical"
                elif "bool" in normalized:
                    sdtype = "categorical"
                elif any(x in normalized for x in ["date", "time", "timestamp"]):
                    sdtype = "datetime"
                elif any(x in normalized for x in ["double", "float", "real"]):
                    sdtype = "numerical"
                elif any(x in normalized for x in ["char", "varchar", "text"]):
                    sdtype = "text"
                else:
                    sdtype = "text"
                
                columns.append({
                    "name": col_name,
                    "type": sdtype,
                    "description": table_descs.get(col_name)
                })
            
            tables.append({
                "name": table_name,
                "primary_key": "id",  # Default
                "introduction": intros_map.get(table_name),
                "columns": columns
            })
        
        return {"tables": tables}

    def _build_prompt(self, metadata: Dict) -> str:
        """Build LLM prompt with schema context."""
        tables = metadata["tables"]
        
        # Build schema description
        schema_text = "DATABASE SCHEMA:\n\n"
        for table in tables[:5]:  # Limit to 5 tables to avoid token limits
            schema_text += f"Table: {table['name']}\n"
            if table.get('introduction'):
                schema_text += f"Description: {table['introduction']}\n"
            schema_text += f"Primary Key: {table['primary_key']}\n"
            schema_text += "Columns:\n"
            
            for col in table['columns'][:10]:  # Limit columns too
                col_desc = f"  - {col['name']} ({col['type']})"
                if col.get('description'):
                    col_desc += f": {col['description']}"
                schema_text += col_desc + "\n"
            schema_text += "\n"
        
        prompt = f"""{schema_text}

Based on the above database schema, generate exactly 30 diverse analytical
questions that users might ask about THIS SPECIFIC DATASET.

CRITICAL REQUIREMENTS:
1. DO NOT use specific table names in questions (like "from TABLE_NAME" or "in TABLE_NAME")
2. Questions should be Specific and conceptual - let the AI agent figure out which tables to query
3. Include different question types:
   - Aggregations: "What's the total count?", "What's the average value?"
   - Trends: "Show me the trend over time", "What's changed recently?"
   - Comparisons: "Top 10 by value", "Which has the highest count?"
   - Distribution: "Breakdown by category", "Distribution across groups"
   - Filtering: "Show me records above threshold", "Which are critical?"
4. Make questions natural and conversational - write them as a business analyst would ask.
5. Each question should be 10-20 words: detailed enough to feel data-specific, concise enough for a UI chip.
6. Focus on business insights and actionable metrics, not technical database structure.
   Use the business meaning from column/table descriptions (e.g. if a column is described as
   "vendor name", say "vendor" - not the raw column name).
7. Questions must feel specific to THIS dataset - use real business entities, metrics, and dimensions
   visible in the schema descriptions above.
8. Avoid vague, generic questions that could apply to ANY dataset.
    BAD examples (too generic - could be any dataset):
        - "What's the total count?"
        - "Show me top 10"
        - "What's the trend over time?"
        - "Show bottom 10 entries"

    BAD examples (expose raw SQL names):
        - "Count of records in INV_POI table"
        - "Average STOCK_VALUE from INVENTORY table"

    Good examples (specific, natural, 10-20 words):
        - "Which vendors received the most purchase orders in the last 6 months?"
        - "What is the average invoice settlement time by payment terms?"
        - "Show total consumption value broken down by cost centre for this fiscal year"
        - "Which items have the highest stock-out frequency across all warehouses?"
        - "What are the top 10 customers by total pending invoice value?"

Format: Return ONLY the questions, one per line, numbered 1-30. No additional text or explanations.
"""
        return prompt

    async def _enhance_questions(
        self, questions: List[str], metadata: Dict
    ) -> List[str]:
        """Rewrite raw questions into richer, standalone enhanced questions.

        Uses a single batched LLM call with schema context to mirror the
        RouterAgent's enhanced_question normalization style.  Falls back to
        the original list if the enhancement call fails for any reason.
        """
        if not self.llm_client or not metadata or not metadata.get("tables"):
            return questions

        try:
            tables = metadata["tables"]
            schema_text = "DATABASE SCHEMA CONTEXT:\n\n"
            for table in tables[:5]:
                schema_text += f"Table: {table['name']}\n"
                if table.get('introduction'):
                    schema_text += f"Description: {table['introduction']}\n"
                schema_text += "Columns:\n"
                for col in table['columns'][:10]:
                    col_desc = f"  - {col['name']} ({col['type']})"
                    if col.get('description'):
                        col_desc += f": {col['description']}"
                    schema_text += col_desc + "\n"
                schema_text += "\n"

            numbered_questions = "\n".join(
                f"{i}. {q}" for i, q in enumerate(questions, 1)
            )

            system_prompt = (
                "You are a question-enhancement assistant. "
                "Your job is to rewrite short analytical questions into richer, "
                "standalone English sentences that a business analyst would ask.\n\n"
                "RULES:\n"
                "- Each output must be a SINGLE clean standalone question.\n"
                "- Use business terms from the schema context (e.g. 'vendor', "
                "'invoice', 'cost centre') — NEVER raw SQL table or column names.\n"
                "- Keep each question between 10 and 20 words.\n"
                "- Preserve the original intent — do NOT invent new questions.\n"
                "- Make the question specific to the dataset described below.\n"
                "- NEVER use meta-phrases like 'Based on the data' or "
                "'According to the schema'.\n"
            )

            user_message = f"""{schema_text}
Below are {len(questions)} raw questions. Rewrite each into a single, clean,
standalone question that feels specific to this dataset.

RAW QUESTIONS:
{numbered_questions}

Return ONLY the enhanced questions, one per line, numbered 1-{len(questions)}.
No additional text or explanations."""

            response = await self.llm_client.generate_completion(
                system_prompt=system_prompt,
                user_message=user_message,
                temperature=0.3,
                max_tokens=4000,
            )

            if response.get("error"):
                logger.warning(f"Enhancement LLM call returned error: {response['error']}")
                return questions

            response_text = response.get("content", "")
            if not response_text or not response_text.strip():
                logger.warning("Enhancement LLM returned empty response, keeping originals")
                return questions

            enhanced = self._parse_questions(response_text, len(questions))

            if len(enhanced) < len(questions):
                logger.warning(
                    f"Enhancement returned {len(enhanced)}/{len(questions)} questions, "
                    "keeping originals"
                )
                return questions

            logger.info(f"Successfully enhanced {len(enhanced)} questions")
            return enhanced

        except Exception as e:
            logger.warning(f"Question enhancement failed, keeping originals: {e}")
            return questions

    async def generate_questions(self, count: int = 10) -> List[str]:
        """
        Generate sample questions using LLM.
        
        Args:
            count: Number of questions to generate (default 10)
            
        Returns:
            List of question strings
        """
        questions = None
        metadata = None
        try:
            if not self.llm_client:
                logger.warning("LLMClient not initialized, using fallback questions")
                questions = self._get_fallback_questions(30)
            else:
                # Load metadata (now async)
                try:
                    metadata = await self._load_metadata()
                    
                    if not metadata["tables"]:
                        logger.warning(f"No tables found for client {self.client_id}, using fallback questions")
                        questions = self._get_fallback_questions(30)
                    else:
                        # Build prompt
                        prompt = self._build_prompt(metadata)
                        logger.info(f"Prompt length: {len(prompt)} characters")
                        logger.debug(f"Prompt (first 300 chars): {prompt[:300]}")
                        
                        # Call LLM using LLMClient (always generate 30 questions)
                        # Use empty system prompt since the prompt already contains all context
                        response = await self.llm_client.generate_completion(
                            system_prompt="",
                            user_message=prompt,
                            temperature=0.7,
                            max_tokens=4000  # Headroom for 30 detailed questions (~10-20 words each)
                        )
                        
                        # Check for errors
                        if response.get("error"):
                            logger.error(f"LLM returned error: {response['error']}")
                            logger.error("Falling back to generic questions")
                            questions = self._get_fallback_questions(30)
                        else:
                            response_text = response.get("content", "")
                            
                            # Log the raw response for debugging
                            logger.info(f"LLM Response length: {len(response_text)}")
                            logger.info(f"LLM Response (first 500 chars): '{response_text[:500]}'")
                            
                            # Check for empty response
                            if not response_text or len(response_text.strip()) == 0:
                                logger.error("LLM returned empty response!")
                                logger.error("Possible causes: 1) Token limit too low, 2) Model name invalid, 3) API issue")
                                logger.error("Falling back to generic questions")
                                questions = self._get_fallback_questions(30)
                            else:
                                # Parse questions from response (expecting 30)
                                questions = self._parse_questions(response_text, 30)
                                
                                if len(questions) < 30:
                                    logger.warning(f"Only generated {len(questions)} questions, expected 30")
                                    # Fill with fallbacks if needed
                                    while len(questions) < 30:
                                        questions.extend(self._get_fallback_questions(30 - len(questions)))
                                    questions = questions[:30]
                except FileNotFoundError as e:
                    logger.error(f"Metadata files not found for client {self.client_id}: {e}")
                    logger.info("Using fallback questions since metadata is not available")
                    questions = self._get_fallback_questions(30)
                except Exception as e:
                    logger.error(f"Error loading metadata or generating questions: {e}", exc_info=True)
                    questions = self._get_fallback_questions(30)
            
            # Enhance raw questions into richer standalone sentences
            if questions and self.llm_client and metadata:
                questions = await self._enhance_questions(questions, metadata)

            # Always save questions to XML (even if fallback)
            if questions:
                self._save_questions_to_xml(questions)
                logger.info(f"Saved {len(questions)} questions to XML for client {self.client_id}")
            
            # Return requested count
            return questions[:count] if questions else self._get_fallback_questions(count)
            
        except Exception as e:
            logger.error(f"Unexpected error generating questions: {e}", exc_info=True)
            # Even on unexpected errors, try to save fallback questions
            try:
                fallback_questions = self._get_fallback_questions(30)
                self._save_questions_to_xml(fallback_questions)
                logger.info(f"Saved fallback questions after error for client {self.client_id}")
                return fallback_questions[:count]
            except Exception as save_error:
                logger.error(f"Failed to save fallback questions: {save_error}", exc_info=True)
                return self._get_fallback_questions(count)
    
    def _parse_questions(self, response_text: str, expected_count: int) -> List[str]:
        """Parse questions from LLM response."""
        import re
        
        lines = response_text.strip().split('\n')
        questions = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Remove various numbering formats:
            # - "1.", "1)", "1-", "1:"
            # - Bullet points: "•", "-", "*"
            # - Markdown: "**1.**"
            line = re.sub(r'^[\*\•\-]\s*', '', line)  # Remove bullet points
            line = re.sub(r'^\d+[\.\)\-\:]\s*', '', line)  # Remove numbered lists
            line = re.sub(r'^\*\*\d+\.\*\*\s*', '', line)  # Remove markdown bold numbers
            line = line.strip()
            
            # Skip empty lines, headers, or very short lines
            if not line or len(line) < 10:
                continue
            
            # Skip common non-question lines
            skip_patterns = [
                r'^(questions?|here|based|format|note|example)',
                r'^(the following|below|above)',
            ]
            if any(re.match(pattern, line.lower()) for pattern in skip_patterns):
                continue
            
            # Clean up any remaining artifacts
            line = line.strip('*_`"')
            
            if line and len(line) > 10:  # Sanity check
                questions.append(line)
                logger.debug(f"Parsed question {len(questions)}: {line}")
        
        logger.info(f"Successfully parsed {len(questions)} questions from LLM response")
        return questions[:expected_count]
    
    def _get_fallback_questions(self, count: int = 10) -> List[str]:
        """Return generic fallback questions if generation fails."""
        fallback = [
            "What's the total count of records?",
            "Show me the top 10 by value",
            "What's the average across all entries?",
            "Show me the distribution by category",
            "Which items have the highest count?",
            "What's the trend over time?",
            "Show me a breakdown by group",
            "What's the total value?",
            "Which categories are most common?",
            "Show me recent activity summary",
            "What are the key metrics?",
            "Compare performance across segments",
            "Show monthly trends",
            "What's the growth rate?",
            "Identify top performers",
            "Show bottom 10 entries",
            "What's the distribution?",
            "Analyze by region",
            "Show year-over-year comparison",
            "What are the outliers?",
            "Calculate total revenue",
            "Show customer breakdown",
            "What's the conversion rate?",
            "Analyze seasonal patterns",
            "Show correlation analysis",
            "What's the retention rate?",
            "Compare quarter performance",
            "Show engagement metrics",
            "What's the churn rate?",
            "Analyze product performance"
        ]
        return fallback[:count]
    
    def _save_questions_to_xml(self, questions: List[str]) -> None:
        """Save questions to XML file."""
        try:
            # Ensure directory exists
            self.base_dir.mkdir(parents=True, exist_ok=True)
            
            # Create XML structure
            root = Element("suggested_questions")
            root.set("client_id", self.client_id)
            root.set("generated_at", datetime.now().isoformat())
            root.set("count", str(len(questions)))
            
            for idx, question in enumerate(questions, 1):
                question_elem = SubElement(root, "question")
                question_elem.set("id", str(idx))
                question_elem.text = question
            
            # Write to file with pretty formatting
            tree = ElementTree(root)
            indent(tree, space="  ")
            tree.write(
                self.questions_file,
                encoding="utf-8",
                xml_declaration=True
            )
            
            logger.info(f"Saved {len(questions)} questions to {self.questions_file}")
            
        except Exception as e:
            logger.error(f"Failed to save questions to XML: {e}", exc_info=True)
    
    def load_questions_from_xml(self) -> Optional[List[str]]:
        """Load questions from XML file if it exists."""
        try:
            if not self.questions_file.exists():
                logger.info(f"No cached questions file found at {self.questions_file}")
                return None
            
            tree = parse(self.questions_file)
            root = tree.getroot()
            
            questions = []
            for question_elem in root.findall("question"):
                question_text = question_elem.text
                if question_text:
                    questions.append(question_text.strip())
            
            logger.info(f"Loaded {len(questions)} questions from XML cache")
            return questions if questions else None
            
        except Exception as e:
            logger.error(f"Failed to load questions from XML: {e}", exc_info=True)
            return None
    
    def load_questions_with_ids(self) -> Optional[List[Dict[str, Any]]]:
        """Load questions from XML file with their IDs."""
        try:
            if not self.questions_file.exists():
                logger.info(f"No cached questions file found at {self.questions_file}")
                return None
            
            tree = parse(self.questions_file)
            root = tree.getroot()
            
            questions = []
            for question_elem in root.findall("question"):
                question_id = question_elem.get("id")
                question_text = question_elem.text
                if question_text and question_id:
                    try:
                        questions.append({
                            "id": int(question_id),
                            "text": question_text.strip()
                        })
                    except ValueError:
                        logger.warning(f"Invalid question ID: {question_id}")
            
            # Sort by ID to ensure correct order
            questions.sort(key=lambda x: x["id"])
            
            logger.info(f"Loaded {len(questions)} questions with IDs from XML cache")
            return questions if questions else None
            
        except Exception as e:
            logger.error(f"Failed to load questions with IDs from XML: {e}", exc_info=True)
            return None
    
    def get_xml_metadata(self) -> Optional[Dict[str, Any]]:
        """Get metadata from XML file (client_id, generated_at, count)."""
        try:
            if not self.questions_file.exists():
                return None
            
            tree = parse(self.questions_file)
            root = tree.getroot()
            
            return {
                "client_id": root.get("client_id"),
                "generated_at": root.get("generated_at"),
                "count": root.get("count")
            }
        except Exception as e:
            logger.error(f"Failed to get XML metadata: {e}", exc_info=True)
            return None