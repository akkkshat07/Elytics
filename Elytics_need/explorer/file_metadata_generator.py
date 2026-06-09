"""
File-based Metadata Generator
Generates AI-powered metadata (table introductions, column descriptions) from Parquet files
without requiring a database connection.
"""

import logging
import pandas as pd
from pathlib import Path
from typing import Awaitable, Callable, List, Dict, Any, Optional
from datetime import datetime
# SECURITY: Use defusedxml for parsing (prevents XXE attacks)
from defusedxml.ElementTree import parse
# Import Element creation classes from standard library (safe for creation, not parsing)
# Safe: Only used for element creation, not parsing untrusted input. All parsing uses defusedxml.
from xml.etree.ElementTree import Element, SubElement, ElementTree as ET, indent  # nosemgrep: python.lang.security.use-defused-xml.use-defused-xml

from explorer.table_intro_agent import TableIntroductionAgent
from explorer.data_description_agent import DataDescriptionAgent
from explorer.models import TableMetadata, ColumnMetadata
from util.dataset_paths import resolve_xml_data_sources_dir

logger = logging.getLogger(__name__)


class FileMetadataGenerator:
    """
    Generates metadata for uploaded files using AI agents.
    Works with Parquet files instead of database connections.
    """
    
    def __init__(
        self,
        client_id: str,
        output_root: Path,
        db: Optional[Any] = None,
        dataset_id: Optional[str] = None,
    ):
        """
        Initialize the file metadata generator.
        
        Args:
            client_id: Client identifier
            output_root: Root directory for output files (xml_prompts/clients/{client_id})
            db: MongoDB database instance for loading client-specific LLM configs
            dataset_id: Optional dataset scope for data_sources XML layout
        """
        self.client_id = client_id
        self.output_root = Path(output_root)
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.data_sources_root = resolve_xml_data_sources_dir(client_id, dataset_id, for_write=True)
        self.data_sources_root.mkdir(parents=True, exist_ok=True)

        # Initialize agents
        self.table_intro_agent = TableIntroductionAgent(client_id=client_id, db=db)
        self.data_desc_agent = DataDescriptionAgent(
            client_id=client_id,
            output_dir=self.output_root,
            db=db,
            data_sources_base=self.data_sources_root,
        )
        
        logger.info(f"FileMetadataGenerator initialized for client {client_id}")

    @staticmethod
    def _build_table_metadata_from_df(
        df: pd.DataFrame,
        table_name: str,
        schema: str = "public",
        max_sample_rows: int = 100,
    ) -> TableMetadata:
        """
        Build TableMetadata from a DataFrame (avoids re-reading Parquet).
        Use when caller already has DataFrame from conversion loop.
        """
        columns = []
        for col in df.columns:
            dtype = df[col].dtype
            sql_type = "VARCHAR"
            if "int" in str(dtype):
                sql_type = "INTEGER"
            elif "float" in str(dtype):
                sql_type = "FLOAT"
            elif "bool" in str(dtype):
                sql_type = "BOOLEAN"
            elif "datetime" in str(dtype):
                sql_type = "TIMESTAMP"
            is_nullable = bool(df[col].isnull().any())
            columns.append(
                ColumnMetadata(name=col, data_type=sql_type, is_nullable=is_nullable)
            )
        sample_data = df.head(max_sample_rows).to_dict("records")
        return TableMetadata(
            schema=schema,
            name=table_name,
            columns=columns,
            sample_rows=sample_data,
        )
    
    async def generate_metadata(self, parquet_dir: Path, max_sample_rows: int = 100) -> Dict[str, Any]:
        """
        Generate metadata for all Parquet files in the directory.
        
        Args:
            parquet_dir: Directory containing Parquet files
            max_sample_rows: Maximum number of rows to sample from each table
            
        Returns:
            Dictionary with generation results and metadata
        """
        try:
            logger.info(f"Starting metadata generation for {parquet_dir}")
            
            # Find all Parquet files
            parquet_files = list(parquet_dir.glob("*.parquet"))
            if not parquet_files:
                raise ValueError(f"No Parquet files found in {parquet_dir}")
            
            logger.info(f"Found {len(parquet_files)} Parquet file(s)")
            
            # Load table metadata from Parquet files
            tables_metadata = []
            for parquet_file in parquet_files:
                metadata = self._load_table_metadata(parquet_file, max_sample_rows)
                tables_metadata.append(metadata)
            
            
            # Generate table introductions using AI in meta_information directory
            meta_info_dir = self.data_sources_root / "meta_information"
            meta_info_dir.mkdir(parents=True, exist_ok=True)
            intro_path = meta_info_dir / "table_introductions.xml"
            await self.table_intro_agent.generate(tables_metadata, intro_path)
            logger.info(f"Generated table introductions at {intro_path}")
            
            # Generate column descriptions using AI
            await self.data_desc_agent.generate(tables_metadata)
            logger.info("Generated column descriptions")
            
            logger.info(f"Metadata generation complete for {len(tables_metadata)} table(s)")
            
            return {
                "success": True,
                "tables_processed": len(tables_metadata),
                "output_directory": str(self.output_root),
                "files_generated": [
                    "table_introductions.xml",
                    "data_descriptions/*.xml"
                ]
            }
            
        except Exception as e:
            logger.error(f"Metadata generation failed: {str(e)}", exc_info=True)
            raise

    async def generate_metadata_from_tables(
        self,
        tables_metadata: List[TableMetadata],
        on_table_done: Optional[Callable[[], Awaitable[None]]] = None,
    ) -> Dict[str, Any]:
        """
        Generate metadata from pre-built TableMetadata (avoids re-reading Parquet).
        Use when caller already has tables_metadata from conversion loop.
        """
        try:
            logger.info(f"Starting metadata generation from {len(tables_metadata)} pre-built table(s)")
            if not tables_metadata:
                raise ValueError("No tables metadata provided")


            meta_info_dir = self.data_sources_root / "meta_information"
            meta_info_dir.mkdir(parents=True, exist_ok=True)
            intro_path = meta_info_dir / "table_introductions.xml"
            await self.table_intro_agent.generate(tables_metadata, intro_path)
            logger.info(f"Generated table introductions at {intro_path}")

            await self.data_desc_agent.generate(tables_metadata, on_table_done=on_table_done)
            logger.info("Generated column descriptions")

            logger.info(f"Metadata generation complete for {len(tables_metadata)} table(s)")

            return {
                "success": True,
                "tables_processed": len(tables_metadata),
                "output_directory": str(self.output_root),
                "files_generated": [
                    "table_introductions.xml",
                    "data_descriptions/*.xml",
                ],
            }

        except Exception as e:
            logger.error(f"Metadata generation failed: {str(e)}", exc_info=True)
            raise

    def _load_table_metadata(self, parquet_file: Path, max_sample_rows: int) -> TableMetadata:
        """
        Load metadata from a single Parquet file.
        
        Args:
            parquet_file: Path to Parquet file
            max_sample_rows: Maximum number of rows to sample
            
        Returns:
            TableMetadata object with schema and sample data
        """
        logger.info(f"Loading metadata from {parquet_file.name}")
        
        # Read Parquet file
        df = pd.read_parquet(parquet_file)
        table_name = parquet_file.stem
        
        # Extract column information
        columns = []
        for col in df.columns:
            dtype = df[col].dtype
            
            # Map pandas dtype to SQL type
            sql_type = 'VARCHAR'
            if 'int' in str(dtype):
                sql_type = 'INTEGER'
            elif 'float' in str(dtype):
                sql_type = 'FLOAT'
            elif 'bool' in str(dtype):
                sql_type = 'BOOLEAN'
            elif 'datetime' in str(dtype):
                sql_type = 'TIMESTAMP'
            
            # Check nullable
            is_nullable = bool(df[col].isnull().any())
            
            # Create ColumnMetadata
            col_meta = ColumnMetadata(
                name=col,
                data_type=sql_type,
                is_nullable=is_nullable
            )
            columns.append(col_meta)
        
        # Sample data (limit rows for AI processing)
        sample_df = df.head(max_sample_rows)
        
        # Convert sample to list of dictionaries
        sample_data = sample_df.to_dict('records')
        
        # Create TableMetadata
        table_meta = TableMetadata(
            schema='public',
            name=table_name,
            columns=columns,
            sample_rows=sample_data
        )
        
        logger.info(f"Loaded {table_name}: {len(columns)} columns, {len(df)} rows, {len(sample_data)} sample rows")
        
        return table_meta
    
    def _map_to_sdtype(self, sql_type: str) -> str:
        """Map SQL types to SDV sdtype."""
        type_lower = sql_type.lower()
        if 'int' in type_lower:
            return 'numerical'
        elif 'float' in type_lower or 'decimal' in type_lower or 'numeric' in type_lower:
            return 'numerical'
        elif 'bool' in type_lower:
            return 'boolean'
        elif 'date' in type_lower or 'time' in type_lower:
            return 'datetime'
        else:
            return 'text'


