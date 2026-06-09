"""
Explorer schema limits: apply subscription plan table/column limits to schema
and produce explorer_limits metadata for API responses.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple


def apply_table_column_limits(
    schema: List[Dict[str, Any]],
    table_limit: Optional[int],
    column_limit: Optional[int],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Apply plan table and column limits to a normalized schema. Tables are
    ordered deterministically (schema, table_name); columns preserve ordinal order.

    Args:
        schema: List of tables. Each table dict must have "table_name" or "name",
                and "columns" (list of column dicts with at least "name").
                Optional "schema" key used for sorting.
        table_limit: Max tables to load (None = unlimited). Must be non-negative int if provided.
        column_limit: Max columns per table to load (None = unlimited). Must be non-negative int if provided.

    Returns:
        (trimmed_schema, limit_metadata) where limit_metadata is the
        explorer_limits object for API responses.
    """
    # Validate and normalize limits
    if table_limit is not None:
        if not isinstance(table_limit, int) or table_limit < 0:
            raise ValueError(f"table_limit must be a non-negative integer, got {table_limit!r}")
    if column_limit is not None:
        if not isinstance(column_limit, int) or column_limit < 0:
            raise ValueError(f"column_limit must be a non-negative integer, got {column_limit!r}")
    
    if not schema:
        return [], _build_limit_metadata(
            schema=[],
            trimmed=[],
            table_limit=table_limit,
            column_limit=column_limit,
        )

    # Normalize: ensure table_name and columns list
    normalized: List[Dict[str, Any]] = []
    for t in schema:
        if not isinstance(t, dict):
            continue  # Skip invalid entries
        name = t.get("table_name") or t.get("name") or ""
        schema_name = t.get("schema", "")
        cols = t.get("columns")
        # Ensure columns is a list
        if not isinstance(cols, list):
            cols = []
        normalized.append({
            "table_name": name,
            "schema": schema_name,
            "columns": cols,  # Already a list, no need for list() conversion
            "_original": t,  # keep rest of keys for later reconstruction if needed
        })

    # Deterministic order: by (schema, table_name)
    normalized.sort(key=lambda x: (x.get("schema", ""), x.get("table_name", "")))

    # Apply table limit (first N tables)
    if table_limit is not None:
        if table_limit <= 0:
            trimmed_tables = []
        else:
            trimmed_tables = normalized[:table_limit]
    else:
        trimmed_tables = normalized  # Already a list, no need for list() conversion

    # Apply column limit per table (first N columns by ordinal order)
    trimmed_schema: List[Dict[str, Any]] = []
    for t in trimmed_tables:
        cols = t["columns"]
        if column_limit is not None:
            if column_limit <= 0:
                cols_trimmed = []
            else:
                cols_trimmed = cols[:column_limit]
        else:
            cols_trimmed = cols  # Already a list, no need for list() conversion
        # Preserve original table structure for file/DB consumers
        out = dict(t["_original"]) if isinstance(t.get("_original"), dict) else {}
        out["table_name"] = t["table_name"]
        out["name"] = t["table_name"]
        if "schema" in t:
            out["schema"] = t["schema"]
        out["columns"] = cols_trimmed
        if "column_count" in out:
            out["column_count"] = len(cols_trimmed)
        trimmed_schema.append(out)

    limit_metadata = _build_limit_metadata(
        schema=normalized,
        trimmed=trimmed_schema,
        table_limit=table_limit,
        column_limit=column_limit,
    )
    return trimmed_schema, limit_metadata


def _build_limit_metadata(
    schema: List[Dict[str, Any]],
    trimmed: List[Dict[str, Any]],
    table_limit: Optional[int],
    column_limit: Optional[int],
) -> Dict[str, Any]:
    """
    Build metadata about applied limits for API responses.
    
    Args:
        schema: Original normalized schema (before limits applied)
        trimmed: Schema after limits applied
        table_limit: Table limit that was applied (None = unlimited)
        column_limit: Column limit that was applied (None = unlimited)
    
    Returns:
        Dictionary with limit metadata for explorer_limits API response
    """
    total_tables_available = len(schema)
    total_tables_loaded = len(trimmed)
    
    # Calculate max columns per table in trimmed schema (safely handle missing columns)
    columns_per_table_loaded = max(
        (len(t.get("columns") or []) for t in trimmed),
        default=0,
    )

    # Check if table limit was reached
    is_table_limit_reached = (
        table_limit is not None 
        and table_limit > 0 
        and total_tables_available > table_limit
    )

    # Check if column limit was reached (any table in original schema exceeds limit)
    is_column_limit_reached = False
    if column_limit is not None and column_limit > 0 and schema:
        for t in schema:
            cols = t.get("columns")
            if isinstance(cols, list):
                ncols = len(cols)
                if ncols > column_limit:
                    is_column_limit_reached = True
                    break

    # Determine if upgrade is required
    upgrade_required = is_table_limit_reached or is_column_limit_reached
    # Zero limits: user cannot explore
    if table_limit == 0 or column_limit == 0:
        upgrade_required = True

    return {
        "table_limit": table_limit,
        "column_limit": column_limit,
        "total_tables_available": total_tables_available,
        "total_tables_loaded": total_tables_loaded,
        "columns_per_table_loaded": columns_per_table_loaded,
        "is_table_limit_reached": is_table_limit_reached,
        "is_column_limit_reached": is_column_limit_reached,
        "upgrade_required": upgrade_required,
    }
