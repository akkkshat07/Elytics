# XML Prompt Files — Naming Convention

## Agent Prompt Files (`agents/`)

| Pattern | Purpose | Example |
|---------|---------|---------|
| `{agent}.xml` | **Standalone fallback.** Used when no client-specific file exists. | `planner.xml`, `python.xml` |
| `{agent}.base.xml` | **Merge template.** Base content used by Explorer Agent's `merge_agent_xml()` to create client-specific files. | `planner.base.xml`, `python.base.xml` |
| `{agent}.{ds_suffix}.xml` | **Data-source overlay.** Merged into `.base.xml` to produce the final client file. Suffix is one of: `sql_db`, `mongodb`, `parquet`. | `planner.sql_db.xml`, `python.parquet.xml` |
| `{agent}.{feature}.xml` | **Feature fragment.** Merged into the main prompt conditionally. | `intent_classifier.data_scientist_route.xml` |

## Merge Flow (Explorer Agent)

```
planner.base.xml + planner.{ds_suffix}.xml  →  clients/{id}/agents/planner.xml
python.base.xml  + python.{ds_suffix}.xml   →  clients/{id}/agents/python.xml
data_science_planner.base.xml + planner.{ds_suffix}.xml → clients/{id}/agents/data_science_planner.xml
```

## Data Source Suffixes (from `util/data_source.py`)

| Data Source Types | Suffix |
|---|---|
| MySQL, PostgreSQL, SAP Oracle, SAP HANA, SAP Sybase | `sql_db` |
| MongoDB | `mongodb` |
| Parquet, File Upload | `parquet` |
