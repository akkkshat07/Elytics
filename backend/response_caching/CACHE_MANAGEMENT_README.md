# Cache Management System - User Guide

## Overview

The Cache Management System provides a comprehensive interface to manage the response caching mechanism in CoreSight. It allows administrators to view, search, and delete cached questions. The system uses **Qdrant** as the vector database for semantic similarity matching, with **Gemini embeddings** (`text-embedding-004`) for question vectorization.

## Features

### Backend API Endpoints

All endpoints are protected with authentication and available at `/api/cache`:

1. **GET /api/cache/questions** - List cached questions with pagination

   - Query Parameters:
     - `page` (int, default: 1) - Page number
     - `page_size` (int, default: 50) - Items per page
     - `search` (string, optional) - Text search query
     - `id_min` (int, optional) - Minimum question ID
     - `id_max` (int, optional) - Maximum question ID

2. **GET /api/cache/search** - Search questions by text

   - Query Parameters:
     - `q` (string, required) - Search query

3. **POST /api/cache/preview-delete** - Preview deletion without executing

   - Request Body: `{ "question_ids": [1, 2, 3] }`
   - Returns: Questions to be deleted and impact summary

4. **DELETE /api/cache/questions** - Delete questions permanently

   - Request Body: `{ "question_ids": [1, 2, 3] }`
   - Deletes from CSV, Parquet, and Vector DB
   - Returns: Count of deletions from each source

5. **GET /api/cache/stats** - Get cache statistics

   - Returns: Total questions, counts from each source, file sizes

6. **POST /api/cache/rebuild** - Rebuild vector database
   - Regenerates vector database from current CSV data
   - Useful after bulk deletions or manual CSV edits

### Frontend Interface

Access the cache management page at: `/cache-management`

#### Features:

1. **Stats Dashboard**

   - View total questions across all storage
   - See individual counts for CSV, Parquet, and Vector DB
   - Monitor file sizes

2. **Search and Filter**

   - Real-time text search (debounced)
   - Filter by ID range (min/max)
   - Apply or clear filters

3. **Question List**

   - Paginated table view
   - Shows question ID, text, and available responses
   - Checkbox selection for bulk operations
   - Individual delete buttons

4. **Delete Operations**

   - **Delete by ID**: Enter comma-separated IDs (e.g., "1,2,3")
   - **Bulk Delete**: Select multiple questions using checkboxes
   - **Preview**: See what will be deleted before confirming
   - **Confirmation**: Two-step confirmation to prevent accidents

5. **Vector DB Rebuild**
   - Rebuild entire vector database from CSV
   - Useful after manual CSV modifications

## Usage Examples

### Deleting a Single Question

1. Navigate to `/cache-management`
2. Find the question in the table or use search
3. Click the "Delete" button for that question
4. Review the preview modal
5. Click "Proceed to Delete"
6. Confirm the final deletion

### Bulk Delete by Selection

1. Navigate to `/cache-management`
2. Use checkboxes to select multiple questions
3. Click "Delete Selected" button
4. Review the preview showing all selected questions
5. Confirm deletion

### Delete by Question IDs

1. Navigate to `/cache-management`
2. In the "Delete by Question ID(s)" section, enter IDs: `1,5,10,25`
3. Click "Preview Delete"
4. Review and confirm

### Searching Questions

1. Type in the search box (automatically filters after 500ms)
2. Use ID filters for range-based searches
3. Click "Apply Filters" to refresh with current filters

### Rebuilding Vector Database

1. Click "Rebuild Vector DB" button
2. Confirm the operation
3. Wait for completion (may take several minutes for large datasets)
4. Check stats to verify the rebuild

## Technical Details

### Vector Database: Qdrant

CoreSight uses **Qdrant** as the vector database for semantic caching. Each client has an isolated Qdrant collection (`responses_{client_id}`) to prevent cross-tenant data leakage.

**Embeddings:** Questions are embedded using Google Gemini `text-embedding-004` model via the `util/embedding_utils.py` module.

**Similarity Thresholds:**
- **>0.995** — Exact cache hit: cached response returned directly (skips the full AI pipeline)
- **>0.50** — Partial hit: cached response provided as reference guidance to agents
- **<0.50** — Cache miss: full pipeline runs, new response optionally cached

### Data Synchronization

When you delete questions, the system:

1. **Removes from Qdrant** - Deletes vectors and metadata from the client's collection
2. **Removes from MongoDB** - Updates the question metadata store

The deletion is atomic per storage — if any part fails, errors are reported but other parts may succeed. Always check the deletion summary.

### Stable ID Generation

Qdrant entries use SHA256 hashes of normalized questions:

```python
normalized = question.strip().lower()
id = f"q_{sha256(normalized)[:16]}"
```

This ensures consistent IDs across rebuilds.

### Authentication

All API endpoints require authentication using the existing auth system:

- Frontend automatically includes auth token
- Token is validated by `require_auth()` middleware
- 401 errors redirect to login

## Best Practices

1. **Preview Before Deleting**: Always use preview to verify what will be deleted
2. **Backup Data**: Consider backing up CSV files before bulk deletions
3. **Rebuild After Bulk Ops**: After deleting many questions, rebuild the vector DB to optimize performance
4. **Audit Trail**: All deletions are logged with user IDs for audit purposes
5. **Test in Development**: Test deletion operations in dev environment first

## Troubleshooting

### Questions Not Appearing

- Check if CSV/Parquet files exist at configured paths
- Verify vector database is initialized
- Check backend logs for errors

### Deletion Failed

- Verify you have write permissions to data files
- Check if files are locked by another process
- Review error messages in the response

### Vector DB Out of Sync

- Use "Rebuild Vector DB" to resynchronize
- This regenerates all Qdrant embeddings from the MongoDB question store

### Authentication Errors

- Ensure you're logged in
- Check if your auth token is valid
- Try logging out and back in

## API Usage Examples

### Using cURL

```bash
# List questions
curl -H "Authorization: Bearer YOUR_TOKEN" \
  "http://localhost:8022/api/cache/questions?page=1&page_size=10"

# Search questions
curl -H "Authorization: Bearer YOUR_TOKEN" \
  "http://localhost:8022/api/cache/search?q=kalol"

# Preview delete
curl -X POST -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"question_ids": [1, 2, 3]}' \
  http://localhost:8022/api/cache/preview-delete

# Delete questions
curl -X DELETE -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"question_ids": [1, 2, 3]}' \
  http://localhost:8022/api/cache/questions

# Get stats
curl -H "Authorization: Bearer YOUR_TOKEN" \
  http://localhost:8022/api/cache/stats

# Rebuild vector DB
curl -X POST -H "Authorization: Bearer YOUR_TOKEN" \
  http://localhost:8022/api/cache/rebuild
```

### Using Python

```python
import requests

API_BASE = "http://localhost:8022/api/cache"
TOKEN = "your_auth_token"
headers = {"Authorization": f"Bearer {TOKEN}"}

# List questions
response = requests.get(f"{API_BASE}/questions?page=1", headers=headers)
print(response.json())

# Delete questions
response = requests.delete(
    f"{API_BASE}/questions",
    headers=headers,
    json={"question_ids": [1, 2, 3]}
)
print(response.json())
```

## Security Considerations

- All endpoints require authentication
- Deletions are logged with user information
- Two-step confirmation prevents accidental deletions
- No direct file system access from frontend
- Input validation on all question IDs

## Maintenance

### Regular Tasks

1. **Monitor Cache Size**: Check stats regularly
2. **Clean Old Questions**: Remove outdated cached responses
3. **Rebuild Vector DB**: After bulk operations or CSV edits
4. **Backup Data**: Regular backups of CSV/Parquet files

### Performance Tips

- Use filters to narrow down question lists
- Rebuild vector DB after deleting >10% of questions
- Monitor file sizes in stats dashboard
- Consider archiving very old questions instead of deleting

## Support

For issues or questions:

1. Check backend logs at `coresight-backend/logs`
2. Review error messages in the UI
3. Contact your system administrator
4. Check the main documentation at `docs/`

---

**Version**: 2.0 (Qdrant + Gemini Embeddings)
**Last Updated**: February 2026
**Maintainer**: CoreSight Team
