# None Client Configuration

## Overview
This directory contains all None-specific prompts, data schemas, and business rules for the CoreSight multi-tenant platform.

## Client Information
- **Client ID**: `account-81f9`
- **Client Name**: None

## Operating  Facilities


## Directory Structure

```
clients/account-81f9/
├── agents/                          # None-specific agent prompts
│   ├── planner.xml                 # Planning logic with None tables and business rules
│   ├── python.xml                  # Python code generation for None data
│   └── business.xml                # Business insights for None context
│
├── domain_knowledge/               # None business domain knowledge
│   └── terminology.xml            # None-specific terms (plants, products, processes)
│
├── data_sources/                   # None data definitions
│   ├── meta_information/
│   │   └── table_introductions.xml # None table descriptions
│   └── data_descriptions/        # Detailed column descriptions
│
└── schemas/                        # Output format schemas
    └── response_schema.json       # Expected response format
```

## Multi-Tenant Architecture

None inherits generic prompts from `xml_prompts/base/` and overrides with client-specific content:
1. Base prompts provide generic business logic
2. None-specific prompts add client customizations
3. Sample values and examples are None-specific only
4. No cross-contamination with other clients

## Created
- Date: 2026-06-08
- Purpose: Multi-tenant client onboarding
