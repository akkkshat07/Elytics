#  Client Configuration

## Overview
This directory contains all -specific prompts, data schemas, and business rules for the CoreSight multi-tenant platform.

## Client Information
- **Client ID**: `account-bd7e`
- **Client Name**: 

## Operating  Facilities


## Directory Structure

```
clients/account-bd7e/
├── agents/                          # -specific agent prompts
│   ├── planner.xml                 # Planning logic with  tables and business rules
│   ├── python.xml                  # Python code generation for  data
│   └── business.xml                # Business insights for  context
│
├── domain_knowledge/               #  business domain knowledge
│   └── terminology.xml            # -specific terms (plants, products, processes)
│
├── data_sources/                   #  data definitions
│   ├── meta_information/
│   │   └── table_introductions.xml #  table descriptions
│   └── data_descriptions/        # Detailed column descriptions
│
└── schemas/                        # Output format schemas
    └── response_schema.json       # Expected response format
```

## Multi-Tenant Architecture

 inherits generic prompts from `xml_prompts/base/` and overrides with client-specific content:
1. Base prompts provide generic business logic
2. -specific prompts add client customizations
3. Sample values and examples are -specific only
4. No cross-contamination with other clients

## Created
- Date: 2026-06-01
- Purpose: Multi-tenant client onboarding
