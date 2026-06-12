# coreops Client Configuration

## Overview
This directory contains all coreops-specific prompts, data schemas, and business rules for the CoreSight multi-tenant platform.

## Client Information
- **Client ID**: `coreops`
- **Client Name**: coreops

## Operating  Facilities


## Directory Structure

```
clients/coreops/
├── agents/                          # coreops-specific agent prompts
│   ├── planner.xml                 # Planning logic with coreops tables and business rules
│   ├── python.xml                  # Python code generation for coreops data
│   └── business.xml                # Business insights for coreops context
│
├── domain_knowledge/               # coreops business domain knowledge
│   └── terminology.xml            # coreops-specific terms (plants, products, processes)
│
├── data_sources/                   # coreops data definitions
│   ├── meta_information/
│   │   └── table_introductions.xml # coreops table descriptions
│   └── data_descriptions/        # Detailed column descriptions
│
└── schemas/                        # Output format schemas
    └── response_schema.json       # Expected response format
```

## Multi-Tenant Architecture

coreops inherits generic prompts from `xml_prompts/base/` and overrides with client-specific content:
1. Base prompts provide generic business logic
2. coreops-specific prompts add client customizations
3. Sample values and examples are coreops-specific only
4. No cross-contamination with other clients

## Created
- Date: 2026-03-09
- Purpose: Multi-tenant client onboarding
