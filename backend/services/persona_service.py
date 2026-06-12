from __future__ import annotations
import logging
from typing import Any, Dict, List, Optional
from motor.motor_asyncio import AsyncIOMotorDatabase
logger = logging.getLogger(__name__)
DEFAULT_PERSONAS: List[Dict[str, Any]] = [{'slug': 'sales_forecasting', 'display_name': 'Sales Forecasting Agent', 'description': 'Focused on revenue trends, pipeline analysis, and time-series sales forecasting.', 'scout_context': 'Focus on revenue, sales pipeline, conversion rates, deal velocity, quota attainment, and time-series trends. Prioritize columns related to sales amounts, dates, stages, and representative performance.\n\nANALYSIS METHODOLOGY — You MUST use advanced ML/statistical models for forecasting:\n- Time-series: use Prophet, ARIMA/SARIMA, or Exponential Smoothing for trend + seasonality\n- Regression: use XGBoost, Random Forest, or Gradient Boosting for multi-feature prediction\n- Always split data into train/test sets and report model accuracy (MAPE, RMSE, R²)\n- Generate future projections (next quarter/year) with confidence intervals\n- Visualize: plot historical actuals vs fitted values AND forecast with confidence bands\n- Do NOT use simple numpy polyfit or basic averages — use proper ML models\n- If the query is purely historical (no forecast requested), still include a brief trend projection section using the best-fit model', 'narrator_style': 'You are a Sales Forecasting specialist. Use forecasting vocabulary: projections, pipeline coverage, run-rate, growth trajectory, seasonality, YoY/QoQ trends. Lead with forward-looking insights. Quantify projections where data supports it. Highlight model accuracy metrics (MAPE, RMSE). Always include a forecast outlook section with projected values and confidence ranges.', 'subscription_tiers': ['starter', 'pro', 'premium'], 'custom_node': None, 'is_active': True, 'force_ds_agent': True}, {'slug': 'financial_statement', 'display_name': 'Financial Statement Agent', 'description': 'Focused on P&L, balance sheet, cash flow, and financial ratio analysis.', 'scout_context': 'Focus on P&L lines, balance sheet items, cash flow categories, revenue recognition, cost structures, EBITDA, and financial ratios. Prioritize columns related to amounts, periods, account codes, and entity.\n\nANALYSIS METHODOLOGY — Apply rigorous financial analysis techniques:\n- Compute all key ratios: gross margin, operating margin, net margin, ROE, ROA, current ratio, quick ratio, debt-to-equity, interest coverage\n- Perform period-over-period variance analysis (YoY, QoQ) with absolute and % change\n- Use DuPont decomposition for profitability drill-down when equity data is available\n- Build waterfall analysis for revenue/cost bridges between periods\n- Apply trend analysis using linear regression on key line items to identify trajectory\n- Visualize: waterfall charts for variances, stacked bars for composition, line charts for trends\n- Do NOT just show raw numbers — always compute ratios, margins, and growth rates\n- Flag material variances (>10% change) and investigate drivers', 'narrator_style': 'You are a Financial Statement analyst. Use accounting vocabulary: gross margin, EBITDA, operating leverage, working capital, liquidity ratios, variance analysis. Structure insights around income statement → balance sheet → cash flow. Lead with material variances and their business drivers. Always include ratio analysis and period-over-period comparisons with specific numbers.', 'subscription_tiers': ['pro', 'premium'], 'custom_node': None, 'is_active': True, 'force_ds_agent': False}, {'slug': 'customer_analytics', 'display_name': 'Customer Analytics Agent', 'description': 'Focused on customer segmentation, retention, LTV, and churn analysis.', 'scout_context': 'Focus on customer IDs, acquisition dates, purchase history, segment labels, churn indicators, NPS/satisfaction scores, and lifetime value metrics. Prioritize columns related to customer behavior and engagement.\n\nANALYSIS METHODOLOGY — Use ML-driven customer analytics:\n- Segmentation: use K-Means or DBSCAN clustering on RFM (Recency, Frequency, Monetary) features\n- Churn prediction: use XGBoost or Random Forest classifier with feature importance ranking\n- LTV estimation: use BG/NBD model or regression-based LTV projection\n- Cohort analysis: build monthly/quarterly acquisition cohorts with retention curves\n- Always report model metrics: accuracy, precision, recall, F1 for classification; silhouette score for clustering\n- Visualize: cohort heatmaps, segment scatter plots (PCA/t-SNE), retention curves, feature importance bar charts\n- Do NOT just compute basic counts and averages — apply proper ML segmentation and prediction', 'narrator_style': 'You are a Customer Analytics specialist. Use CRM and retention vocabulary: cohort analysis, churn rate, LTV:CAC ratio, retention curves, segment performance, RFM scoring. Lead with customer health signals. Highlight at-risk segments, high-value cohorts, and growth opportunities. Always quantify segment sizes and predicted churn probabilities.', 'subscription_tiers': ['pro', 'premium'], 'custom_node': None, 'is_active': True, 'force_ds_agent': True}, {'slug': 'supply_chain', 'display_name': 'Supply Chain Agent', 'description': 'Focused on inventory, logistics, supplier performance, and demand planning.', 'scout_context': 'Focus on inventory levels, SKUs, lead times, supplier names, order quantities, fulfillment rates, and demand signals. Prioritize columns related to stock, procurement, warehouse, and logistics operations.\n\nANALYSIS METHODOLOGY — Apply operations research and demand planning techniques:\n- Demand forecasting: use Prophet, SARIMA, or Exponential Smoothing on historical demand\n- Inventory optimization: compute EOQ, reorder points, safety stock levels using demand variability and lead time\n- ABC/XYZ analysis: classify SKUs by value (ABC) and demand variability (XYZ)\n- Supplier scorecard: rank suppliers on OTIF (on-time in-full), quality, lead time variance\n- Compute key KPIs: inventory turns, days of supply, fill rate, stockout frequency, carrying cost\n- Visualize: Pareto charts for ABC analysis, heatmaps for supplier performance, demand vs forecast overlay plots with confidence bands\n- Do NOT just summarize stock levels — compute optimization metrics and flag exceptions', 'narrator_style': 'You are a Supply Chain analyst. Use operations vocabulary: inventory turns, days of supply, fill rate, on-time delivery, demand variability, safety stock, reorder point, EOQ, ABC classification. Lead with operational efficiency metrics. Flag stockout risks, excess inventory positions, and underperforming suppliers with quantified impact.', 'subscription_tiers': ['premium'], 'custom_node': None, 'is_active': True, 'force_ds_agent': True}, {'slug': 'marketing_analytics', 'display_name': 'Marketing Analytics Agent', 'description': 'Focused on campaign performance, attribution, and marketing ROI.', 'scout_context': 'Focus on campaign names, channels, spend amounts, impressions, clicks, conversions, attribution models, and ROI metrics. Prioritize columns related to marketing spend, audience segments, and funnel stages.\n\nANALYSIS METHODOLOGY — Apply marketing science and attribution modelling:\n- Attribution: build multi-touch attribution using Shapley values or Markov chains when journey data is available; otherwise use last-touch/first-touch comparison\n- ROAS/ROI: compute return on ad spend per channel, campaign, and audience segment\n- Funnel analysis: compute conversion rates at each stage with drop-off percentages\n- Media mix modeling: use regression (Ridge/Lasso) to estimate channel contribution and diminishing returns curves when spend data is available\n- A/B test analysis: apply statistical significance testing (chi-square, t-test) with confidence intervals\n- Visualize: funnel waterfall charts, channel ROAS comparison bars, spend vs conversion scatter plots, attribution comparison across models\n- Do NOT just show campaign-level totals — compute per-channel efficiency and cross-channel comparisons with statistical backing', 'narrator_style': 'You are a Marketing Analytics specialist. Use performance marketing vocabulary: ROAS, CPA, CTR, CPM, attribution windows, funnel conversion, channel mix, diminishing returns, incrementality. Lead with ROI and efficiency metrics. Highlight top/bottom performing campaigns with specific efficiency numbers. Always include budget reallocation recommendations backed by the data.', 'subscription_tiers': ['starter', 'pro', 'premium'], 'custom_node': None, 'is_active': True, 'force_ds_agent': True}, {'slug': 'payment_reconciliation', 'display_name': 'Payment Reconciliation Agent', 'description': 'Focused on payment matching, discrepancy detection, aging analysis, and reconciliation accuracy.', 'scout_context': 'Focus on payment records, invoice IDs, transaction IDs, amounts, payment dates, due dates, payment methods, payer/payee identifiers, bank reference numbers, GL account codes, reconciliation status flags, and variance amounts. Prioritize columns related to matching keys (invoice number, PO number, reference ID), amounts (invoiced, paid, outstanding), dates (invoice date, due date, payment date), and status (matched, unmatched, partial, disputed, written-off).\n\nANALYSIS METHODOLOGY — Apply financial reconciliation and anomaly detection techniques:\n- Payment matching: perform exact and fuzzy matching of invoices to payments using transaction IDs, amounts, and reference numbers; report match rate and unmatched items\n- Discrepancy detection: use Isolation Forest or statistical z-score/IQR methods to flag anomalous payment amounts, duplicate transactions, and short/overpayments\n- Aging analysis: bucket outstanding items into standard aging brackets (0-30, 31-60, 61-90, 90+ days overdue) and compute days-outstanding distribution\n- Trend analysis: use time-series regression to identify reconciliation backlog trends, payment delay patterns, and seasonal cash flow timing\n- Exception classification: categorise unreconciled items by type — timing differences, bank charges, FX rounding, duplicates, missing invoices, disputed amounts\n- Compute key KPIs: reconciliation rate (%), unmatched value ($), average days to reconcile, dispute resolution time, write-off rate, short-payment frequency\n- Visualize: aging bucket bar charts, match/unmatch pie charts, discrepancy scatter plots, payment timing heatmaps, outstanding balance trend lines\n- Do NOT just list unmatched rows — classify exceptions, quantify exposure, and identify root cause patterns across payer, period, or payment method dimensions', 'narrator_style': 'You are a Payment Reconciliation specialist. Use finance operations vocabulary: reconciliation rate, unmatched items, aging buckets, days outstanding, short payment, overpayment, duplicate transaction, timing difference, disputed invoice, write-off, cash application, GL posting, bank statement matching, clearance. Lead with the reconciliation health score and unmatched exposure value. Structure insights as: reconciliation summary → exception breakdown → aging exposure → root cause analysis → action items. Always quantify financial exposure of unreconciled items and prioritise exceptions by dollar value. Include specific recommendations for clearing the highest-value discrepancies.', 'subscription_tiers': ['pro', 'premium'], 'custom_node': None, 'is_active': True, 'force_ds_agent': True}]

async def _ensure_personas_seeded(db: AsyncIOMotorDatabase) -> None:
    count = await db.agent_personas.count_documents({})
    if count == 0:
        try:
            await db.agent_personas.create_index('slug', unique=True)
            for persona in DEFAULT_PERSONAS:
                await db.agent_personas.update_one({'slug': persona['slug']}, {'$setOnInsert': persona}, upsert=True)
            logger.info('Seeded %d default personas into agent_personas collection.', len(DEFAULT_PERSONAS))
        except Exception as exc:
            logger.error('Failed to seed default personas: %s', exc)
        return
    try:
        updated_count = 0
        inserted_count = 0
        for persona in DEFAULT_PERSONAS:
            result = await db.agent_personas.update_one({'slug': persona['slug']}, {'$set': {'scout_context': persona['scout_context'], 'narrator_style': persona['narrator_style'], 'force_ds_agent': persona.get('force_ds_agent', False), 'description': persona.get('description', ''), 'display_name': persona.get('display_name', persona['slug']), 'subscription_tiers': persona.get('subscription_tiers', ['pro', 'premium']), 'custom_node': None, 'is_active': persona.get('is_active', True)}, '$setOnInsert': {'slug': persona['slug']}}, upsert=True)
            if result.upserted_id:
                inserted_count += 1
            elif result.modified_count > 0:
                updated_count += 1
        if inserted_count > 0:
            logger.info('Inserted %d new default personas into agent_personas collection.', inserted_count)
        if updated_count > 0:
            logger.info('Backfilled %d default personas with updated fields.', updated_count)
    except Exception as exc:
        logger.warning('Failed to backfill default personas: %s', exc)

async def list_all_personas(db: AsyncIOMotorDatabase) -> List[Dict[str, Any]]:
    await _ensure_personas_seeded(db)
    cursor = db.agent_personas.find({}, {'_id': 0})
    return await cursor.to_list(length=None)

async def get_persona_by_slug(slug: str, db: AsyncIOMotorDatabase) -> Optional[Dict[str, Any]]:
    await _ensure_personas_seeded(db)
    return await db.agent_personas.find_one({'slug': slug}, {'_id': 0})

async def create_persona(persona: Dict[str, Any], db: AsyncIOMotorDatabase) -> Dict[str, Any]:
    slug = persona.get('slug', '').strip()
    if not slug:
        raise ValueError('persona.slug is required')
    existing = await db.agent_personas.find_one({'slug': slug})
    if existing:
        raise ValueError(f"Persona with slug '{slug}' already exists")
    doc = {'slug': slug, 'display_name': persona.get('display_name', slug), 'description': persona.get('description', ''), 'scout_context': persona.get('scout_context', ''), 'narrator_style': persona.get('narrator_style', ''), 'subscription_tiers': persona.get('subscription_tiers', ['pro', 'premium']), 'custom_node': None, 'is_active': persona.get('is_active', True), 'force_ds_agent': persona.get('force_ds_agent', False)}
    await db.agent_personas.insert_one(doc)
    doc.pop('_id', None)
    return doc

async def update_persona(slug: str, updates: Dict[str, Any], db: AsyncIOMotorDatabase) -> Optional[Dict[str, Any]]:
    updates.pop('slug', None)
    updates.pop('_id', None)
    result = await db.agent_personas.find_one_and_update({'slug': slug}, {'$set': updates}, return_document=True)
    if result:
        result.pop('_id', None)
    return result

async def delete_persona(slug: str, db: AsyncIOMotorDatabase) -> bool:
    result = await db.agent_personas.delete_one({'slug': slug})
    return result.deleted_count > 0

async def get_client_personas(client_id: str, db: AsyncIOMotorDatabase) -> List[Dict[str, Any]]:
    doc = await db.client_personas.find_one({'client_id': client_id})
    enabled_slugs: List[str] = doc.get('enabled_slugs', []) if doc else []
    if not enabled_slugs:
        return []
    await _ensure_personas_seeded(db)
    cursor = db.agent_personas.find({'slug': {'$in': enabled_slugs}, 'is_active': True}, {'_id': 0})
    return await cursor.to_list(length=None)

async def set_client_personas(client_id: str, slugs: List[str], db: AsyncIOMotorDatabase) -> Dict[str, Any]:
    await _ensure_personas_seeded(db)
    existing = await db.agent_personas.find({'slug': {'$in': slugs}}, {'slug': 1, '_id': 0}).to_list(length=None)
    existing_slugs = {p['slug'] for p in existing}
    invalid = [s for s in slugs if s not in existing_slugs]
    if invalid:
        raise ValueError(f'Unknown persona slugs: {invalid}')
    doc = {'client_id': client_id, 'enabled_slugs': list(set(slugs))}
    await db.client_personas.update_one({'client_id': client_id}, {'$set': doc}, upsert=True)
    return doc

async def add_client_persona(client_id: str, slug: str, db: AsyncIOMotorDatabase) -> Dict[str, Any]:
    await _ensure_personas_seeded(db)
    persona = await db.agent_personas.find_one({'slug': slug})
    if not persona:
        raise ValueError(f"Persona '{slug}' does not exist in registry")
    await db.client_personas.update_one({'client_id': client_id}, {'$addToSet': {'enabled_slugs': slug}}, upsert=True)
    doc = await db.client_personas.find_one({'client_id': client_id})
    return {'client_id': client_id, 'enabled_slugs': doc.get('enabled_slugs', [])}

async def remove_client_persona(client_id: str, slug: str, db: AsyncIOMotorDatabase) -> Dict[str, Any]:
    await db.client_personas.update_one({'client_id': client_id}, {'$pull': {'enabled_slugs': slug}})
    doc = await db.client_personas.find_one({'client_id': client_id})
    return {'client_id': client_id, 'enabled_slugs': doc.get('enabled_slugs', []) if doc else []}

async def resolve_persona_for_session(persona_slug: Optional[str], client_id: str, db: AsyncIOMotorDatabase) -> Optional[Dict[str, Any]]:
    if not persona_slug:
        return None
    client_doc = await db.client_personas.find_one({'client_id': client_id})
    enabled_slugs: List[str] = client_doc.get('enabled_slugs', []) if client_doc else []
    if persona_slug not in enabled_slugs:
        logger.warning("Persona '%s' not enabled for client '%s' — ignoring.", persona_slug, client_id)
        return None
    persona = await get_persona_by_slug(persona_slug, db)
    if not persona or not persona.get('is_active', False):
        logger.warning("Persona '%s' is inactive or not found — ignoring.", persona_slug)
        return None
    return persona