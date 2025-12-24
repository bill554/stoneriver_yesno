"""
StoneRiver Prospect Processor - Cost Optimized Version
=======================================================

This is an updated version of processor.py that uses a tiered model approach
to reduce Claude API costs by 60-80%.

Changes from original:
- Tier 1 (Haiku): Quick disqualification filter for obvious non-fits
- Tier 2 (Sonnet): Detailed evaluation for prospects that pass Tier 1
- Batch processing option for Tier 1
- Cost tracking and reporting

Deploy: Replace the existing processor.py in your repo with this file.
"""

import os
import sys
import json
import time
import sqlite3
import logging
import argparse
from datetime import datetime, date
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from enum import Enum

import anthropic
from notion_client import Client as NotionClient

# ============================================================
# CONFIGURATION
# ============================================================

# Environment variables
NOTION_API_KEY = os.environ.get("NOTION_API_KEY")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
NOTION_DATABASE_ID = os.environ.get("NOTION_DATABASE_ID", "1f4e5363-46f0-80e8-a91c-c01aisnotreal")

# Processing settings
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "10"))
SLEEP_BETWEEN_PROSPECTS = float(os.environ.get("SLEEP_BETWEEN_PROSPECTS", "2.0"))
SLEEP_BETWEEN_BATCHES = float(os.environ.get("SLEEP_BETWEEN_BATCHES", "30.0"))
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")

# Cost optimization settings
USE_TIERED_MODELS = os.environ.get("USE_TIERED_MODELS", "true").lower() == "true"
TIER1_MODEL = "claude-haiku-4-5-20251001"
TIER2_MODEL = "claude-sonnet-4-5-20250929"

# Checkpoint database path
CHECKPOINT_DB = os.environ.get("CHECKPOINT_DB", "/data/processor_checkpoint.db")

# Setup logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Initialize clients
notion = NotionClient(auth=NOTION_API_KEY) if NOTION_API_KEY else None
anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY) if ANTHROPIC_API_KEY else None

# ============================================================
# STONERIVER QUALIFICATION CRITERIA (cached, not sent every call)
# ============================================================

STONERIVER_CRITERIA = """
StoneRiver Investment Fund III Criteria:

STRATEGY: Multifamily real estate in Southeastern US (Sunbelt)
TICKET SIZE: $5-25M investments
MINIMUM COMMITMENT: $500,000

QUALIFIED ALLOCATOR TYPES:
- Family offices (single or multi-family)
- RIAs with $500M+ AUM and alternatives capability
- Multi-family offices
- Endowments & foundations
- Public/corporate pensions with real estate allocations
- OCIOs with real asset mandates

AUTOMATIC DISQUALIFIERS:
- Retail-focused firms without alternatives capability
- AUM below $200M (cannot write $5M+ checks)
- No real estate or real assets allocation history
- Insurance/benefits-focused without investment mandates
- Broker-dealers without discretionary management
- Accounting/tax/legal firms without investment arm
"""

# ============================================================
# TIER 1: HAIKU QUICK FILTER
# ============================================================

TIER1_SYSTEM_PROMPT = """You are a fast prospect screener for StoneRiver, a multifamily real estate fund.

DISQUALIFY IMMEDIATELY if ANY of these are clearly true from the firm info:
- Insurance brokerage or benefits administration firm
- Retail investment platform (robo-advisor, retail brokerage)
- Broker-dealer without asset management capability
- Accounting, tax, or legal firm without investment arm
- Clearly stated AUM under $200M
- Wealth management for retail/mass affluent only

PASS TO DETAILED REVIEW if:
- Could potentially be a qualified institutional allocator
- Shows signs of: family office, RIA with alternatives, endowment, pension, foundation
- Unclear from available information
- AUM appears to be $200M+ or is not stated

Respond ONLY with valid JSON:
{"action": "DISQUALIFY" or "PASS", "confidence": "HIGH" or "MEDIUM", "reason": "10 words max"}"""

def tier1_quick_filter(firm_name: str, firm_info: str) -> Dict[str, Any]:
    """
    Tier 1: Haiku-based quick disqualification
    Cost: ~$0.001 per prospect
    """
    try:
        response = anthropic_client.messages.create(
            model=TIER1_MODEL,
            max_tokens=100,
            system=TIER1_SYSTEM_PROMPT,
            messages=[{
                "role": "user",
                "content": f"Firm: {firm_name}\nInfo: {firm_info[:2000]}"
            }]
        )
        
        result_text = response.content[0].text.strip()
        # Handle potential markdown code blocks
        if result_text.startswith("```"):
            result_text = result_text.split("```")[1]
            if result_text.startswith("json"):
                result_text = result_text[4:]
        
        result = json.loads(result_text)
        
        cost = (response.usage.input_tokens * 0.25 + response.usage.output_tokens * 1.25) / 1_000_000
        
        return {
            "action": result.get("action", "PASS"),
            "confidence": result.get("confidence", "MEDIUM"),
            "reason": result.get("reason", ""),
            "cost": cost,
            "tokens": response.usage.input_tokens + response.usage.output_tokens
        }
    except json.JSONDecodeError:
        logger.warning(f"Tier 1 JSON parse failed for {firm_name}, passing to Tier 2")
        return {"action": "PASS", "confidence": "LOW", "reason": "Parse error", "cost": 0.001, "tokens": 0}
    except Exception as e:
        logger.error(f"Tier 1 error for {firm_name}: {e}")
        return {"action": "PASS", "confidence": "LOW", "reason": str(e)[:50], "cost": 0, "tokens": 0}

# ============================================================
# TIER 2: SONNET DETAILED EVALUATION
# ============================================================

TIER2_SYSTEM_PROMPT = """You are evaluating prospects for StoneRiver Investment Fund III.

""" + STONERIVER_CRITERIA + """

TIER DEFINITIONS:
- Tier 1: Strong fit - institutional allocator with clear RE/alternatives allocation capability
- Tier 2: Potential fit - meets basic criteria but needs more research to confirm
- De-Prioritize: Does not meet criteria (too small, wrong type, no alternatives)
- Needs Verification: Insufficient information to make determination

Evaluate based on:
1. Firm type (must be qualified allocator type)
2. AUM scale (can they write $5-25M checks? Need $200M+ AUM)
3. Real estate/alternatives signals (do they invest in private RE or alternatives?)

Respond ONLY with valid JSON matching this exact schema:
{
    "tier": "Tier 1" or "Tier 2" or "De-Prioritize" or "Needs Verification",
    "reason": "1-2 sentence explanation with specific evidence",
    "firm_category": "Family Office" or "RIA" or "Multi-Family Office" or "Endowment" or "Foundation" or "Pension" or "OCIO" or "Broker-Dealer" or "Insurance" or "Other",
    "real_estate_signal": "None" or "Weak" or "Strong",
    "real_asset_signal": "None" or "Weak" or "Strong",
    "allocator_sophistication": "Retail" or "Semi-Institutional" or "Institutional",
    "next_action": "Skip" or "Deep Research" or "Find Decision Makers" or "Outreach",
    "stage1_survives": true or false
}"""

def tier2_detailed_evaluation(firm_name: str, firm_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Tier 2: Sonnet-based detailed evaluation
    Cost: ~$0.01-0.03 per prospect
    """
    # Build firm info string from available data
    firm_info_parts = [f"Firm Name: {firm_name}"]
    
    if firm_data.get("website"):
        firm_info_parts.append(f"Website: {firm_data['website']}")
    if firm_data.get("firm_type"):
        firm_info_parts.append(f"Firm Type: {firm_data['firm_type']}")
    if firm_data.get("aum"):
        firm_info_parts.append(f"AUM: {firm_data['aum']}")
    if firm_data.get("city") or firm_data.get("state"):
        location = ", ".join(filter(None, [firm_data.get("city"), firm_data.get("state"), firm_data.get("country")]))
        firm_info_parts.append(f"Location: {location}")
    
    firm_info = "\n".join(firm_info_parts)
    
    try:
        response = anthropic_client.messages.create(
            model=TIER2_MODEL,
            max_tokens=400,
            system=TIER2_SYSTEM_PROMPT,
            messages=[{
                "role": "user",
                "content": f"Evaluate this prospect:\n\n{firm_info}"
            }]
        )
        
        result_text = response.content[0].text.strip()
        # Handle potential markdown code blocks
        if result_text.startswith("```"):
            result_text = result_text.split("```")[1]
            if result_text.startswith("json"):
                result_text = result_text[4:]
        
        result = json.loads(result_text)
        
        cost = (response.usage.input_tokens * 3 + response.usage.output_tokens * 15) / 1_000_000
        
        return {
            "success": True,
            "tier": result.get("tier", "Needs Verification"),
            "reason": result.get("reason", "No reason provided"),
            "firm_category": result.get("firm_category", "Other"),
            "real_estate_signal": result.get("real_estate_signal", "None"),
            "real_asset_signal": result.get("real_asset_signal", "None"),
            "allocator_sophistication": result.get("allocator_sophistication", "Retail"),
            "next_action": result.get("next_action", "Skip"),
            "stage1_survives": result.get("stage1_survives", False),
            "cost": cost,
            "tokens": response.usage.input_tokens + response.usage.output_tokens
        }
    except json.JSONDecodeError as e:
        logger.error(f"Tier 2 JSON parse failed for {firm_name}: {e}")
        return {
            "success": False,
            "tier": "Needs Verification",
            "reason": "Evaluation failed - JSON parse error",
            "firm_category": "Other",
            "real_estate_signal": "None",
            "real_asset_signal": "None",
            "allocator_sophistication": "Retail",
            "next_action": "Deep Research",
            "stage1_survives": False,
            "cost": 0.02,
            "tokens": 0,
            "error": str(e)
        }
    except Exception as e:
        logger.error(f"Tier 2 error for {firm_name}: {e}")
        return {
            "success": False,
            "tier": "Needs Verification",
            "reason": f"Evaluation error: {str(e)[:100]}",
            "firm_category": "Other",
            "real_estate_signal": "None",
            "real_asset_signal": "None",
            "allocator_sophistication": "Retail",
            "next_action": "Deep Research",
            "stage1_survives": False,
            "cost": 0,
            "tokens": 0,
            "error": str(e)
        }

# ============================================================
# SINGLE MODEL FALLBACK (original behavior)
# ============================================================

SINGLE_MODEL_PROMPT = TIER2_SYSTEM_PROMPT  # Same as Tier 2

def single_model_evaluation(firm_name: str, firm_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Original single-model evaluation (for comparison or fallback)
    """
    return tier2_detailed_evaluation(firm_name, firm_data)

# ============================================================
# MAIN EVALUATION FUNCTION
# ============================================================

def evaluate_prospect(firm_name: str, firm_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main evaluation function that routes to tiered or single model approach.
    
    Returns evaluation result with cost tracking.
    """
    total_cost = 0.0
    total_tokens = 0
    tiers_used = []
    
    if not USE_TIERED_MODELS:
        # Original single-model behavior
        logger.info(f"[Single Model] Evaluating {firm_name}")
        result = single_model_evaluation(firm_name, firm_data)
        result["tiers_used"] = ["Sonnet"]
        return result
    
    # TIERED APPROACH
    
    # Build summary for Tier 1
    firm_summary = f"{firm_data.get('firm_type', '')} | {firm_data.get('aum', '')} | {firm_data.get('website', '')}"
    
    # Tier 1: Quick filter
    logger.info(f"[Tier 1] Quick screening {firm_name}")
    tier1_result = tier1_quick_filter(firm_name, firm_summary)
    total_cost += tier1_result.get("cost", 0)
    total_tokens += tier1_result.get("tokens", 0)
    tiers_used.append("Haiku")
    
    # If Tier 1 disqualifies with high confidence, skip Tier 2
    if tier1_result["action"] == "DISQUALIFY" and tier1_result["confidence"] == "HIGH":
        logger.info(f"[Tier 1] DISQUALIFIED {firm_name}: {tier1_result['reason']}")
        return {
            "success": True,
            "tier": "De-Prioritize",
            "reason": f"Quick screen: {tier1_result['reason']}",
            "firm_category": "Other",
            "real_estate_signal": "None",
            "real_asset_signal": "None",
            "allocator_sophistication": "Retail",
            "next_action": "Skip",
            "stage1_survives": False,
            "cost": total_cost,
            "tokens": total_tokens,
            "tiers_used": tiers_used
        }
    
    # Tier 2: Detailed evaluation
    logger.info(f"[Tier 2] Detailed evaluation for {firm_name}")
    tier2_result = tier2_detailed_evaluation(firm_name, firm_data)
    total_cost += tier2_result.get("cost", 0)
    total_tokens += tier2_result.get("tokens", 0)
    tiers_used.append("Sonnet")
    
    tier2_result["cost"] = total_cost
    tier2_result["tokens"] = total_tokens
    tier2_result["tiers_used"] = tiers_used
    
    return tier2_result

# ============================================================
# NOTION INTEGRATION
# ============================================================

def get_unprocessed_prospects(limit: int = 100) -> List[Dict[str, Any]]:
    """Fetch prospects from Notion that need processing."""
    if not notion:
        logger.error("Notion client not initialized")
        return []
    
    try:
        response = notion.databases.query(
            database_id=NOTION_DATABASE_ID,
            filter={
                "and": [
                    {"property": "FIRM NAME", "title": {"is_not_empty": True}},
                    {
                        "or": [
                            {"property": "Status", "select": {"is_empty": True}},
                            {"property": "Status", "select": {"equals": "Queued"}},
                            {"property": "Status", "select": {"equals": "Enriched"}}
                        ]
                    }
                ]
            },
            page_size=min(limit, 100)
        )
        
        prospects = []
        for page in response.get("results", []):
            props = page.get("properties", {})
            
            # Extract firm name
            firm_name_prop = props.get("FIRM NAME", {})
            firm_name = ""
            if firm_name_prop.get("title"):
                firm_name = firm_name_prop["title"][0]["plain_text"] if firm_name_prop["title"] else ""
            
            # Extract other properties
            website = props.get("WEBSITE", {}).get("url", "")
            firm_type = props.get("FIRM TYPE", {}).get("select", {})
            firm_type = firm_type.get("name", "") if firm_type else ""
            aum = props.get("AUM", {}).get("select", {})
            aum = aum.get("name", "") if aum else ""
            city = props.get("CITY", {}).get("rich_text", [])
            city = city[0]["plain_text"] if city else ""
            state = props.get("STATE", {}).get("rich_text", [])
            state = state[0]["plain_text"] if state else ""
            country = props.get("COUNTRY", {}).get("rich_text", [])
            country = country[0]["plain_text"] if country else ""
            
            prospects.append({
                "page_id": page["id"],
                "firm_name": firm_name,
                "website": website,
                "firm_type": firm_type,
                "aum": aum,
                "city": city,
                "state": state,
                "country": country
            })
        
        return prospects
    
    except Exception as e:
        logger.error(f"Error fetching prospects from Notion: {e}")
        return []

def update_notion_prospect(page_id: str, result: Dict[str, Any]) -> bool:
    """Update a prospect in Notion with evaluation results."""
    if not notion:
        logger.error("Notion client not initialized")
        return False
    
    try:
        # Map tier to select option
        tier_mapping = {
            "Tier 1": "Tier 1",
            "Tier 2": "Tier 2",
            "De-Prioritize": "De-Prioritize",
            "Needs Verification": "Needs Verification"
        }
        
        properties = {
            "Status": {"select": {"name": "Rated"}},
            "StoneRiver Tier": {"select": {"name": tier_mapping.get(result["tier"], "Needs Verification")}},
            "Reason (1-2 lines)": {"rich_text": [{"text": {"content": result["reason"][:2000]}}]},
            "Firm Category (Expanded)": {"select": {"name": result.get("firm_category", "Other")}},
            "Real Estate Signal": {"select": {"name": result.get("real_estate_signal", "None")}},
            "Real Asset Signal": {"select": {"name": result.get("real_asset_signal", "None")}},
            "Allocator Sophistication": {"select": {"name": result.get("allocator_sophistication", "Retail")}},
            "Next Action": {"select": {"name": result.get("next_action", "Skip")}},
            "Stage 1 Survives": {"checkbox": result.get("stage1_survives", False)},
            "Last Processed": {"date": {"start": date.today().isoformat()}}
        }
        
        notion.pages.update(page_id=page_id, properties=properties)
        return True
    
    except Exception as e:
        logger.error(f"Error updating Notion page {page_id}: {e}")
        # Try to write error to Notion
        try:
            notion.pages.update(
                page_id=page_id,
                properties={
                    "Status": {"select": {"name": "Error"}},
                    "Error": {"rich_text": [{"text": {"content": str(e)[:2000]}}]}
                }
            )
        except:
            pass
        return False

# ============================================================
# CHECKPOINT DATABASE
# ============================================================

def init_checkpoint_db():
    """Initialize SQLite checkpoint database."""
    os.makedirs(os.path.dirname(CHECKPOINT_DB), exist_ok=True)
    conn = sqlite3.connect(CHECKPOINT_DB)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS processed (
            page_id TEXT PRIMARY KEY,
            firm_name TEXT,
            tier TEXT,
            cost REAL,
            tokens INTEGER,
            tiers_used TEXT,
            processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS cost_tracking (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT,
            total_cost REAL,
            total_tokens INTEGER,
            prospects_processed INTEGER,
            tier1_filtered INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

def is_processed(page_id: str) -> bool:
    """Check if a prospect has already been processed."""
    try:
        conn = sqlite3.connect(CHECKPOINT_DB)
        cursor = conn.cursor()
        cursor.execute("SELECT 1 FROM processed WHERE page_id = ?", (page_id,))
        result = cursor.fetchone()
        conn.close()
        return result is not None
    except:
        return False

def mark_processed(page_id: str, firm_name: str, result: Dict[str, Any]):
    """Mark a prospect as processed."""
    try:
        conn = sqlite3.connect(CHECKPOINT_DB)
        cursor = conn.cursor()
        cursor.execute(
            """INSERT OR REPLACE INTO processed 
               (page_id, firm_name, tier, cost, tokens, tiers_used, processed_at) 
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                page_id,
                firm_name,
                result.get("tier", "Unknown"),
                result.get("cost", 0),
                result.get("tokens", 0),
                ",".join(result.get("tiers_used", [])),
                datetime.now().isoformat()
            )
        )
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"Error marking processed: {e}")

def get_daily_stats() -> Dict[str, Any]:
    """Get cost statistics for today."""
    try:
        conn = sqlite3.connect(CHECKPOINT_DB)
        cursor = conn.cursor()
        today = date.today().isoformat()
        cursor.execute(
            """SELECT COUNT(*), SUM(cost), SUM(tokens) 
               FROM processed 
               WHERE DATE(processed_at) = ?""",
            (today,)
        )
        count, total_cost, total_tokens = cursor.fetchone()
        
        # Count Tier 1 filtered
        cursor.execute(
            """SELECT COUNT(*) FROM processed 
               WHERE DATE(processed_at) = ? AND tiers_used = 'Haiku'""",
            (today,)
        )
        tier1_only = cursor.fetchone()[0]
        
        conn.close()
        return {
            "date": today,
            "prospects_processed": count or 0,
            "total_cost": total_cost or 0,
            "total_tokens": total_tokens or 0,
            "tier1_filtered": tier1_only or 0,
            "avg_cost_per_prospect": (total_cost or 0) / (count or 1)
        }
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        return {}

# ============================================================
# MAIN PROCESSING LOOP
# ============================================================

def process_batch(prospects: List[Dict[str, Any]], dry_run: bool = False) -> Dict[str, Any]:
    """Process a batch of prospects."""
    results = {
        "processed": 0,
        "success": 0,
        "failed": 0,
        "tier1_filtered": 0,
        "total_cost": 0.0,
        "total_tokens": 0
    }
    
    for prospect in prospects:
        page_id = prospect["page_id"]
        firm_name = prospect["firm_name"]
        
        # Skip if already processed
        if is_processed(page_id):
            logger.info(f"Skipping already processed: {firm_name}")
            continue
        
        logger.info(f"Processing: {firm_name}")
        
        # Evaluate prospect
        result = evaluate_prospect(firm_name, prospect)
        
        results["processed"] += 1
        results["total_cost"] += result.get("cost", 0)
        results["total_tokens"] += result.get("tokens", 0)
        
        if result.get("tiers_used") == ["Haiku"]:
            results["tier1_filtered"] += 1
        
        logger.info(f"Result: {result['tier']} - {result['reason'][:100]}")
        logger.info(f"Cost: ${result.get('cost', 0):.4f} | Tiers: {result.get('tiers_used', [])}")
        
        if not dry_run:
            # Update Notion
            if update_notion_prospect(page_id, result):
                results["success"] += 1
                mark_processed(page_id, firm_name, result)
            else:
                results["failed"] += 1
        else:
            results["success"] += 1
            logger.info("[DRY RUN] Would update Notion")
        
        # Rate limiting
        time.sleep(SLEEP_BETWEEN_PROSPECTS)
    
    return results

def run_processor(max_prospects: int = None, dry_run: bool = False, continuous: bool = False, interval: int = 1800):
    """Main processor entry point."""
    logger.info("=" * 60)
    logger.info("STONERIVER PROSPECT PROCESSOR - COST OPTIMIZED")
    logger.info("=" * 60)
    logger.info(f"Tiered models: {'ENABLED' if USE_TIERED_MODELS else 'DISABLED'}")
    logger.info(f"Tier 1 model: {TIER1_MODEL}")
    logger.info(f"Tier 2 model: {TIER2_MODEL}")
    logger.info(f"Dry run: {dry_run}")
    logger.info(f"Continuous: {continuous}")
    logger.info("=" * 60)
    
    # Initialize checkpoint database
    init_checkpoint_db()
    
    # Show previous stats
    stats = get_daily_stats()
    if stats.get("prospects_processed", 0) > 0:
        logger.info(f"Today's stats so far: {stats}")
    
    total_results = {
        "processed": 0,
        "success": 0,
        "failed": 0,
        "tier1_filtered": 0,
        "total_cost": 0.0
    }
    
    while True:
        # Fetch unprocessed prospects
        limit = max_prospects - total_results["processed"] if max_prospects else BATCH_SIZE
        if max_prospects and limit <= 0:
            break
        
        prospects = get_unprocessed_prospects(limit=min(limit, BATCH_SIZE))
        
        if not prospects:
            if continuous:
                logger.info(f"No prospects to process. Sleeping {interval}s...")
                time.sleep(interval)
                continue
            else:
                logger.info("No more prospects to process.")
                break
        
        logger.info(f"Found {len(prospects)} prospects to process")
        
        # Process batch
        batch_results = process_batch(prospects, dry_run=dry_run)
        
        # Aggregate results
        for key in total_results:
            if key in batch_results:
                total_results[key] += batch_results[key]
        
        logger.info(f"Batch complete: {batch_results}")
        
        # Check if we've hit the max
        if max_prospects and total_results["processed"] >= max_prospects:
            break
        
        # Sleep between batches
        if continuous or len(prospects) == BATCH_SIZE:
            logger.info(f"Sleeping {SLEEP_BETWEEN_BATCHES}s between batches...")
            time.sleep(SLEEP_BETWEEN_BATCHES)
    
    # Final summary
    logger.info("=" * 60)
    logger.info("PROCESSING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Total processed: {total_results['processed']}")
    logger.info(f"Successful: {total_results['success']}")
    logger.info(f"Failed: {total_results['failed']}")
    logger.info(f"Tier 1 filtered: {total_results['tier1_filtered']}")
    logger.info(f"Total cost: ${total_results['total_cost']:.4f}")
    if total_results['processed'] > 0:
        avg_cost = total_results['total_cost'] / total_results['processed']
        logger.info(f"Avg cost per prospect: ${avg_cost:.4f}")
        
        # Estimate savings
        sonnet_only_cost = total_results['processed'] * 0.025  # ~$0.025 per prospect with Sonnet only
        savings = sonnet_only_cost - total_results['total_cost']
        savings_pct = (savings / sonnet_only_cost * 100) if sonnet_only_cost > 0 else 0
        logger.info(f"Estimated savings vs Sonnet-only: ${savings:.4f} ({savings_pct:.1f}%)")
    
    return total_results

# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="StoneRiver Prospect Processor (Cost Optimized)")
    parser.add_argument("--max", type=int, help="Maximum prospects to process")
    parser.add_argument("--dry-run", action="store_true", help="Don't write to Notion")
    parser.add_argument("--continuous", action="store_true", help="Run continuously")
    parser.add_argument("--interval", type=int, default=1800, help="Seconds between checks in continuous mode")
    parser.add_argument("--no-tiered", action="store_true", help="Disable tiered models (use Sonnet for all)")
    parser.add_argument("--stats", action="store_true", help="Show daily stats and exit")
    
    args = parser.parse_args()
    
    if args.no_tiered:
        USE_TIERED_MODELS = False
    
    if args.stats:
        init_checkpoint_db()
        stats = get_daily_stats()
        print(json.dumps(stats, indent=2))
        sys.exit(0)
    
    # Validate environment
    if not NOTION_API_KEY:
        logger.error("NOTION_API_KEY not set")
        sys.exit(1)
    if not ANTHROPIC_API_KEY:
        logger.error("ANTHROPIC_API_KEY not set")
        sys.exit(1)
    
    run_processor(
        max_prospects=args.max,
        dry_run=args.dry_run,
        continuous=args.continuous,
        interval=args.interval
    )
