#!/usr/bin/env python3
"""
StoneRiver Prospect Processor
=============================
Automated prospect qualification system for StoneRiver Fund III.

This script:
1. Queries Notion for unprocessed prospects (Status != "Rated", has name + website)
2. Sends each to Claude API for evaluation against StoneRiver criteria
3. Updates Notion with qualification results
4. Handles errors gracefully with retries and checkpointing

Designed for Railway deployment with resilience for thousands of records.
"""

import os
import sys
import json
import time
import logging
import sqlite3
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, asdict
import hashlib

import anthropic
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ============================================================================
# CONFIGURATION
# ============================================================================

# Environment variables (set in Railway)
NOTION_API_KEY = os.environ.get("NOTION_API_KEY")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
DATABASE_ID = os.environ.get("NOTION_DATABASE_ID", "2cec16a0-949c-80c6-b291-000bfd569cff")

# Processing configuration - slower for thorough research
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "5"))  # Prospects per batch
SLEEP_BETWEEN_PROSPECTS = float(os.environ.get("SLEEP_BETWEEN_PROSPECTS", "10.0"))  # Seconds - longer for web research
SLEEP_BETWEEN_BATCHES = float(os.environ.get("SLEEP_BETWEEN_BATCHES", "60.0"))  # Seconds
MAX_RETRIES = int(os.environ.get("MAX_RETRIES", "3"))
CHECKPOINT_FILE = os.environ.get("CHECKPOINT_FILE", "/tmp/processor_checkpoint.db")

# Logging
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# ============================================================================
# STONERIVER EVALUATION PROMPT
# ============================================================================

EVALUATION_SYSTEM_PROMPT = """You are an investment analyst evaluating whether prospect firms are good fits for StoneRiver Investment Fund III. You have web search capabilities and should use them to research each prospect thoroughly.

**StoneRiver Strategy:** 
- Multifamily real estate in the Southeastern U.S.
- $5-25M ticket size per investment
- Target fund size: $200-300M
- Target investors: Institutional and sophisticated allocators

**Qualified Allocator Types:**
- Family offices (single or multi-family) with real asset allocation
- RIAs with $500M+ AUM and alternatives/private placement capability
- Multi-family offices (MFOs) serving UHNW clients
- Endowments & foundations with real estate allocations
- Public/corporate pensions with real estate allocations
- OCIOs with real asset mandates

**Disqualifiers (automatic NO):**
- Retail-focused RIAs without alternatives/private placement capability
- AUM too small to write $5M+ checks (generally <$200M AUM)
- No real estate or real assets allocation history/capability
- Insurance/benefits-focused firms without investment mandates
- Broker-dealers without discretionary management
- International firms without US investment appetite
- Firms focused exclusively on public markets/ETFs/mutual funds
- Tax preparation or accounting firms without investment management
- Personal financial planning firms focused on retirement planning only

**Research Instructions:**
1. ALWAYS search for information about the firm using their name and website
2. Look for: AUM, investment philosophy, asset classes they invest in, client types
3. Check if they mention: real estate, alternatives, private investments, direct investments
4. Look for team bios that mention real estate or alternatives experience
5. Check for any news about their investments or fund allocations

**Evaluation Criteria (with evidence required):**
1. Firm Type: What type of allocator are they? Cite evidence.
2. AUM Scale: What is their AUM? Can they write $5-25M checks?
3. Real Estate/Alternatives: Do they invest in private RE? What evidence exists?
4. Geographic Fit: US-based or investing in US real estate?
5. Client Base: Institutional, UHNW, or retail?

**Your Response Format (JSON only, no markdown):**
{
  "qualified": true/false,
  "tier": "Tier 1" | "Tier 2" | "De-Prioritize" | "Needs Verification",
  "reason": "2-3 sentence explanation citing SPECIFIC evidence found (AUM figures, mentions of real estate, client types, etc.)",
  "firm_category": "Family Office - Institutional" | "Family Office - Lifestyle" | "RIA - Institutional" | "RIA - Retail" | "Endowment/Foundation" | "Pension" | "OCIO" | "Bank / Trust" | "Insurance" | "Sovereign" | "Other",
  "real_estate_signal": "None" | "Weak" | "Strong",
  "real_asset_signal": "None" | "Weak" | "Strong", 
  "allocator_sophistication": "Retail" | "Semi" | "Institutional",
  "next_action": "Skip" | "Deep Research" | "Find Decision Makers" | "Outreach",
  "confidence": "Low" | "Medium" | "High",
  "aum_found": "string or null - exact AUM if found",
  "evidence_summary": "Brief summary of key findings from research"
}

**Tier Definitions:**
- Tier 1: Strong fit - institutional allocator with VERIFIED RE allocation history, confirmed appropriate AUM ($500M+), clear alternatives capability
- Tier 2: Potential fit - meets some criteria, shows promise but needs verification (e.g., AUM unclear but appears substantial, mentions alternatives but no specifics)
- De-Prioritize: Clearly does not meet criteria (retail focus, too small, no alternatives)
- Needs Verification: Could not find enough information to evaluate properly

**Confidence Definitions:**
- High: Found AUM, clear investment philosophy, specific evidence of RE/alternatives
- Medium: Found some information but missing key details (e.g., no AUM but clear institutional focus)
- Low: Limited information available, evaluation based on inferences

Return ONLY valid JSON, no explanation text before or after."""

EVALUATION_USER_TEMPLATE = """Research and evaluate this prospect for StoneRiver Fund III:

**Firm Name:** {firm_name}
**Website:** {website}
**Firm Type (from database):** {firm_type}
**Location:** {city}, {state}, {country}
**AUM (from database):** {aum}

**Instructions:**
1. Search for "{firm_name}" to find information about this firm
2. If the website is provided, search for content from {website}
3. Look for: AUM, investment strategy, asset classes, client types, team backgrounds
4. Determine if they invest in real estate, alternatives, or private investments
5. Evaluate against StoneRiver's criteria

Provide your evaluation as JSON with specific evidence from your research."""

# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class Prospect:
    """Represents a prospect from Notion"""
    page_id: str
    firm_name: str
    website: Optional[str]
    firm_type: Optional[str]
    city: Optional[str]
    state: Optional[str]
    country: Optional[str]
    aum: Optional[str]
    status: Optional[str]
    
@dataclass
class EvaluationResult:
    """Result of Claude's evaluation"""
    qualified: bool
    tier: str
    reason: str
    firm_category: str
    real_estate_signal: str
    real_asset_signal: str
    allocator_sophistication: str
    next_action: str
    confidence: str
    error: Optional[str] = None

# ============================================================================
# CHECKPOINT DATABASE (for resilience)
# ============================================================================

class CheckpointDB:
    """SQLite-based checkpoint system for tracking processed prospects"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS processed (
                    page_id TEXT PRIMARY KEY,
                    firm_name TEXT,
                    processed_at TEXT,
                    success INTEGER,
                    error TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS stats (
                    key TEXT PRIMARY KEY,
                    value TEXT
                )
            """)
            conn.commit()
    
    def is_processed(self, page_id: str) -> bool:
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute(
                "SELECT 1 FROM processed WHERE page_id = ?", (page_id,)
            )
            return cur.fetchone() is not None
    
    def mark_processed(self, page_id: str, firm_name: str, success: bool, error: str = None):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO processed (page_id, firm_name, processed_at, success, error)
                VALUES (?, ?, ?, ?, ?)
            """, (page_id, firm_name, datetime.now(timezone.utc).isoformat(), int(success), error))
            conn.commit()
    
    def get_stats(self) -> Dict[str, int]:
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute("SELECT COUNT(*) FROM processed WHERE success = 1")
            success = cur.fetchone()[0]
            cur = conn.execute("SELECT COUNT(*) FROM processed WHERE success = 0")
            failed = cur.fetchone()[0]
            return {"success": success, "failed": failed, "total": success + failed}

# ============================================================================
# NOTION CLIENT
# ============================================================================

class NotionClient:
    """Notion API client with retry logic"""
    
    BASE_URL = "https://api.notion.com/v1"
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.session = self._create_session()
    
    def _create_session(self) -> requests.Session:
        session = requests.Session()
        retry = Retry(
            total=MAX_RETRIES,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "POST", "PATCH"]
        )
        adapter = HTTPAdapter(max_retries=retry)
        session.mount("https://", adapter)
        session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Notion-Version": "2022-06-28",
            "Content-Type": "application/json"
        })
        return session
    
    def query_database(self, database_id: str, filter_obj: Dict = None, 
                       start_cursor: str = None, page_size: int = 100) -> Dict:
        """Query a Notion database with optional filter"""
        url = f"{self.BASE_URL}/databases/{database_id}/query"
        payload = {"page_size": page_size}
        if filter_obj:
            payload["filter"] = filter_obj
        if start_cursor:
            payload["start_cursor"] = start_cursor
        
        response = self.session.post(url, json=payload)
        response.raise_for_status()
        return response.json()
    
    def update_page(self, page_id: str, properties: Dict) -> Dict:
        """Update a Notion page's properties"""
        url = f"{self.BASE_URL}/pages/{page_id}"
        payload = {"properties": properties}
        
        response = self.session.patch(url, json=payload)
        response.raise_for_status()
        return response.json()
    
    def get_unprocessed_prospects(self, database_id: str) -> List[Prospect]:
        """Fetch all prospects that need processing"""
        prospects = []
        start_cursor = None
        
        # Filter: Has firm name, has website, status is not "Rated" or "Error"
        filter_obj = {
            "and": [
                {"property": "FIRM NAME", "title": {"is_not_empty": True}},
                {"property": "WEBSITE", "url": {"is_not_empty": True}},
                {
                    "or": [
                        {"property": "Status", "select": {"is_empty": True}},
                        {"property": "Status", "select": {"equals": "Queued"}},
                        {"property": "Status", "select": {"equals": "Enriched"}}
                    ]
                }
            ]
        }
        
        while True:
            result = self.query_database(database_id, filter_obj, start_cursor)
            
            for page in result.get("results", []):
                prospect = self._parse_prospect(page)
                if prospect:
                    prospects.append(prospect)
            
            if not result.get("has_more"):
                break
            start_cursor = result.get("next_cursor")
            time.sleep(0.5)  # Rate limiting
        
        return prospects
    
    def _parse_prospect(self, page: Dict) -> Optional[Prospect]:
        """Parse a Notion page into a Prospect object"""
        try:
            props = page.get("properties", {})
            
            # Extract title (FIRM NAME)
            firm_name_prop = props.get("FIRM NAME", {})
            firm_name = ""
            if firm_name_prop.get("title"):
                firm_name = "".join(
                    t.get("plain_text", "") for t in firm_name_prop["title"]
                )
            
            # Extract URL (WEBSITE)
            website = props.get("WEBSITE", {}).get("url")
            
            # Extract select values
            firm_type = props.get("FIRM TYPE", {}).get("select", {})
            firm_type = firm_type.get("name") if firm_type else None
            
            aum = props.get("AUM", {}).get("select", {})
            aum = aum.get("name") if aum else None
            
            status = props.get("Status", {}).get("select", {})
            status = status.get("name") if status else None
            
            # Extract text values
            city = self._get_rich_text(props.get("CITY", {}))
            state = self._get_rich_text(props.get("STATE", {}))
            country = self._get_rich_text(props.get("COUNTRY", {}))
            
            return Prospect(
                page_id=page["id"],
                firm_name=firm_name,
                website=website,
                firm_type=firm_type,
                city=city,
                state=state,
                country=country,
                aum=aum,
                status=status
            )
        except Exception as e:
            logger.error(f"Error parsing prospect: {e}")
            return None
    
    def _get_rich_text(self, prop: Dict) -> Optional[str]:
        """Extract plain text from a rich_text property"""
        rich_text = prop.get("rich_text", [])
        if rich_text:
            return "".join(t.get("plain_text", "") for t in rich_text)
        return None
    
    def update_prospect_with_evaluation(self, page_id: str, result: EvaluationResult):
        """Update a Notion page with evaluation results"""
        properties = {
            "Status": {"select": {"name": "Rated"}},
            "StoneRiver Tier": {"select": {"name": result.tier}},
            "Reason (1-2 lines)": {
                "rich_text": [{"type": "text", "text": {"content": result.reason[:2000]}}]
            },
            "Firm Category (Expanded)": {"select": {"name": result.firm_category}},
            "Real Estate Signal": {"select": {"name": result.real_estate_signal}},
            "Real Asset Signal": {"select": {"name": result.real_asset_signal}},
            "Allocator Sophistication": {"select": {"name": result.allocator_sophistication}},
            "Next Action": {"select": {"name": result.next_action}},
            "Enirhcment Confidence": {"select": {"name": result.confidence}},
            "Last Processed": {
                "date": {"start": datetime.now(timezone.utc).strftime("%Y-%m-%d")}
            }
        }
        
        # Set Stage 1 Survives based on qualification
        properties["Stage 1 Survives"] = {"checkbox": result.qualified}
        
        # Add error if present
        if result.error:
            properties["Error"] = {
                "rich_text": [{"type": "text", "text": {"content": result.error[:2000]}}]
            }
        
        self.update_page(page_id, properties)
    
    def mark_prospect_error(self, page_id: str, error_message: str):
        """Mark a prospect as having an error"""
        properties = {
            "Status": {"select": {"name": "Error"}},
            "Error": {
                "rich_text": [{"type": "text", "text": {"content": error_message[:2000]}}]
            },
            "Last Processed": {
                "date": {"start": datetime.now(timezone.utc).strftime("%Y-%m-%d")}
            }
        }
        self.update_page(page_id, properties)

# ============================================================================
# CLAUDE EVALUATOR
# ============================================================================

class ClaudeEvaluator:
    """Claude API client for prospect evaluation with web search"""
    
    def __init__(self, api_key: str):
        self.client = anthropic.Anthropic(api_key=api_key)
    
    def evaluate_prospect(self, prospect: Prospect) -> EvaluationResult:
        """Evaluate a prospect using Claude with web search"""
        user_message = EVALUATION_USER_TEMPLATE.format(
            firm_name=prospect.firm_name or "Unknown",
            website=prospect.website or "Not provided",
            firm_type=prospect.firm_type or "Unknown",
            city=prospect.city or "Unknown",
            state=prospect.state or "",
            country=prospect.country or "Unknown",
            aum=prospect.aum or "Unknown"
        )
        
        try:
            # First, do web research using Claude with web search tool
            research_response = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=4096,
                tools=[{
                    "type": "web_search_20250305",
                    "name": "web_search",
                    "max_uses": 5
                }],
                messages=[{
                    "role": "user", 
                    "content": f"""Research this investment firm thoroughly:

Firm: {prospect.firm_name}
Website: {prospect.website or 'Not provided'}

Search for:
1. Their AUM (assets under management)
2. Their investment philosophy and strategy
3. What asset classes they invest in (especially real estate, alternatives, private investments)
4. Their client base (institutional, UHNW, retail)
5. Any news about their investments or allocations
6. Team members' backgrounds in real estate or alternatives

Provide a comprehensive summary of your findings."""
                }]
            )
            
            # Extract the research findings
            research_text = ""
            for block in research_response.content:
                if hasattr(block, 'text'):
                    research_text += block.text + "\n"
            
            # Now evaluate based on research
            eval_response = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1024,
                system=EVALUATION_SYSTEM_PROMPT,
                messages=[{
                    "role": "user",
                    "content": f"""{user_message}

**Research Findings:**
{research_text}

Based on the above research, provide your JSON evaluation."""
                }]
            )
            
            # Parse JSON response
            content = eval_response.content[0].text.strip()
            
            # Handle potential markdown code blocks
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
                content = content.strip()
            
            data = json.loads(content)
            
            # Build reason with evidence
            reason = data.get("reason", "No reason provided")
            evidence = data.get("evidence_summary", "")
            aum_found = data.get("aum_found")
            
            if aum_found:
                reason = f"AUM: {aum_found}. {reason}"
            
            return EvaluationResult(
                qualified=data.get("qualified", False),
                tier=data.get("tier", "Needs Verification"),
                reason=reason[:2000],  # Truncate if too long
                firm_category=data.get("firm_category", "Other"),
                real_estate_signal=data.get("real_estate_signal", "None"),
                real_asset_signal=data.get("real_asset_signal", "None"),
                allocator_sophistication=data.get("allocator_sophistication", "Retail"),
                next_action=data.get("next_action", "Skip"),
                confidence=data.get("confidence", "Low")
            )
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON parse error for {prospect.firm_name}: {e}")
            return EvaluationResult(
                qualified=False,
                tier="Needs Verification",
                reason=f"Evaluation parse error: {str(e)[:100]}",
                firm_category="Other",
                real_estate_signal="None",
                real_asset_signal="None",
                allocator_sophistication="Retail",
                next_action="Deep Research",
                confidence="Low",
                error=f"JSON parse error: {str(e)}"
            )
        except anthropic.APIError as e:
            logger.error(f"Claude API error for {prospect.firm_name}: {e}")
            raise  # Re-raise to trigger retry logic

# ============================================================================
# MAIN PROCESSOR
# ============================================================================

class ProspectProcessor:
    """Main processor that orchestrates the evaluation pipeline"""
    
    def __init__(self):
        self._validate_config()
        self.notion = NotionClient(NOTION_API_KEY)
        self.evaluator = ClaudeEvaluator(ANTHROPIC_API_KEY)
        self.checkpoint = CheckpointDB(CHECKPOINT_FILE)
    
    def _validate_config(self):
        """Validate required configuration"""
        if not NOTION_API_KEY:
            raise ValueError("NOTION_API_KEY environment variable is required")
        if not ANTHROPIC_API_KEY:
            raise ValueError("ANTHROPIC_API_KEY environment variable is required")
    
    def run(self, max_prospects: int = None, dry_run: bool = False):
        """Run the processor"""
        logger.info("=" * 60)
        logger.info("StoneRiver Prospect Processor Starting")
        logger.info("=" * 60)
        
        # Get checkpoint stats
        stats = self.checkpoint.get_stats()
        logger.info(f"Previous run stats: {stats}")
        
        # Fetch unprocessed prospects
        logger.info(f"Fetching unprocessed prospects from Notion...")
        prospects = self.notion.get_unprocessed_prospects(DATABASE_ID)
        logger.info(f"Found {len(prospects)} unprocessed prospects")
        
        # Filter out already processed (from checkpoint)
        prospects = [
            p for p in prospects 
            if not self.checkpoint.is_processed(p.page_id)
        ]
        logger.info(f"After checkpoint filter: {len(prospects)} remaining")
        
        if max_prospects:
            prospects = prospects[:max_prospects]
            logger.info(f"Limited to {max_prospects} prospects")
        
        if not prospects:
            logger.info("No prospects to process. Exiting.")
            return
        
        # Process in batches
        processed = 0
        errors = 0
        
        for i, prospect in enumerate(prospects):
            try:
                logger.info(f"[{i+1}/{len(prospects)}] Processing: {prospect.firm_name}")
                
                if dry_run:
                    logger.info(f"  DRY RUN - Would evaluate: {prospect.website}")
                    continue
                
                # Evaluate with Claude
                result = self._evaluate_with_retry(prospect)
                
                # Update Notion
                self.notion.update_prospect_with_evaluation(prospect.page_id, result)
                
                # Mark checkpoint
                self.checkpoint.mark_processed(
                    prospect.page_id, 
                    prospect.firm_name, 
                    success=True
                )
                
                processed += 1
                logger.info(f"  Result: {result.tier} - {result.reason[:80]}...")
                
                # Rate limiting
                time.sleep(SLEEP_BETWEEN_PROSPECTS)
                
                # Batch pause
                if (i + 1) % BATCH_SIZE == 0:
                    logger.info(f"Batch complete. Sleeping {SLEEP_BETWEEN_BATCHES}s...")
                    time.sleep(SLEEP_BETWEEN_BATCHES)
                    
            except Exception as e:
                errors += 1
                logger.error(f"  ERROR: {e}")
                
                # Mark error in Notion
                try:
                    self.notion.mark_prospect_error(prospect.page_id, str(e))
                except:
                    pass
                
                # Mark checkpoint
                self.checkpoint.mark_processed(
                    prospect.page_id,
                    prospect.firm_name,
                    success=False,
                    error=str(e)
                )
                
                # Continue to next prospect
                continue
        
        # Final stats
        logger.info("=" * 60)
        logger.info("Processing Complete")
        logger.info(f"  Processed: {processed}")
        logger.info(f"  Errors: {errors}")
        logger.info(f"  Total checkpoint: {self.checkpoint.get_stats()}")
        logger.info("=" * 60)
    
    def _evaluate_with_retry(self, prospect: Prospect, max_retries: int = MAX_RETRIES) -> EvaluationResult:
        """Evaluate a prospect with retry logic"""
        last_error = None
        
        for attempt in range(max_retries):
            try:
                return self.evaluator.evaluate_prospect(prospect)
            except anthropic.RateLimitError as e:
                last_error = e
                wait_time = (attempt + 1) * 30  # Exponential backoff
                logger.warning(f"Rate limited. Waiting {wait_time}s... (attempt {attempt + 1})")
                time.sleep(wait_time)
            except anthropic.APIError as e:
                last_error = e
                wait_time = (attempt + 1) * 5
                logger.warning(f"API error. Waiting {wait_time}s... (attempt {attempt + 1})")
                time.sleep(wait_time)
        
        # All retries exhausted
        raise last_error

# ============================================================================
# ENTRY POINT
# ============================================================================

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="StoneRiver Prospect Processor")
    parser.add_argument("--max", type=int, help="Maximum prospects to process")
    parser.add_argument("--dry-run", action="store_true", help="Dry run (no changes)")
    parser.add_argument("--continuous", action="store_true", help="Run continuously")
    parser.add_argument("--interval", type=int, default=3600, help="Seconds between runs (continuous mode)")
    
    args = parser.parse_args()
    
    processor = ProspectProcessor()
    
    if args.continuous:
        logger.info(f"Running in continuous mode (interval: {args.interval}s)")
        while True:
            try:
                processor.run(max_prospects=args.max, dry_run=args.dry_run)
            except Exception as e:
                logger.error(f"Run failed: {e}")
            
            logger.info(f"Sleeping {args.interval}s until next run...")
            time.sleep(args.interval)
    else:
        processor.run(max_prospects=args.max, dry_run=args.dry_run)

if __name__ == "__main__":
    main()
