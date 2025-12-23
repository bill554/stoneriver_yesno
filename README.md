# StoneRiver Prospect Processor

Automated prospect qualification system for StoneRiver Investment Fund III. This service continuously processes prospects from a Notion database, evaluates them against StoneRiver's investor criteria using Claude AI, and writes the results back to Notion.

## Features

- **Continuous Processing**: Runs 24/7, checking for new unprocessed prospects
- **Resilient**: Checkpoint system survives restarts, retries on failures
- **Rate Limited**: Respects API limits for both Notion and Claude
- **Detailed Logging**: Full visibility into processing status
- **Configurable**: All parameters adjustable via environment variables

## Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Notion DB      │────▶│   Processor     │────▶│  Notion DB      │
│  (Prospects)    │     │  (Railway)      │     │  (Results)      │
└─────────────────┘     └────────┬────────┘     └─────────────────┘
                                 │
                                 ▼
                        ┌─────────────────┐
                        │  Claude API     │
                        │  (Evaluation)   │
                        └─────────────────┘
```

## Processing Flow

1. **Query Notion**: Fetch prospects where:
   - `FIRM NAME` is not empty
   - `WEBSITE` is not empty
   - `Status` is empty, "Queued", or "Enriched"

2. **Evaluate with Claude**: For each prospect:
   - Send firm details to Claude
   - Get structured qualification assessment

3. **Update Notion**: Write back:
   - `Status` → "Rated"
   - `StoneRiver Tier` → Tier 1/2/De-Prioritize/Needs Verification
   - `Reason (1-2 lines)` → Qualification reasoning
   - `Firm Category (Expanded)` → Detailed firm type
   - `Real Estate Signal` → None/Weak/Strong
   - `Real Asset Signal` → None/Weak/Strong
   - `Allocator Sophistication` → Retail/Semi/Institutional
   - `Next Action` → Skip/Deep Research/Find Decision Makers/Outreach
   - `Stage 1 Survives` → Yes/No
   - `Last Processed` → Today's date

## Railway Deployment

### 1. Create Railway Project

```bash
# Install Railway CLI
npm install -g @railway/cli

# Login
railway login

# Create project
railway init
```

### 2. Set Environment Variables

In the Railway dashboard, add these variables:

| Variable | Required | Description |
|----------|----------|-------------|
| `NOTION_API_KEY` | Yes | Notion integration token |
| `ANTHROPIC_API_KEY` | Yes | Anthropic API key |
| `NOTION_DATABASE_ID` | No | Database ID (default: StoneRiver Q1 Surge) |
| `BATCH_SIZE` | No | Prospects per batch (default: 10) |
| `SLEEP_BETWEEN_PROSPECTS` | No | Seconds between prospects (default: 2.0) |
| `SLEEP_BETWEEN_BATCHES` | No | Seconds between batches (default: 30.0) |
| `LOG_LEVEL` | No | Logging level (default: INFO) |

### 3. Add Persistent Volume

In Railway dashboard:
1. Go to your service
2. Click "Add Volume"
3. Mount path: `/data`
4. This stores the checkpoint database

### 4. Deploy

```bash
# Deploy from local directory
railway up

# Or connect to GitHub for auto-deploys
railway link
```

## Local Development

### Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install dependencies
pip install -r requirements.txt

# Set environment variables
export NOTION_API_KEY="your-notion-key"
export ANTHROPIC_API_KEY="your-anthropic-key"
```

### Run

```bash
# Process all unprocessed prospects (one-time)
python processor.py

# Process limited number
python processor.py --max 10

# Dry run (no changes)
python processor.py --dry-run --max 5

# Continuous mode (for production)
python processor.py --continuous --interval 1800
```

## Notion Setup

### Required Integration Permissions

Your Notion integration needs:
- Read content
- Update content
- Insert content

### Database Schema

The processor expects these Notion properties:

**Input (read):**
- `FIRM NAME` (title)
- `WEBSITE` (url)
- `FIRM TYPE` (select)
- `CITY` (rich_text)
- `STATE` (rich_text)
- `COUNTRY` (rich_text)
- `AUM` (select)
- `Status` (select)

**Output (written):**
- `Status` (select)
- `StoneRiver Tier` (select)
- `Reason (1-2 lines)` (rich_text)
- `Firm Category (Expanded)` (select)
- `Real Estate Signal` (select)
- `Real Asset Signal` (select)
- `Allocator Sophistication` (select)
- `Next Action` (select)
- `Stage 1 Survives` (checkbox)
- `Last Processed` (date)
- `Error` (rich_text)
- `Enirhcment Confidence` (select)

## StoneRiver Qualification Criteria

### Qualified Allocator Types
- Family offices (single or multi-family)
- RIAs with $500M+ AUM and alternatives capability
- Endowments & foundations
- Public/corporate pensions with real estate allocations
- OCIOs with real asset mandates

### Automatic Disqualifiers
- Retail-focused without alternatives capability
- AUM < $200M (can't write $5M+ checks)
- No real estate/real assets allocation history
- Insurance/benefits-focused without investment mandates
- Broker-dealers without discretionary management

### Tier Definitions
- **Tier 1**: Strong fit - institutional allocator with clear RE allocation
- **Tier 2**: Potential fit - meets criteria but needs more research
- **De-Prioritize**: Does not meet criteria
- **Needs Verification**: Insufficient information

## Monitoring

### Logs

Railway provides real-time logs. Key log patterns:

```
[INFO] Processing: [Firm Name]
[INFO] Result: Tier 1 - Strong institutional family office...
[WARNING] Rate limited. Waiting 30s...
[ERROR] API error for [Firm Name]: ...
```

### Checkpoint Stats

The processor logs checkpoint statistics at startup:
```
Previous run stats: {'success': 150, 'failed': 3, 'total': 153}
```

### Recovery

If the processor crashes:
1. Railway auto-restarts the container
2. Checkpoint database (on persistent volume) tracks processed prospects
3. Processing resumes from where it left off

## Cost Estimates

### Claude API
- ~500 tokens per evaluation
- At $0.003/1K input + $0.015/1K output (Sonnet)
- ~$0.01 per prospect
- 1,000 prospects ≈ $10

### Railway
- Starter plan: $5/month includes enough compute
- Volume storage: $0.25/GB/month

## Troubleshooting

### "Rate limited" warnings
Normal behavior - the processor automatically waits and retries.

### "Notion API error 400"
Usually a schema mismatch. Verify all select options exist in Notion.

### Stuck on same prospect
Check the checkpoint database:
```bash
sqlite3 /data/processor_checkpoint.db "SELECT * FROM processed ORDER BY processed_at DESC LIMIT 10"
```

### Reset checkpoint (reprocess all)
```bash
rm /data/processor_checkpoint.db
# Or in Railway: delete and recreate the volume
```

## License

Proprietary - StoneRiver Company
