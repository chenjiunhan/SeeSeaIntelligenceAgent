# SeeSeaIntelligenceAgent

AI Agent for analyzing vessel traffic data using LangGraph and Google Gemini.

## Features

- **LangGraph Agent**: Conversational AI agent with tool-calling capabilities
- **FastAPI Server**: REST API for interacting with the agent
- **Data Query Tools**: Query vessel arrivals, compare periods, get summaries
- **Session Management**: Maintain conversation context across requests

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment

Create `.env` file:

```env
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-2.0-flash
```

### 3. Run the Server

```bash
# Start FastAPI server
python server.py

# Or with uvicorn directly
uvicorn server:app --host 0.0.0.0 --port 8002 --reload
```

Server will be available at:
- API: `http://localhost:8002`
- Docs: `http://localhost:8002/docs`
- Health: `http://localhost:8002/health`

## API Endpoints

### Chat with Agent

```bash
POST /chat
Content-Type: application/json

{
  "message": "How many container ships passed through in January 2026?",
  "session_id": "user-123"
}
```

**Response:**
```json
{
  "response": "In January 2026, there were 123 container ships...",
  "session_id": "user-123",
  "tool_calls": [
    {
      "tool": "query_vessel_data",
      "args": {"start_date": "2026-01-01", "end_date": "2026-01-31"}
    }
  ]
}
```

### Direct Data Query

```bash
POST /query/direct
Content-Type: application/json

{
  "start_date": "2026-01-01",
  "end_date": "2026-01-31",
  "vessel_type": "container",
  "aggregate": "total"
}
```

### Get Dataset Summary

```bash
GET /query/summary
```

### Session Management

```bash
# List active sessions
GET /sessions

# Clear a session
POST /clear-session?session_id=user-123
```

## Example Usage

### Python Client

```python
import requests

# Chat with agent
response = requests.post("http://localhost:8002/chat", json={
    "message": "Compare vessel traffic between 2024 and 2025",
    "session_id": "user-1"
})

print(response.json()["response"])
```

### cURL

```bash
# Chat
curl -X POST http://localhost:8002/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is the dataset summary?", "session_id": "test-1"}'

# Direct query
curl -X POST http://localhost:8002/query/direct \
  -H "Content-Type: application/json" \
  -d '{"start_date": "2026-01-01", "end_date": "2026-01-31", "vessel_type": "tanker"}'
```

## Agent Architecture

```
User Request
    ↓
FastAPI Server (server.py)
    ↓
LangGraph Agent (logistics_agent.py)
    ↓
┌─────────────────┐
│ Tool Selection  │
│  - query_vessel_data
│  - get_data_summary
│  - compare_periods
└─────────────────┘
    ↓
Data Processing (pandas)
    ↓
Response to User
```

## Available Tools

1. **query_vessel_data**: Query vessel arrivals by date range and type
2. **get_data_summary**: Get dataset statistics and info
3. **compare_periods**: Compare traffic between two time periods

## Data Source

The agent queries data from `SeeSeaIntelligence/processed/logistics/chokepoints/bab-el-mandeb/vessel_arrivals/vessel_arrivals.csv`

Data coverage:
- **Date Range**: 2019-01-01 to 2026-01-25
- **Vessel Types**: container, dry_bulk, tanker, general_cargo, roro
- **Update Frequency**: As per data collection schedule

## Testing

### Test Agent Directly

```bash
python logistics_agent.py
```

### Test API Server

```bash
# Start server
python server.py

# In another terminal, test endpoints
curl http://localhost:8002/health
curl http://localhost:8002/query/summary
```

## Deployment

### Local Development with Docker Compose

```bash
# 1. Configure environment
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY

# 2. Start service
docker-compose up -d

# 3. Check status
docker-compose ps
docker-compose logs -f
```

### Deploy to AWS EC2

```bash
# 1. Configure environment
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY

# 2. Deploy to AWS
./scripts/deploy-aws.sh

# 3. Update API Nginx (if not already done)
# The Nginx config has been updated to route /agent/* to the Agent service
# Redeploy the API to apply the changes:
cd ../SeeSeaIntelligenceAPI
./scripts/deploy-aws.sh
```

**After deployment, Agent will be available at:**
- Internal: `http://localhost:8002`
- External: `https://api.seesea.ai/agent/*`

**API Endpoints:**
- `https://api.seesea.ai/agent/chat` - Chat with agent
- `https://api.seesea.ai/agent/query/direct` - Direct data query
- `https://api.seesea.ai/agent/query/summary` - Dataset summary
- `https://api.seesea.ai/agent/health` - Health check

### Architecture

```
User Request → Nginx (SSL, port 443)
               ↓
           Docker Nginx (port 80 internal)
               ↓
           /agent/* → Agent Container (port 8002)
                      ↓
                   Data Volume (/data/processed)
```

The Agent runs in its own Docker container and:
- Shares the `seesea-network` with API services
- Mounts data from `/home/ubuntu/SeeSeaIntelligence/processed` (if available)
- Proxied through Nginx at `/agent/*` path

### Production Considerations

- ✅ Docker Compose deployment
- ✅ Shared Docker network with API
- ✅ Data volume mounting
- ✅ Health checks
- ✅ SSL via Nginx
- ⚠️ Use Redis for session storage instead of in-memory
- ⚠️ Add authentication/API keys
- ⚠️ Set up logging and monitoring

## License

Proprietary
