#!/bin/bash

# SeeSea Intelligence Agent - AWS EC2 Deployment Script
# Usage: ./scripts/deploy-aws.sh

set -e

# Configuration
SSH_KEY="${SSH_KEY:-/home/jaqq-fast-doge/kacha.pem}"
SSH_USER="ubuntu"
SSH_HOST="ec2-13-52-37-94.us-west-1.compute.amazonaws.com"
REMOTE_DIR="/home/ubuntu/seesea-agent"
REMOTE_API_DIR="/home/ubuntu/seesea-api"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}SeeSea Agent Deployment to AWS EC2${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# Check SSH key
if [ ! -f "$SSH_KEY" ]; then
    echo -e "${RED}Error: SSH key not found at ${SSH_KEY}${NC}"
    exit 1
fi

# Check .env file
if [ ! -f ".env" ]; then
    echo -e "${RED}Error: .env file not found${NC}"
    echo -e "${YELLOW}Please create it from .env.example before deploying${NC}"
    exit 1
fi

echo -e "${GREEN}[1/6] Testing SSH connection...${NC}"
ssh -i "$SSH_KEY" -o ConnectTimeout=10 "${SSH_USER}@${SSH_HOST}" "echo 'SSH connection successful'" || {
    echo -e "${RED}Error: Cannot connect to EC2 instance${NC}"
    exit 1
}

echo -e "${GREEN}[2/6] Creating remote directory structure...${NC}"
ssh -i "$SSH_KEY" "${SSH_USER}@${SSH_HOST}" << 'EOF'
    mkdir -p ~/seesea-agent
    mkdir -p ~/seesea-agent/backup
EOF

echo -e "${GREEN}[3/6] Backing up current deployment (if exists)...${NC}"
ssh -i "$SSH_KEY" "${SSH_USER}@${SSH_HOST}" << EOF
    if [ -d "${REMOTE_DIR}/server.py" ]; then
        BACKUP_NAME="backup-\$(date +%Y%m%d-%H%M%S)"
        echo "Creating backup: \${BACKUP_NAME}"
        mkdir -p "${REMOTE_DIR}/backup/\${BACKUP_NAME}"

        # Backup current deployment
        cp -r ${REMOTE_DIR}/*.py ${REMOTE_DIR}/backup/\${BACKUP_NAME}/ 2>/dev/null || true
        cp ${REMOTE_DIR}/.env ${REMOTE_DIR}/backup/\${BACKUP_NAME}/ 2>/dev/null || true

        # Keep only last 5 backups
        cd ${REMOTE_DIR}/backup && ls -t | tail -n +6 | xargs -r rm -rf
    fi
EOF

echo -e "${GREEN}[4/6] Uploading application files...${NC}"
rsync -avz --progress \
    -e "ssh -i $SSH_KEY" \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='venv/' \
    --exclude='.pytest_cache' \
    --exclude='notebooks/' \
    --exclude='.ipynb_checkpoints' \
    ./ "${SSH_USER}@${SSH_HOST}:${REMOTE_DIR}/"

echo -e "${GREEN}[5/6] Starting Agent service...${NC}"
ssh -i "$SSH_KEY" "${SSH_USER}@${SSH_HOST}" << EOF
    cd ${REMOTE_DIR}

    # Make sure Docker network exists (shared with API)
    docker network create seesea-network 2>/dev/null || echo "Network already exists"

    # Stop existing container
    docker-compose down 2>/dev/null || true

    # Build and start service
    docker-compose up -d --build

    # Wait for service to start
    echo "Waiting for service to start..."
    sleep 10

    # Check service status
    docker-compose ps
EOF

echo -e "${GREEN}[6/6] Verifying deployment...${NC}"
ssh -i "$SSH_KEY" "${SSH_USER}@${SSH_HOST}" << EOF
    cd ${REMOTE_DIR}

    echo ""
    echo "=== Service Health Check ==="

    # Check Agent API
    if curl -s http://localhost:8002/health > /dev/null 2>&1; then
        echo "✓ Agent API is running (port 8002)"
    else
        echo "✗ Agent API is not responding"
    fi

    echo ""
    echo "=== Container Status ==="
    docker ps --filter name=seesea-agent --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

    echo ""
    echo "=== Recent Logs ==="
    docker-compose logs --tail=20 agent
EOF

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Deployment Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "${YELLOW}Agent is now running at:${NC}"
echo -e "  Internal: http://localhost:8002"
echo -e "  External: https://api.seesea.ai/agent/* (after Nginx config update)"
echo ""
echo -e "${YELLOW}API Endpoints:${NC}"
echo -e "  Chat:         POST /agent/chat"
echo -e "  Direct Query: POST /agent/query/direct"
echo -e "  Summary:      GET  /agent/query/summary"
echo -e "  Health:       GET  /agent/health"
echo ""
echo -e "${YELLOW}Useful commands:${NC}"
echo -e "  View logs:    ssh -i $SSH_KEY ${SSH_USER}@${SSH_HOST} 'cd ${REMOTE_DIR} && docker-compose logs -f'"
echo -e "  Restart:      ssh -i $SSH_KEY ${SSH_USER}@${SSH_HOST} 'cd ${REMOTE_DIR} && docker-compose restart'"
echo -e "  Stop:         ssh -i $SSH_KEY ${SSH_USER}@${SSH_HOST} 'cd ${REMOTE_DIR} && docker-compose down'"
echo ""
echo -e "${YELLOW}Next Steps:${NC}"
echo -e "  1. Update API Nginx config to route /agent/* to Agent service"
echo -e "  2. Restart API Nginx: cd ${REMOTE_API_DIR}/infrastructure/docker && docker-compose restart nginx"
echo ""
