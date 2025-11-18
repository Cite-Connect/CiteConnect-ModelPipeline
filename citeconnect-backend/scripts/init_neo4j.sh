#!/bin/bash

# Neo4j Schema Initialization Script
# Reads credentials from .env file

echo "============================================================"
echo "  Initializing Neo4j Schema for CiteConnect"
echo "============================================================"
echo ""

# Load environment variables from .env
if [ -f "../.env" ]; then
    export $(cat ../.env | grep -v '^#' | xargs)
    echo "✓ Loaded configuration from .env"
else
    echo "✗ .env file not found!"
    echo "  Please create .env file in ModelPipeline root"
    exit 1
fi

# Use environment variables (with defaults as fallback)
CONTAINER="${NEO4J_CONTAINER:-citeconnect-neo4j}"
USER="${NEO4J_USER:-neo4j}"
PASSWORD="${NEO4J_PASSWORD:-password}"

echo "Using Neo4j credentials from .env:"
echo "  Container: $CONTAINER"
echo "  User: $USER"
echo ""

echo "Step 1: Creating Node Constraints..."
echo "--------------------------------------"

docker exec $CONTAINER cypher-shell -u $USER -p $PASSWORD \
  "CREATE CONSTRAINT paper_id_unique IF NOT EXISTS FOR (p:Paper) REQUIRE p.paper_id IS UNIQUE;" 2>/dev/null

docker exec $CONTAINER cypher-shell -u $USER -p $PASSWORD \
  "CREATE CONSTRAINT user_id_unique IF NOT EXISTS FOR (u:User) REQUIRE u.user_id IS UNIQUE;" 2>/dev/null

echo "✓ Constraints created"
echo ""

echo "Step 2: Creating Indexes..."
echo "--------------------------------------"

docker exec $CONTAINER cypher-shell -u $USER -p $PASSWORD \
  "CREATE INDEX paper_id_index IF NOT EXISTS FOR (p:Paper) ON (p.paper_id);" 2>/dev/null

docker exec $CONTAINER cypher-shell -u $USER -p $PASSWORD \
  "CREATE INDEX paper_domain_index IF NOT EXISTS FOR (p:Paper) ON (p.domain);" 2>/dev/null

docker exec $CONTAINER cypher-shell -u $USER -p $PASSWORD \
  "CREATE INDEX paper_year_index IF NOT EXISTS FOR (p:Paper) ON (p.year);" 2>/dev/null

docker exec $CONTAINER cypher-shell -u $USER -p $PASSWORD \
  "CREATE INDEX user_id_index IF NOT EXISTS FOR (u:User) ON (u.user_id);" 2>/dev/null

docker exec $CONTAINER cypher-shell -u $USER -p $PASSWORD \
  "CREATE INDEX user_domain_index IF NOT EXISTS FOR (u:User) ON (u.domain);" 2>/dev/null

echo "✓ Indexes created"
echo ""

echo "Step 3: Verifying Schema..."
echo "--------------------------------------"
echo ""

# Show constraints
docker exec $CONTAINER cypher-shell -u $USER -p $PASSWORD \
  "SHOW CONSTRAINTS;" > /tmp/neo4j_constraints.txt 2>&1
cat /tmp/neo4j_constraints.txt

echo ""

# Show indexes
docker exec $CONTAINER cypher-shell -u $USER -p $PASSWORD \
  "SHOW INDEXES;" > /tmp/neo4j_indexes.txt 2>&1
cat /tmp/neo4j_indexes.txt

echo ""
echo "============================================================"
echo "  Neo4j Schema Initialized Successfully!"
echo "============================================================"
echo ""
echo "Access Neo4j Browser: http://localhost:7474"
echo "  Username: $USER"
echo "  Password: [from .env]"
echo ""
