#!/bin/bash

# Seed Neo4j with sample citation data
# Reads credentials from .env file

echo "Seeding Neo4j with sample data..."
echo ""

# Load environment variables from .env
if [ -f "../../.env" ]; then
    export $(cat ../../.env | grep -v '^#' | xargs)
    echo "✓ Loaded configuration from .env"
else
    echo "✗ .env file not found!"
    exit 1
fi

# Use environment variables
CONTAINER="${NEO4J_CONTAINER:-citeconnect-neo4j}"
USER="${NEO4J_USER:-neo4j}"
PASSWORD="${NEO4J_PASSWORD:-password}"

echo "Using Neo4j: $CONTAINER (user: $USER)"
echo ""

echo "Creating paper nodes..."

docker exec $CONTAINER cypher-shell -u $USER -p $PASSWORD \
  "CREATE (p1:Paper {paper_id: 'arxiv:2401.001', title: 'AlphaFold 2: Improved Protein Structure Prediction', year: 2021, domain: 'healthcare', citation_count: 9432});" 2>/dev/null

docker exec $CONTAINER cypher-shell -u $USER -p $PASSWORD \
  "CREATE (p2:Paper {paper_id: 'arxiv:2401.002', title: 'RoseTTAFold: Accurate Protein Structure Prediction', year: 2021, domain: 'healthcare', citation_count: 2156});" 2>/dev/null

docker exec $CONTAINER cypher-shell -u $USER -p $PASSWORD \
  "CREATE (p3:Paper {paper_id: 'arxiv:2401.003', title: 'Deep Learning for Drug Discovery', year: 2023, domain: 'healthcare', citation_count: 450});" 2>/dev/null

docker exec $CONTAINER cypher-shell -u $USER -p $PASSWORD \
  "CREATE (p4:Paper {paper_id: 'arxiv:2401.004', title: 'Transformer Models for Genomics', year: 2024, domain: 'healthcare', citation_count: 120});" 2>/dev/null

docker exec $CONTAINER cypher-shell -u $USER -p $PASSWORD \
  "CREATE (p5:Paper {paper_id: 'arxiv:2401.005', title: 'AI in Clinical Decision Support', year: 2024, domain: 'healthcare', citation_count: 89});" 2>/dev/null

echo "✓ 5 papers created"

echo "Creating citation relationships..."

docker exec $CONTAINER cypher-shell -u $USER -p $PASSWORD \
  "MATCH (p3:Paper {paper_id: 'arxiv:2401.003'}), (p1:Paper {paper_id: 'arxiv:2401.001'}) CREATE (p3)-[:CITES {citation_context: 'In the introduction'}]->(p1);" 2>/dev/null

docker exec $CONTAINER cypher-shell -u $USER -p $PASSWORD \
  "MATCH (p3:Paper {paper_id: 'arxiv:2401.003'}), (p2:Paper {paper_id: 'arxiv:2401.002'}) CREATE (p3)-[:CITES]->(p2);" 2>/dev/null

docker exec $CONTAINER cypher-shell -u $USER -p $PASSWORD \
  "MATCH (p4:Paper {paper_id: 'arxiv:2401.004'}), (p1:Paper {paper_id: 'arxiv:2401.001'}) CREATE (p4)-[:CITES]->(p1);" 2>/dev/null

docker exec $CONTAINER cypher-shell -u $USER -p $PASSWORD \
  "MATCH (p5:Paper {paper_id: 'arxiv:2401.005'}), (p3:Paper {paper_id: 'arxiv:2401.003'}) CREATE (p5)-[:CITES]->(p3);" 2>/dev/null

echo "✓ 4 citation relationships created"

echo "Creating reverse CITED_BY relationships..."

docker exec $CONTAINER cypher-shell -u $USER -p $PASSWORD \
  "MATCH (p1:Paper)-[c:CITES]->(p2:Paper) WHERE NOT exists((p2)-[:CITED_BY]->(p1)) CREATE (p2)-[:CITED_BY]->(p1);" 2>/dev/null

echo "✓ Reverse relationships created"
echo ""

# Count results
echo "Verifying data..."
PAPER_COUNT=$(docker exec $CONTAINER cypher-shell -u $USER -p $PASSWORD \
  "MATCH (p:Paper) RETURN count(p) as count;" 2>/dev/null | grep -E "^\s*[0-9]+" | tr -d ' ')

CITATION_COUNT=$(docker exec $CONTAINER cypher-shell -u $USER -p $PASSWORD \
  "MATCH ()-[r:CITES]->() RETURN count(r) as count;" 2>/dev/null | grep -E "^\s*[0-9]+" | tr -d ' ')

echo ""
echo "============================================================"
echo "  Sample Data Seeded Successfully!"
echo "============================================================"
echo ""
echo "Created:"
echo "  - Papers: $PAPER_COUNT"
echo "  - Citations: $CITATION_COUNT"
echo ""
echo "View in Neo4j Browser: http://localhost:7474"
echo "Try query: MATCH (p:Paper) RETURN p LIMIT 25"
echo ""
