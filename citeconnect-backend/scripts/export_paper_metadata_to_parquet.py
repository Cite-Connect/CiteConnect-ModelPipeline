#!/usr/bin/env python3
"""
Export paper metadata from database to parquet format for bias analysis.

This script:
1. Queries all papers from the database
2. Maps domain to fieldsOfStudy (research fields)
3. Exports to parquet format expected by model_bias_slicing.py

Output: data/combined_gcs_data.parquet
"""

import asyncio
import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.db.connection import DatabaseConnection
from app.utils.logger import get_logger

logger = get_logger(__name__)

# Domain to research fields mapping
# Maps your domain categories to standard research fields
DOMAIN_TO_FIELDS = {
    "healthcare": [
        "Medicine",
        "Biology",
        "Health Sciences",
        "Biomedical Engineering",
        "Pharmacology",
        "Public Health"
    ],
    "fintech": [
        "Computer Science",
        "Economics",
        "Finance",
        "Business",
        "Mathematics",
        "Statistics"
    ],
    "quantum_computing": [
        "Computer Science",
        "Physics",
        "Mathematics",
        "Quantum Physics",
        "Theoretical Computer Science"
    ],
    "ai": [
        "Computer Science",
        "Artificial Intelligence",
        "Machine Learning",
        "Mathematics",
        "Statistics",
        "Cognitive Science"
    ]
}


def map_domain_to_fields(domain: str) -> list:
    """
    Map domain to list of research fields.
    
    Args:
        domain: Paper domain (healthcare, fintech, quantum_computing)
        
    Returns:
        List of research fields
    """
    if domain and domain in DOMAIN_TO_FIELDS:
        return DOMAIN_TO_FIELDS[domain]
    return ["Unknown"]


async def export_paper_metadata():
    """Export all paper metadata to parquet format."""
    print("\n" + "=" * 80)
    print("  EXPORT PAPER METADATA TO PARQUET")
    print("=" * 80 + "\n")
    
    db = DatabaseConnection()
    await db.connect()
    
    try:
        # Query all papers
        print("📊 Querying papers from database...")
        query = """
            SELECT 
                paper_id,
                title,
                year,
                citation_count,
                domain,
                venue,
                authors
            FROM papers
            ORDER BY paper_id
        """
        
        rows = await db.fetch(query)
        print(f"   Found {len(rows)} papers in database\n")
        
        if not rows:
            print("⚠️  No papers found in database!")
            return
        
        # Convert to list of dicts
        papers_data = []
        for row in rows:
            paper_dict = dict(row)
            
            # Map domain to fieldsOfStudy
            domain = paper_dict.get("domain")
            fields_of_study = map_domain_to_fields(domain)
            
            # Create record in expected format
            # Note: We use paperId as primary key (some scripts expect this)
            # But also include paper_id for compatibility
            paper_record = {
                "paperId": str(paper_dict["paper_id"]),  # Primary key (camelCase for compatibility)
                "title": paper_dict.get("title", ""),
                "year": paper_dict.get("year"),
                "citationCount": paper_dict.get("citation_count", 0),
                "fieldsOfStudy": fields_of_study,  # List of research fields
                "domain": domain,  # Keep original domain for reference
                "venue": paper_dict.get("venue"),
                "authors": paper_dict.get("authors", [])
            }
            papers_data.append(paper_record)
        
        # Create DataFrame
        print("📝 Creating DataFrame...")
        df = pd.DataFrame(papers_data)
        
        # Show summary
        print(f"\n   Total papers: {len(df)}")
        print(f"   Domains: {df['domain'].value_counts().to_dict()}")
        print(f"   Papers with year: {df['year'].notna().sum()}")
        print(f"   Papers with citations: {(df['citationCount'] > 0).sum()}")
        
        # Show field distribution
        print(f"\n   Research fields distribution:")
        all_fields = []
        for fields in df['fieldsOfStudy']:
            all_fields.extend(fields)
        field_counts = pd.Series(all_fields).value_counts()
        for field, count in field_counts.head(10).items():
            print(f"      {field}: {count}")
        
        # Create output directory
        output_dir = Path(__file__).parent.parent / "data"
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / "combined_gcs_data.parquet"
        
        # Save to parquet
        print(f"\n💾 Saving to parquet...")
        df.to_parquet(output_path, index=False, engine='pyarrow')
        
        print(f"\n✅ Successfully exported {len(df)} papers to:")
        print(f"   {output_path.resolve()}")
        
        # Verify file
        if output_path.exists():
            file_size = output_path.stat().st_size / (1024 * 1024)  # MB
            print(f"   File size: {file_size:.2f} MB")
            
            # Quick verification read
            verify_df = pd.read_parquet(output_path)
            print(f"\n✅ Verification: Read {len(verify_df)} papers from parquet")
            print(f"   Columns: {list(verify_df.columns)}")
            print(f"   Sample fieldsOfStudy: {verify_df['fieldsOfStudy'].iloc[0]}")
        
        print("\n" + "=" * 80)
        print("  NEXT STEPS")
        print("=" * 80)
        print("\n1. Run paper-field bias analysis:")
        print("   python scripts/model_bias_slicing.py")
        print("\n2. This will generate:")
        print("   - model_bias_report.json")
        print("   - fairness_config.json (with under-served fields)")
        print("   - bias_plots/model_precision_by_field.png")
        print("\n3. The fairness service will then be able to:")
        print("   - Map papers to research fields")
        print("   - Boost papers from under-served fields")
        print("=" * 80 + "\n")
        
    except Exception as e:
        logger.error(f"Export failed: {e}", exc_info=True)
        print(f"\n❌ Error: {e}")
        raise
    
    finally:
        await db.disconnect()


if __name__ == "__main__":
    asyncio.run(export_paper_metadata())
