# alembic/versions/001_initial_schema.py

"""
Initial database schema

Creates all tables for CiteConnect application based on LLD Section 5.1.

Tables created:
- users
- user_domains
- user_interests
- user_profile_embeddings
- papers
- user_interactions
- user_saved_papers
- user_liked_papers
- paper_clusters
- cluster_papers
- rate_limits
- system_metrics

Revision ID: 001_initial_schema
Create Date: 2025-11-11
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers
revision = '001_initial_schema'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Create all tables."""
    
    # ==================== Users & Authentication ====================
    
    # Users table
    op.create_table(
        'users',
        sa.Column('user_id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('email', sa.String(length=255), nullable=False),
        sa.Column('password_hash', sa.String(length=255), nullable=False),
        sa.Column('name', sa.String(length=255), nullable=False),
        sa.Column('created_at', sa.TIMESTAMP(), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=False),
        sa.Column('updated_at', sa.TIMESTAMP(), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=False),
        sa.Column('is_active', sa.Boolean(), server_default=sa.text('TRUE'), nullable=False),
        sa.Column('google_scholar_url', sa.String(length=500), nullable=True),
        sa.Column('semantic_scholar_author_id', sa.String(length=100), nullable=True),
        sa.PrimaryKeyConstraint('user_id'),
        sa.UniqueConstraint('email')
    )
    
    # Create indexes for users
    op.create_index('idx_users_email', 'users', ['email'])
    op.create_index('idx_users_semantic_scholar', 'users', ['semantic_scholar_author_id'])
    
    # User domains table (single domain per user)
    op.create_table(
        'user_domains',
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('domain', sa.String(length=50), nullable=False),
        sa.Column('selected_at', sa.TIMESTAMP(), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=False),
        sa.ForeignKeyConstraint(['user_id'], ['users.user_id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('user_id'),
        sa.CheckConstraint("domain IN ('healthcare', 'fintech', 'quantum_computing')", name='check_domain')
    )
    
    # User interests table
    op.create_table(
        'user_interests',
        sa.Column('interest_id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('interest_keyword', sa.String(length=100), nullable=False),
        sa.Column('source', sa.String(length=50), nullable=False),
        sa.Column('weight', sa.Float(), server_default=sa.text('1.0'), nullable=False),
        sa.Column('created_at', sa.TIMESTAMP(), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=False),
        sa.ForeignKeyConstraint(['user_id'], ['users.user_id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('interest_id'),
        sa.CheckConstraint("source IN ('manual', 'google_scholar', 'inferred')", name='check_source'),
        sa.CheckConstraint('weight >= 0.0 AND weight <= 1.0', name='check_weight')
    )
    
    op.create_index('idx_user_interests_user', 'user_interests', ['user_id'])
    
    # User profile embeddings table
    op.create_table(
        'user_profile_embeddings',
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('embedding_vector', postgresql.ARRAY(sa.Float()), nullable=False),
        sa.Column('last_updated', sa.TIMESTAMP(), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=False),
        sa.Column('based_on_papers', postgresql.ARRAY(sa.Text()), nullable=True),
        sa.Column('interaction_count', sa.Integer(), server_default=sa.text('0'), nullable=False),
        sa.ForeignKeyConstraint(['user_id'], ['users.user_id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('user_id')
    )
    
    # ==================== Papers & Metadata ====================
    
    # Papers table
    op.create_table(
        'papers',
        sa.Column('paper_id', sa.String(length=100), nullable=False),
        sa.Column('title', sa.Text(), nullable=False),
        sa.Column('authors', postgresql.ARRAY(sa.Text()), nullable=True),
        sa.Column('year', sa.Integer(), nullable=True),
        sa.Column('venue', sa.String(length=255), nullable=True),
        sa.Column('citation_count', sa.Integer(), server_default=sa.text('0'), nullable=False),
        sa.Column('abstract', sa.Text(), nullable=True),
        sa.Column('summary', sa.Text(), nullable=True),
        sa.Column('introduction', sa.Text(), nullable=True),
        sa.Column('gcs_pdf_path', sa.String(length=500), nullable=True),
        sa.Column('domain', sa.String(length=50), nullable=True),
        sa.Column('ingested_at', sa.TIMESTAMP(), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=False),
        sa.Column('updated_at', sa.TIMESTAMP(), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=False),
        sa.PrimaryKeyConstraint('paper_id'),
        sa.CheckConstraint("domain IN ('healthcare', 'fintech', 'quantum_computing')", name='check_paper_domain')
    )
    
    # Create indexes for papers
    op.create_index('idx_papers_domain', 'papers', ['domain'])
    op.create_index('idx_papers_year', 'papers', ['year'])
    op.create_index('idx_papers_citation_count', 'papers', ['citation_count'])
    
    # Full-text search indexes
    op.execute("""
        CREATE INDEX idx_papers_title_gin ON papers 
        USING gin(to_tsvector('english', title))
    """)
    
    op.execute("""
        CREATE INDEX idx_papers_abstract_gin ON papers 
        USING gin(to_tsvector('english', abstract))
    """)
    
    # ==================== User Interactions ====================
    
    # User interactions table
    op.create_table(
        'user_interactions',
        sa.Column('interaction_id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('paper_id', sa.String(length=100), nullable=False),
        sa.Column('interaction_type', sa.String(length=50), nullable=False),
        sa.Column('duration_seconds', sa.Integer(), nullable=True),
        sa.Column('context', postgresql.JSONB(), nullable=True),
        sa.Column('created_at', sa.TIMESTAMP(), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=False),
        sa.ForeignKeyConstraint(['user_id'], ['users.user_id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['paper_id'], ['papers.paper_id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('interaction_id'),
        sa.CheckConstraint(
            "interaction_type IN ('view', 'click', 'save', 'like', 'read_time', 'click_node', 'search')",
            name='check_interaction_type'
        )
    )
    
    # Create indexes for interactions
    op.create_index('idx_interactions_user', 'user_interactions', ['user_id'])
    op.create_index('idx_interactions_paper', 'user_interactions', ['paper_id'])
    op.create_index('idx_interactions_type', 'user_interactions', ['interaction_type'])
    op.create_index('idx_interactions_created', 'user_interactions', ['created_at'])
    
    # JSONB index for context
    op.execute("""
        CREATE INDEX idx_interactions_context_gin ON user_interactions 
        USING gin(context)
    """)
    
    # User saved papers table
    op.create_table(
        'user_saved_papers',
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('paper_id', sa.String(length=100), nullable=False),
        sa.Column('saved_at', sa.TIMESTAMP(), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=False),
        sa.Column('notes', sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(['user_id'], ['users.user_id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['paper_id'], ['papers.paper_id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('user_id', 'paper_id')
    )
    
    op.create_index('idx_saved_papers_user', 'user_saved_papers', ['user_id'])
    op.create_index('idx_saved_papers_saved_at', 'user_saved_papers', ['saved_at'])
    
    # User liked papers table
    op.create_table(
        'user_liked_papers',
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('paper_id', sa.String(length=100), nullable=False),
        sa.Column('liked_at', sa.TIMESTAMP(), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=False),
        sa.ForeignKeyConstraint(['user_id'], ['users.user_id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['paper_id'], ['papers.paper_id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('user_id', 'paper_id')
    )
    
    op.create_index('idx_liked_papers_user', 'user_liked_papers', ['user_id'])
    
    # ==================== Clustering ====================
    
    # Paper clusters table
    op.create_table(
        'paper_clusters',
        sa.Column('cluster_id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('cluster_name', sa.String(length=255), nullable=False),
        sa.Column('theme_description', sa.Text(), nullable=True),
        sa.Column('domain', sa.String(length=50), nullable=True),
        sa.Column('paper_count', sa.Integer(), server_default=sa.text('0'), nullable=False),
        sa.Column('created_at', sa.TIMESTAMP(), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=False),
        sa.Column('expires_at', sa.TIMESTAMP(), nullable=True),
        sa.ForeignKeyConstraint(['user_id'], ['users.user_id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('cluster_id'),
        sa.CheckConstraint("domain IN ('healthcare', 'fintech', 'quantum_computing')", name='check_cluster_domain')
    )
    
    op.create_index('idx_clusters_user', 'paper_clusters', ['user_id'])
    op.create_index('idx_clusters_expires', 'paper_clusters', ['expires_at'])
    
    # Cluster-paper relationships table
    op.create_table(
        'cluster_papers',
        sa.Column('cluster_id', sa.Integer(), nullable=False),
        sa.Column('paper_id', sa.String(length=100), nullable=False),
        sa.Column('centrality_score', sa.Float(), nullable=True),
        sa.Column('is_reference_paper', sa.Boolean(), server_default=sa.text('FALSE'), nullable=False),
        sa.Column('similarity_to_reference', sa.Float(), nullable=True),
        sa.Column('position_x', sa.Float(), nullable=True),
        sa.Column('position_y', sa.Float(), nullable=True),
        sa.ForeignKeyConstraint(['cluster_id'], ['paper_clusters.cluster_id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['paper_id'], ['papers.paper_id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('cluster_id', 'paper_id')
    )
    
    op.create_index('idx_cluster_papers_cluster', 'cluster_papers', ['cluster_id'])
    op.create_index('idx_cluster_papers_paper', 'cluster_papers', ['paper_id'])
    op.create_index('idx_cluster_papers_reference', 'cluster_papers', ['is_reference_paper'])
    
    # ==================== System Tables ====================
    
    # Rate limiting table
    op.create_table(
        'rate_limits',
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('endpoint', sa.String(length=255), nullable=False),
        sa.Column('request_count', sa.Integer(), server_default=sa.text('0'), nullable=False),
        sa.Column('window_start', sa.TIMESTAMP(), nullable=False),
        sa.ForeignKeyConstraint(['user_id'], ['users.user_id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('user_id', 'endpoint', 'window_start')
    )
    
    op.create_index('idx_rate_limits_window', 'rate_limits', ['window_start'])
    
    # System metrics table
    op.create_table(
        'system_metrics',
        sa.Column('metric_id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('metric_name', sa.String(length=100), nullable=False),
        sa.Column('metric_value', sa.Float(), nullable=False),
        sa.Column('metadata', postgresql.JSONB(), nullable=True),
        sa.Column('recorded_at', sa.TIMESTAMP(), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=False),
        sa.PrimaryKeyConstraint('metric_id')
    )
    
    op.create_index('idx_metrics_name', 'system_metrics', ['metric_name'])
    op.create_index('idx_metrics_recorded', 'system_metrics', ['recorded_at'])


def downgrade() -> None:
    """Drop all tables."""
    
    # Drop in reverse order to handle foreign keys
    op.drop_table('system_metrics')
    op.drop_table('rate_limits')
    op.drop_table('cluster_papers')
    op.drop_table('paper_clusters')
    op.drop_table('user_liked_papers')
    op.drop_table('user_saved_papers')
    op.drop_table('user_interactions')
    op.drop_table('papers')
    op.drop_table('user_profile_embeddings')
    op.drop_table('user_interests')
    op.drop_table('user_domains')
    op.drop_table('users')
