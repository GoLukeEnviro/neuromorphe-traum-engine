"""Add harmonic_complexity and rhythmic_complexity fields to Stem table

Revision ID: add_musical_complexity
Revises: previous_migration
Create Date: 2025-01-03
"""

from alembic import op
import sqlalchemy as sa


# revision identifiers
revision = 'add_musical_complexity'
down_revision = None  # Update this with the actual previous revision
branch_labels = None
depends_on = None


def upgrade():
    """Add harmonic_complexity and rhythmic_complexity columns to stems table"""
    # Add harmonic_complexity column
    op.add_column('stems', sa.Column('harmonic_complexity', sa.Float(), nullable=True))
    
    # Add rhythmic_complexity column
    op.add_column('stems', sa.Column('rhythmic_complexity', sa.Float(), nullable=True))
    
    # Create indexes for performance
    op.create_index('ix_stems_harmonic_complexity', 'stems', ['harmonic_complexity'])
    op.create_index('ix_stems_rhythmic_complexity', 'stems', ['rhythmic_complexity'])
    
    # Create composite index for musical analysis queries
    op.create_index('ix_stems_musical_complexity', 'stems', ['harmonic_complexity', 'rhythmic_complexity'])


def downgrade():
    """Remove harmonic_complexity and rhythmic_complexity columns from stems table"""
    # Drop indexes first
    op.drop_index('ix_stems_musical_complexity', 'stems')
    op.drop_index('ix_stems_rhythmic_complexity', 'stems')
    op