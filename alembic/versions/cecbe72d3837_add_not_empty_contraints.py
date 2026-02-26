"""add not-empty-contraints

Revision ID: cecbe72d3837
Revises: d2aca3e90cf3
Create Date: 2024-08-05 15:51:55.693719

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'cecbe72d3837'
down_revision: Union[str, None] = 'd2aca3e90cf3'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add CHECK constraints
    op.create_check_constraint(
        'check_serial_not_empty',
        'motor',
        sa.text("serial <> ''")
    )
    op.create_check_constraint(
        'check_part_not_empty',
        'motor',
        sa.text("part <> ''")
    )
    op.create_check_constraint(
        'check_recipe_not_empty',
        'application',
        sa.text("recipe <> ''")
    )
    op.create_check_constraint(
        'check_contextcode_not_empty',
        'application',
        sa.text("context_code <> ''")
    )


def downgrade() -> None:
    # Remove CHECK constraints
    op.drop_constraint('check_serial_not_empty', 'motor', type_='check')
    op.drop_constraint('check_part_not_empty', 'motor', type_='check')
    op.drop_constraint('check_recipe_not_empty', 'application', type_='check')
    op.drop_constraint('check_contextcode_not_empty', 'application', type_='check')
