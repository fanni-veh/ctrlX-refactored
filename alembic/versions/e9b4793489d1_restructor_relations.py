"""Restructor_relations

Revision ID: 55890bcc4ec2
Revises: 96efb20c7140
Create Date: 2025-09-17 13:07:36.067476

"""
from typing import Sequence, Union

from alembic import op
import pandas as pd
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '55890bcc4ec2'
down_revision: Union[str, None] = '96efb20c7140'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def run_migration(conn):
    '''
    Manual migration to reduce state_metadata in Runs
    from a full dataframe to only the relevant information:
    {
        "bad_cycles": [...],
        "good_cycles": [...],
        "confidence_mean": ...,
        "confidence_median": ...,
        "used_total_time_s": ...,
        "used_train_run_id": ...
    }
    '''

    # 1. Load Runs with task prediction and state success
    runs_table = sa.table(
        "run",
        sa.column("id", sa.Integer),
        sa.column("state_metadata", sa.JSON),
        sa.column("cycledata_id", postgresql.ARRAY(sa.Integer)),
        sa.column("user_id", sa.Integer),
        sa.column("task", sa.String),
        sa.column("state", sa.String),
    )

    cycle_table = sa.table(
        "cycledata",
        sa.column("id", sa.Integer),
        sa.column("classification", sa.String),
    )

    stmt = sa.select(
        runs_table.c.id,
        runs_table.c.state_metadata,
        runs_table.c.task,
        runs_table.c.cycledata_id,
    ).where(
        runs_table.c.state == "success",
    )
    results = conn.execute(stmt).fetchall()

    cycle_ids = {cid for r in results for cid in (r[3] or [])}

    if cycle_ids:
        stmt = sa.select(
            cycle_table.c.id,
            cycle_table.c.classification
        ).where(
            cycle_table.c.id.in_(cycle_ids),
        )
        cycle_results = conn.execute(stmt).fetchall()
    else:
        cycle_results = []
    classification_map = {cid: cls for cid, cls in cycle_results}

    # 2. Process each run
    for run_id, state_metadata, task, cycles in results:
        if task == "prediction":
            # Predictions has no classification on cycledata - we need to load from df in state_metadata
            state_metadata = state_metadata or {}
            data = state_metadata.get("data", {})
            data_result = data.get("result", {})
            df = pd.DataFrame(data_result.get('dataframe', {}))
            if df.empty or 'label' not in df.columns or 'cycle_id' not in df.columns:
                raise ValueError(f"Run ID {run_id} has invalid or missing dataframe data.")

            bad_cycles_ids = df[df['label'] == 0]['cycle_id'].unique().tolist()
            good_cycles_ids = df[df['label'] == 1]['cycle_id'].unique().tolist()
            skipped_cycles_ids = [cid for cid in (cycles or []) if cid not in bad_cycles_ids and cid not in good_cycles_ids]

            new_state_metadata = {
                "bad_cycles": bad_cycles_ids,
                "good_cycles": good_cycles_ids,
                "skipped_cycles": skipped_cycles_ids,
                "confidence_mean": data_result.get('confidence_mean'),
                "confidence_median": data_result.get('confidence_median'),
                "used_total_time_s": data.get('used_total_time_s', None),
                "used_train_run_id": data.get('used_train_run_id', None),
            }
        elif task == "train":
            # each cycledata must have a classification
            bad_cycles_ids = [cid for cid in (cycles or []) if classification_map.get(cid) == 'bad']
            good_cycles_ids = [cid for cid in (cycles or []) if classification_map.get(cid) == 'good']
            skipped_cycles_ids = [cid for cid in (cycles or []) if cid not in bad_cycles_ids and cid not in good_cycles_ids]
            state_metadata = state_metadata or {}

            new_state_metadata = {
                "bad_cycles": bad_cycles_ids,
                "good_cycles": good_cycles_ids,
                "skipped_cycles": skipped_cycles_ids,
                "used_total_time_s": state_metadata.get('data', {}).get('used_total_time_s', None),
            }
        else:
            raise ValueError(f"Run ID {run_id} has unknown task '{task}'.")

        # 3. Update Run
        conn.execute(
            runs_table.update()
            .where(runs_table.c.id == run_id)
            .values(
                state_metadata=new_state_metadata,
                user_id=state_metadata.get('user_id')
            )
        )


def upgrade() -> None:
    # Create enum types
    state_enum = sa.Enum('idle', 'running', 'error', 'success', name='run_state')
    task_enum = sa.Enum('prediction', 'train', name='run_task')
    user_role_enum = sa.Enum('admin', 'guest', 'service', 'exhibition', name='user_role')
    classification_enum = sa.Enum('good', 'bad', 'unknown', name='cycle_classification')

    state_enum.create(op.get_bind(), checkfirst=True)
    task_enum.create(op.get_bind(), checkfirst=True)
    user_role_enum.create(op.get_bind(), checkfirst=True)
    classification_enum.create(op.get_bind(), checkfirst=True)

    # --- Data Migration ---
    # 1. Add user_id nullable
    op.add_column('run', sa.Column('user_id', sa.Integer(), nullable=True))
    op.create_foreign_key('run_user_id_fkey', 'run', 'user', ['user_id'], ['id'])
    # 2. Migrate run.state_metadata
    run_migration(op.get_bind())
    # 3. Migrate CycleData.classification
    op.execute("""
    ALTER TABLE cycledata
    ALTER COLUMN classification
    TYPE cycle_classification
    USING CASE
        WHEN classification IN ('good', 'bad') THEN classification::cycle_classification
        ELSE 'unknown'::cycle_classification
    END
    """)
    op.alter_column("cycledata", "classification", nullable=False)

    # --- Create new M:N tables ---
    op.create_table('motor_user',
                    sa.Column('motor_id', sa.Integer(), nullable=False),
                    sa.Column('user_id', sa.Integer(), nullable=False),
                    sa.ForeignKeyConstraint(['motor_id'], ['motor.id'], ondelete='CASCADE'),
                    sa.ForeignKeyConstraint(['user_id'], ['user.id'], ondelete='CASCADE'),
                    sa.PrimaryKeyConstraint('motor_id', 'user_id')
                    )
    op.create_table('cycle_run',
                    sa.Column('cycle_id', sa.Integer(), nullable=False),
                    sa.Column('run_id', sa.UUID(), nullable=False),
                    sa.ForeignKeyConstraint(['cycle_id'], ['cycledata.id'], ondelete='CASCADE'),
                    sa.ForeignKeyConstraint(['run_id'], ['run.id'], ondelete='CASCADE'),
                    sa.PrimaryKeyConstraint('cycle_id', 'run_id')
                    )

    op.create_table('prediction',
                    sa.Column('id', sa.Integer(), nullable=False),
                    sa.Column('run_id', sa.UUID(), nullable=False),
                    sa.Column('model_id', sa.Integer(), nullable=False),
                    sa.Column('cycle_id', sa.Integer(), nullable=False),
                    sa.Column('metrics', sa.String(), nullable=False),
                    sa.Column('value', sa.Float(), nullable=False),
                    sa.ForeignKeyConstraint(['model_id'], ['model.id'], ),
                    sa.ForeignKeyConstraint(['run_id'], ['run.id'], ),
                    sa.PrimaryKeyConstraint('id')
                    )
    op.create_index(op.f('ix_prediction_run_id'), 'prediction', ['run_id'], unique=False)
    # --- Migrate existing 1:1 relationships into M:N --
    conn = op.get_bind()
    # Motor-User
    conn.execute(sa.text("""
        INSERT INTO motor_user (motor_id, user_id)
        SELECT id AS motor_id, user_id
        FROM motor
        WHERE user_id IS NOT NULL
    """))
    # CycleData-Run
    valid_cycle_ids_set = {row[0] for row in conn.execute(sa.text("SELECT id FROM cycledata")).fetchall()}
    results = conn.execute(sa.text("SELECT id, cycledata_id FROM run WHERE cycledata_id IS NOT NULL")).fetchall()
    for run_id, cycle_ids in results:
        if not cycle_ids:
            continue
        # Keep only cycle_ids that actually exist in cycledata
        valid_cycle_ids = [cid for cid in cycle_ids if cid in valid_cycle_ids_set]
        if not valid_cycle_ids:
            print(f"Run {run_id} has no valid cycle_ids; skipping.")
            continue
        for cycle_id in valid_cycle_ids:
            conn.execute(
                sa.text("INSERT INTO cycle_run (cycle_id, run_id) VALUES (:cycle_id, :run_id)"),
                {"cycle_id": cycle_id, "run_id": run_id}
            )

    # Create other DB jobs
    op.create_table('api_key',
                    sa.Column('id', sa.Integer(), nullable=False),
                    sa.Column('user_id', sa.Integer(), nullable=False),
                    sa.Column('access_key', sa.String(), nullable=False),
                    sa.Column('counter', sa.Integer(), server_default='0', nullable=False),
                    sa.Column('role', sa.String(), server_default='read', nullable=False),
                    sa.Column('last_used', sa.TIMESTAMP(timezone=True), nullable=True),
                    sa.Column('expired_at', sa.TIMESTAMP(timezone=True), nullable=True),
                    sa.Column('disabled', sa.Boolean(), server_default='false', nullable=False),
                    sa.Column('time_created', sa.TIMESTAMP(timezone=True), server_default=sa.text('now()'), nullable=False),
                    sa.ForeignKeyConstraint(['user_id'], ['user.id'], ),
                    sa.PrimaryKeyConstraint('id')
                    )
    op.drop_constraint(op.f('motor_user_id_fkey'), 'motor', type_='foreignkey')
    op.drop_column('motor', 'user_id')
    op.alter_column('run', 'task',
                    existing_type=sa.VARCHAR(),
                    type_=task_enum,
                    existing_nullable=False,
                    postgresql_using='task::run_task')
    op.alter_column('run', 'state',
                    existing_type=sa.VARCHAR(),
                    type_=state_enum,
                    existing_nullable=False,
                    postgresql_using='state::run_state')
    op.drop_column('run', 'cycledata_id')
    op.alter_column('user', 'role', server_default=None)
    op.alter_column('user', 'role',
                    existing_type=sa.VARCHAR(),
                    type_=user_role_enum,
                    existing_nullable=False,
                    postgresql_using='role::user_role')
    op.alter_column('user', 'role', server_default='guest')
    # ### end Alembic commands ###


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.alter_column('user', 'role',
                    existing_type=sa.Enum('admin', 'guest', 'service', 'exhibition', name='role'),
                    type_=sa.VARCHAR(),
                    existing_nullable=False,
                    existing_server_default=sa.text("'guest'::character varying"))
    op.add_column('run', sa.Column('cycledata_id', postgresql.ARRAY(sa.INTEGER()), autoincrement=False, nullable=True))
    op.drop_constraint('run_user_id_fkey', 'run', type_='foreignkey')
    op.alter_column('run', 'state',
                    existing_type=sa.Enum('idle', 'running', 'error', 'success', name='state'),
                    type_=sa.VARCHAR(),
                    existing_nullable=False)
    op.alter_column('run', 'task',
                    existing_type=sa.Enum('prediction', 'train', name='task'),
                    type_=sa.VARCHAR(),
                    existing_nullable=False)
    op.drop_column('run', 'user_id')
    op.add_column('motor', sa.Column('user_id', sa.INTEGER(), autoincrement=False, nullable=True))
    op.create_foreign_key(op.f('motor_user_id_fkey'), 'motor', 'user', ['user_id'], ['id'])

    op.execute("""
    ALTER TABLE cycledata
    ALTER COLUMN classification
    TYPE VARCHAR
    USING classification::text
    """)

    op.drop_table('cycle_run')
    op.drop_table('motor_user')
    op.drop_table('api_key')
    op.drop_table('prediction')

    # Drop enum types
    op.execute('DROP TYPE IF EXISTS "run_task"')
    op.execute('DROP TYPE IF EXISTS "run_state"')
    op.execute('DROP TYPE IF EXISTS "user_role"')
    op.execute('DROP TYPE IF EXISTS "cycle_classification"')
    # ### end Alembic commands ###
