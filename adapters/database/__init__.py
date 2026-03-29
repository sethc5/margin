"""
Database health adapter for margin.

Connection pool, query performance, replication, and storage health.
Works with any database — thresholds are for the connection layer, not the engine.
"""
from .health import DB_METRICS, parse_db, db_expression
