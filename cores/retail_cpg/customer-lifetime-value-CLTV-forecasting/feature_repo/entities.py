"""Entity definitions for CLTV Forecasting Feature Store."""

from feast import Entity

# Customer entity for RFM features
customer_entity = Entity(
    name="customer",
    join_keys=["customer_id"],
    description="A unique customer identifier for CLTV feature lookups.",
)
