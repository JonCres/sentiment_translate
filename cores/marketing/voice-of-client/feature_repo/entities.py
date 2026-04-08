"""Entity definitions for the AI Core Template Feature Store."""

from feast import Entity

# Primary entity for customer-level features
customer_entity = Entity(
    name="customer",
    join_keys=["customer_id"],
    description="A unique customer identifier for feature lookups.",
)
