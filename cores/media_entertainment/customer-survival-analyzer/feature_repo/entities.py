"""Entity definitions for Customer survival analyzer Feature Store."""

from feast import Entity

# Customer entity for non-contractual (BTYD) features
customer_entity = Entity(
    name="customer",
    join_keys=["customer_id"],
    description="A unique customer identifier for CLTV feature lookups.",
)

# Subscription entity for contractual (survival) features
subscription_entity = Entity(
    name="subscription",
    join_keys=["customer_id"],
    description="A subscription identifier for survival analysis features.",
)
