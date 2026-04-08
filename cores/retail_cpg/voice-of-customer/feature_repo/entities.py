"""Entity definitions for Voice of Customer Feature Store."""

from feast import Entity

# Review/interaction entity for VoC features
review_entity = Entity(
    name="review",
    join_keys=["interaction_id"],
    description="A unique review/interaction identifier for VoC feature lookups.",
)
