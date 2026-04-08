"""Visualization pipeline for Voice of Customer analysis."""
from kedro.pipeline import Pipeline, node, pipeline
from .nodes import (
    combine_sentiment_topic_data,
    create_sentiment_distribution_plot,
    create_sentiment_confidence_plot,
    create_sentiment_by_rating_plot,
    create_topic_distribution_plot,
    create_topic_wordclouds,
    create_sentiment_topic_heatmap,
    create_combined_overview_dashboard,
    save_visualizations,
    # Business KPIs
    calculate_business_kpis,
    # AI Interpretation
    interpret_visualization,
    save_interpretations,
)



def create_pipeline(**kwargs) -> Pipeline:
    """Create the visualization pipeline for VoC analysis.
    
    This pipeline creates comprehensive visualizations for:
    - Sentiment analysis results
    - Topic modeling results
    - Combined sentiment-topic insights
    
    Note: Customer-level visualizations (sunburst, timeline, treemap, review volume)
    are generated dynamically in the Streamlit app based on user-selected customers.
    
    Returns:
        A Pipeline object containing visualization nodes.
    """
    return pipeline(
        [
            # Combine sentiment and topic data
            node(
                func=combine_sentiment_topic_data,
                inputs=["sentiment_predictions", "topic_assignments"],
                outputs="combined_sentiment_topics",
                name="combine_sentiment_topic_data_node",
                tags=["visualization", "data_prep"]
            ),
            
            # Sentiment visualizations
            node(
                func=create_sentiment_distribution_plot,
                inputs=["sentiment_predictions", "params:visualization.sentiment_dist"],
                outputs="sentiment_distribution_plot",
                name="create_sentiment_distribution_node",
                tags=["visualization", "sentiment"]
            ),
            node(
                func=create_sentiment_confidence_plot,
                inputs=["sentiment_predictions", "params:visualization.sentiment_confidence"],
                outputs="sentiment_confidence_plot",
                name="create_sentiment_confidence_node",
                tags=["visualization", "sentiment"]
            ),
            node(
                func=create_sentiment_by_rating_plot,
                inputs=["sentiment_predictions", "params:visualization.sentiment_rating"],
                outputs="sentiment_by_rating_plot",
                name="create_sentiment_by_rating_node",
                tags=["visualization", "sentiment"]
            ),
            
            # Topic visualizations
            node(
                func=create_topic_distribution_plot,
                inputs=["topic_assignments", "params:visualization.topic_dist"],
                outputs="topic_distribution_plot",
                name="create_topic_distribution_node",
                tags=["visualization", "topics"]
            ),
            node(
                func=create_topic_wordclouds,
                inputs=["topic_assignments", "topic_model", "params:visualization.topic_wordclouds"],
                outputs="topic_wordclouds_plot",
                name="create_topic_wordclouds_node",
                tags=["visualization", "topics"]
            ),
            
            # Combined visualizations
            node(
                func=create_sentiment_topic_heatmap,
                inputs=["combined_sentiment_topics", "params:visualization.sentiment_topic_heatmap"],
                outputs="sentiment_topic_heatmap_plot",
                name="create_sentiment_topic_heatmap_node",
                tags=["visualization", "combined"]
            ),
            node(
                func=create_combined_overview_dashboard,
                inputs=[
                    "sentiment_predictions",
                    "topic_assignments",
                    "sentiment_summary",
                    "topic_summary",
                    "params:visualization.overview_dashboard"
                ],
                outputs="overview_dashboard_plot",
                name="create_overview_dashboard_node",
                tags=["visualization", "combined", "dashboard"]
            ),
            
            # Save visualizations
            node(
                func=save_visualizations,
                inputs=[
                    "sentiment_distribution_plot",
                    "sentiment_confidence_plot",
                    "sentiment_by_rating_plot",
                    "topic_distribution_plot",
                    "topic_wordclouds_plot",
                    "sentiment_topic_heatmap_plot",
                    "overview_dashboard_plot",
                    "params:visualization.save",
                ],
                outputs=None,
                name="save_visualizations_node",
                tags=["visualization", "output"]
            ),
            
            # AI-powered visualization interpretation
            node(
                func=interpret_visualization,
                inputs=[
                    "combined_sentiment_topics",
                    "sentiment_predictions",
                    "topic_assignments",
                    "customer_sentiment_profiles",
                    "customer_topic_profiles",
                    "params:visualization.interpretation",
                ],
                outputs="visualization_interpretations",
                name="interpret_visualizations_node",
                tags=["visualization", "interpretation", "llm"]
            ),
            node(
                func=save_interpretations,
                inputs=[
                    "visualization_interpretations",
                    "params:visualization.save",
                ],
                outputs=None,
                name="save_interpretations_node",
                tags=["visualization", "interpretation", "output"]
            ),
            
            # Business KPI calculations (Merged from reporting)
            node(
                func=calculate_business_kpis,
                inputs=["unified_interaction_records", "surveys", "params:business_kpis"],
                outputs="business_kpis_summary",
                name="calculate_business_kpis_node",
                tags=["visualization", "reporting", "kpis"]
            ),
        ],
        tags=["visualization_pipeline"]
    )
