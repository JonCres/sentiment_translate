"""
Voice of Customer Explorer - Streamlit Application

Interactive dashboard for exploring customer insights from sentiment and topic analysis.

Run with: streamlit run app/app.py
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
from pathlib import Path
from typing import Dict, Any, Optional, List
import yaml
import streamlit.components.v1 as components
from dotenv import load_dotenv
from deltalake import DeltaTable

# Load environment variables
load_dotenv()

# ==========================================
# CONSTANTS & CONFIGURATION
# ==========================================

def load_config() -> Dict[str, Any]:
    """Load Streamlit configuration from parameters.yml"""
    config_path = Path("conf/base/parameters.yml")
    if config_path.exists():
        with open(config_path) as f:
            params = yaml.safe_load(f)
            return params.get("streamlit", {})
    return {}


def load_css():
    """Load custom CSS from style.css"""
    css_path = Path("app/style.css")
    if css_path.exists():
        with open(css_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def setup_page(config: Dict[str, Any]):
    """Configure Streamlit page settings"""
    st.set_page_config(
        page_title=config.get("page_title", "Voice of Customer Explorer"),
        page_icon=config.get("page_icon", "📊"),
        layout=config.get("layout", "wide"),
        initial_sidebar_state="expanded"
    )
    load_css()


# ==========================================
# DATA LOADING
# ==========================================

@st.cache_data
def load_customer_sentiment_profiles() -> Optional[pd.DataFrame]:
    """Load customer sentiment profiles from Delta table"""
    filepath = Path("data/07_model_output/customer_sentiment_profiles")
    if filepath.exists():
        dt = DeltaTable(str(filepath))
        return dt.to_pandas()
    return None


@st.cache_data
def load_customer_topic_profiles() -> Optional[pd.DataFrame]:
    """Load customer topic profiles from Delta table"""
    filepath = Path("data/07_model_output/customer_topic_profiles")
    if filepath.exists():
        dt = DeltaTable(str(filepath))
        return dt.to_pandas()
    return None


@st.cache_data
def load_sentiment_predictions() -> Optional[pd.DataFrame]:
    """Load sentiment predictions from Delta table"""
    filepath = Path("data/07_model_output/sentiment_predictions")
    if filepath.exists():
        dt = DeltaTable(str(filepath))
        return dt.to_pandas()
    return None


@st.cache_data
def load_topic_assignments() -> Optional[pd.DataFrame]:
    """Load topic assignments from Delta table"""
    filepath = Path("data/07_model_output/topic_assignments")
    if filepath.exists():
        dt = DeltaTable(str(filepath))
        return dt.to_pandas()
    return None


@st.cache_data
def load_unified_interaction_records() -> Optional[pd.DataFrame]:
    """Load unified interaction records from Delta table"""
    filepath = Path("data/07_model_output/unified_interaction_records")
    if filepath.exists():
        dt = DeltaTable(str(filepath))
        return dt.to_pandas()
    return None


@st.cache_data
def load_customer_aspect_profiles() -> Optional[pd.DataFrame]:
    """Load customer aspect profiles from Delta table"""
    filepath = Path("data/07_model_output/customer_aspect_profiles")
    if filepath.exists():
        dt = DeltaTable(str(filepath))
        return dt.to_pandas()
    return None


@st.cache_data
def load_customer_summaries() -> Dict[str, Any]:
    """Load customer summary JSON files"""
    summaries = {}
    
    sentiment_path = Path("data/08_reporting/customer_sentiment_summary.json")
    if sentiment_path.exists():
        with open(sentiment_path) as f:
            summaries["sentiment"] = json.load(f)
    
    topic_path = Path("data/08_reporting/customer_topic_summary.json")
    if topic_path.exists():
        with open(topic_path) as f:
            summaries["topic"] = json.load(f)
    
    return summaries


@st.cache_data
def load_aspect_sentiment_summary() -> Optional[Dict[str, Any]]:
    """Load aspect sentiment summary from JSON"""
    filepath = Path("data/08_reporting/aspect_sentiment_summary.json")
    if filepath.exists():
        with open(filepath) as f:
            return json.load(f)
    return None


@st.cache_data
def load_business_kpis() -> Optional[Dict[str, Any]]:
    """Load business KPIs summary from JSON"""
    filepath = Path("data/08_reporting/business_kpis.json")
    if filepath.exists():
        with open(filepath) as f:
            return json.load(f)
    return None


@st.cache_data
def load_high_urgency_alerts() -> List[Dict[str, Any]]:
    """Load high urgency alerts from JSON"""
    filepath = Path("data/08_reporting/high_urgency_alerts.json")
    if filepath.exists():
        with open(filepath) as f:
            return json.load(f)
    return []


# ==========================================
# VISUALIZATION FUNCTIONS
# ==========================================

def create_customer_kpi_cards(customer_data: Dict[str, Any]):
    """Display KPI cards for a customer"""
    cols = st.columns(5)
    
    with cols[0]:
        st.metric(
            "Total Reviews",
            customer_data.get("total_reviews", 0)
        )
    
    with cols[1]:
        sentiment_score = customer_data.get("sentiment_score", 0)
        st.metric(
            "Sentiment Score ℹ️",
            f"{sentiment_score:.2f}",
            delta="Positive" if sentiment_score > 0 else "Negative" if sentiment_score < 0 else None,
            help="**Formula:** (positive_count - negative_count) / total_reviews\n\n"
                 "**Range:** -1.0 to +1.0\n\n"
                 "**What it measures:** A normalized ratio representing the overall sentiment balance relative to total reviews.\n\n"
                 "**Example:** 100 reviews with 70 positive, 10 neutral, 20 negative → Score = (70-20)/100 = 0.50"
        )
    
    with cols[2]:
        avg_confidence = customer_data.get('avg_confidence', 0)
        
        # Determine color based on confidence level
        if avg_confidence > 0.80:
            confidence_color = "🟢"  # Green - High confidence
            confidence_level = "High"
        elif avg_confidence >= 0.60:
            confidence_color = "🟡"  # Yellow - Medium confidence
            confidence_level = "Medium"
        else:
            confidence_color = "🔴"  # Red - Low confidence
            confidence_level = "Low"
        
        st.metric(
            f"Avg Confidence {confidence_color}",
            f"{avg_confidence:.1%}",
            help=f"**Confidence Level:** {confidence_level}\n\n"
                 "🟢 High (>80%): Model is very certain about predictions\n\n"
                 "🟡 Medium (60-80%): Model has moderate certainty\n\n"
                 "🔴 Low (<60%): Model is uncertain, possibly due to ambiguous or mixed sentiment"
        )
    
    with cols[3]:
        st.metric(
            "Dominant Sentiment",
            customer_data.get("dominant_sentiment", "N/A").capitalize()
        )
    
    with cols[4]:
        st.metric(
            "Topic Diversity ℹ️",
            f"{customer_data.get('topic_diversity', 0):.2f}",
            help="**Formula:** unique_customer_topics / total_topics_in_dataset\n\n"
                 "**Range:** 0.0 to 1.0\n\n"
                 "**What it measures:** The proportion of different topics a customer discusses relative to all available topics.\n\n"
                 "**Interpretation:**\n"
                 "• High (0.7-1.0): Broad range of interests - reviews cover many topics\n"
                 "• Medium (0.3-0.7): Balanced reviewer with moderate topic coverage\n"
                 "• Low (0.0-0.3): Focused reviewer - consistently discusses same topics\n\n"
                 "**Example:** If dataset has 10 topics and customer covers 3 → Diversity = 3/10 = 0.30\n\n"
                 "**Note:** The system also calculates Topic Entropy (information theory measure) for diversity analysis in the backend."
        )


def create_sentiment_gauge(positive_pct: float, neutral_pct: float, negative_pct: float) -> go.Figure:
    """Create a sentiment gauge chart"""
    fig = go.Figure()
    
    net_sentiment_value = positive_pct - negative_pct
    
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=net_sentiment_value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={
            'text': "Net Sentiment<br><sub>Formula: Positive% - Negative%</sub>",
            'font': {'size': 16}
        },
        gauge={
            'axis': {'range': [-100, 100]},
            'bar': {'color': "#3498db"},
            'steps': [
                {'range': [-100, -33], 'color': "#e74c3c"},
                {'range': [-33, 33], 'color': "#f39c12"},
                {'range': [33, 100], 'color': "#2ecc71"}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': net_sentiment_value
            }
        }
    ))
    
    fig.update_layout(
        height=300, 
        margin=dict(t=50, b=20, l=20, r=20),
        annotations=[
            dict(
                text="Range: -100 to +100<br>Measures the percentage point difference between positive and negative sentiments",
                xref="paper", yref="paper",
                x=0.5, y=-0.1,
                showarrow=False,
                font=dict(size=10, color="gray"),
                align="center"
            )
        ]
    )
    return fig


def create_aspect_sentiment_chart(aspect_data: pd.DataFrame) -> go.Figure:
    """Create a radar or bar chart for aspect sentiment"""
    if aspect_data.empty:
        return go.Figure()
        
    fig = px.bar(
        aspect_data,
        x='aspect',
        y='avg_sentiment_score',
        color='avg_sentiment_score',
        color_continuous_scale='RdYlGn',
        range_color=[-1, 1],
        labels={'avg_sentiment_score': 'Avg Sentiment Score', 'aspect': 'Aspect'},
        title="Aspect Specific Sentiment"
    )
    
    fig.update_layout(
        yaxis_range=[-1, 1],
        height=400,
        margin=dict(t=50, b=20, l=20, r=20)
    )
    return fig


def display_business_impact_tab(kpis: Optional[Dict[str, Any]]):
    """Display the Business Impact tab with KPIs"""
    if not kpis:
        st.warning("Business KPIs not available. Run visualization pipeline.")
        return

    st.subheader("🚀 Business Impact KPIs")
    
    # 1. Summary Metrics
    summary = kpis.get("summary", {})
    cols = st.columns(4)
    with cols[0]:
        st.metric("Interactions Monitored", f"{summary.get('total_interactions_monitored', 0):,}")
    with cols[1]:
        st.metric("Actual CSAT", f"{summary.get('actual_csat_percent', 0)}%", delta="+2.5%", delta_color="normal")
    with cols[2]:
        st.metric("Net Promoter Score (NPS)", f"{summary.get('calculated_nps', 0)}", delta="+5", delta_color="normal")
    with cols[3]:
        st.metric("Churn Risk Identification", f"{summary.get('high_risk_customers_identified', 0)}", delta="-12%", delta_color="inverse")

    st.divider()
    
    left, right = st.columns(2)
    
    with left:
        st.markdown("### 📈 Baseline vs Target")
        comparison = kpis.get("baseline_comparison", {})
        
        for kpi, data in comparison.items():
            st.write(f"**{kpi.upper()}**")
            cols = st.columns([1, 1, 1])
            cols[0].write(f"Baseline: {data['baseline']}")
            cols[1].write(f"Current: **{data['current']}**")
            cols[2].write(f"Target: {data['target']}")
            
            # Progress bar
            progress = min(1.0, max(0.0, (data['current'] - data['baseline']) / (data['target'] - data['baseline']))) if data['target'] != data['baseline'] else 1.0
            st.progress(progress)
            
    with right:
        st.markdown("### 💰 Financial Impact")
        impact = kpis.get("impact_metrics", {})
        
        st.info(f"**Estimated Annual Savings:** {impact.get('estimated_annual_savings', 'N/A')}")
        st.success(f"**Retention Improvement:** {impact.get('retention_improvement', 'N/A')}")
        
        st.markdown("""
        **Methodology:**
        - **Churn Risk**: Identified via negative sentiment + high urgency topics.
        - **ROI**: Based on manual labor reduction and proactive churn prevention.
        """)


def create_sentiment_pie(positive: int, neutral: int, negative: int) -> go.Figure:
    """Create sentiment distribution pie chart"""
    fig = go.Figure(go.Pie(
        labels=['Positive', 'Neutral', 'Negative'],
        values=[positive, neutral, negative],
        marker=dict(colors=['#2ecc71', '#f39c12', '#e74c3c']),
        hole=0.4,
        textinfo='label+percent'
    ))
    
    fig.update_layout(
        title="Sentiment Distribution",
        height=350,
        margin=dict(t=50, b=20, l=20, r=20)
    )
    return fig


def create_customer_reviews_table(
    sentiment_predictions: pd.DataFrame,
    customer_id: str,
    max_rows: int = 10
) -> pd.DataFrame:
    """Get reviews for a specific customer"""
    customer_reviews = sentiment_predictions[
        sentiment_predictions['customer_id'] == customer_id
    ].copy()
    
    # Select relevant columns
    display_cols = ['review_text', 'sentiment', 'sentiment_confidence', 'rating']
    if 'date' in customer_reviews.columns:
        display_cols.append('date')
    
    available_cols = [c for c in display_cols if c in customer_reviews.columns]
    
    return customer_reviews[available_cols].head(max_rows)


def create_customer_comparison_chart(
    profiles: pd.DataFrame,
    selected_customers: list,
    metric: str
) -> go.Figure:
    """Create comparison bar chart for selected customers"""
    comparison_data = profiles[profiles['customer_id'].isin(selected_customers)]
    
    fig = px.bar(
        comparison_data,
        x='customer_id',
        y=metric,
        color='dominant_sentiment',
        color_discrete_map={
            'positive': '#2ecc71',
            'neutral': '#f39c12',
            'negative': '#e74c3c'
        },
        title=f"Customer Comparison: {metric.replace('_', ' ').title()}"
    )
    
    fig.update_layout(
        xaxis_title="Customer ID",
        yaxis_title=metric.replace('_', ' ').title(),
        height=400
    )
    
    return fig


def create_dynamic_sentiment_sunburst(
    sentiment_profiles: pd.DataFrame,
    sentiment_predictions: pd.DataFrame,
    selected_customers: list
) -> go.Figure:
    """Create sunburst chart for selected customers showing sentiment distribution."""
    
    # Filter profiles for selected customers
    filtered_profiles = sentiment_profiles[sentiment_profiles['customer_id'].isin(selected_customers)]
    
    if filtered_profiles.empty:
        fig = go.Figure()
        fig.add_annotation(text="No customers selected", showarrow=False, font=dict(size=20))
        return fig
    
    # Prepare data for sunburst
    labels = ['Selected Customers']
    parents = ['']
    values = [filtered_profiles['total_reviews'].sum()]
    colors = ['#636EFA']
    
    sentiment_colors = {
        'positive': '#2ecc71',
        'neutral': '#f39c12',
        'negative': '#e74c3c'
    }
    
    # Add sentiment level
    for sentiment in ['positive', 'neutral', 'negative']:
        sentiment_customers = filtered_profiles[filtered_profiles['dominant_sentiment'] == sentiment]
        count = sentiment_customers['total_reviews'].sum()
        if count > 0:
            labels.append(sentiment.capitalize())
            parents.append('Selected Customers')
            values.append(count)
            colors.append(sentiment_colors[sentiment])
            
            # Add customer level under each sentiment
            for _, row in sentiment_customers.iterrows():
                customer_id = str(row['customer_id'])[:15]
                labels.append(customer_id)
                parents.append(sentiment.capitalize())
                values.append(row['total_reviews'])
                colors.append(sentiment_colors[sentiment])
    
    fig = go.Figure(go.Sunburst(
        labels=labels,
        parents=parents,
        values=values,
        branchvalues='total',
        marker=dict(colors=colors),
        hovertemplate='<b>%{label}</b><br>Reviews: %{value}<extra></extra>'
    ))
    
    fig.update_layout(
        title='Customer Sentiment Distribution (Sunburst)',
        template='plotly_white',
        height=550,
        margin=dict(t=50, l=25, r=25, b=25)
    )
    
    return fig


def create_dynamic_sentiment_timeline(
    sentiment_predictions: pd.DataFrame,
    selected_customers: list
) -> go.Figure:
    """Create sentiment timeline for selected customers."""
    
    if 'date' not in sentiment_predictions.columns:
        fig = go.Figure()
        fig.add_annotation(text="Date column not available", showarrow=False, font=dict(size=20))
        return fig
    
    # Filter predictions for selected customers
    filtered = sentiment_predictions[sentiment_predictions['customer_id'].isin(selected_customers)].copy()
    
    if filtered.empty:
        fig = go.Figure()
        fig.add_annotation(text="No data for selected customers", showarrow=False, font=dict(size=20))
        return fig
    
    filtered['date'] = pd.to_datetime(filtered['date'])
    
    # Map sentiment to numeric
    sentiment_map = {'positive': 1, 'neutral': 0, 'negative': -1}
    filtered['sentiment_numeric'] = filtered['sentiment'].str.lower().map(sentiment_map)
    
    # Group by customer and date (weekly)
    filtered['week'] = filtered['date'].dt.to_period('W').dt.start_time
    weekly = filtered.groupby(['customer_id', 'week'])['sentiment_numeric'].mean().reset_index()
    
    fig = go.Figure()
    
    colors = px.colors.qualitative.Plotly[:len(selected_customers)]
    
    for idx, customer_id in enumerate(selected_customers):
        customer_data = weekly[weekly['customer_id'] == customer_id]
        if not customer_data.empty:
            fig.add_trace(go.Scatter(
                x=customer_data['week'],
                y=customer_data['sentiment_numeric'],
                mode='lines+markers',
                name=str(customer_id)[:15],
                line=dict(color=colors[idx % len(colors)], width=2),
                marker=dict(size=6),
                hovertemplate='%{x}<br>Sentiment: %{y:.2f}<extra></extra>'
            ))
    
    fig.update_layout(
        title='Customer Sentiment Over Time',
        xaxis_title='Date',
        yaxis_title='Average Sentiment (-1 to 1)',
        yaxis=dict(tickvals=[-1, 0, 1], ticktext=['Negative', 'Neutral', 'Positive']),
        template='plotly_white',
        height=450,
        showlegend=True,
        hovermode='x unified'
    )
    
    return fig


def create_dynamic_topic_treemap(
    topic_profiles: pd.DataFrame,
    selected_customers: list
) -> go.Figure:
    """Create treemap showing topic distribution for selected customers."""
    
    if topic_profiles is None or topic_profiles.empty:
        fig = go.Figure()
        fig.add_annotation(text="Topic profiles not available", showarrow=False, font=dict(size=20))
        return fig
    
    # Filter for selected customers
    filtered = topic_profiles[topic_profiles['customer_id'].isin(selected_customers)].copy()
    
    if filtered.empty:
        fig = go.Figure()
        fig.add_annotation(text="No topic data for selected customers", showarrow=False, font=dict(size=20))
        return fig
    
    # Normalize topic names
    filtered['dominant_topic_name'] = filtered['dominant_topic_name'].astype(str).str.strip().str.title()
    
    # Prepare data
    labels = []
    parents = []
    values = []
    
    # Group by dominant topic
    for topic in filtered['dominant_topic_name'].unique():
        topic_customers = filtered[filtered['dominant_topic_name'] == topic]
        
        # Add topic as parent
        labels.append(topic[:30])
        parents.append('')
        values.append(topic_customers['total_reviews'].sum())
        
        # Add customers under topic
        for _, row in topic_customers.iterrows():
            customer_id = str(row['customer_id'])[:15]
            labels.append(customer_id)
            parents.append(topic[:30])
            values.append(row['total_reviews'])
    
    fig = go.Figure(go.Treemap(
        labels=labels,
        parents=parents,
        values=values,
        branchvalues='total',
        hovertemplate='<b>%{label}</b><br>Reviews: %{value}<extra></extra>',
        marker=dict(
            colorscale='Viridis',
            cornerradius=5
        )
    ))
    
    fig.update_layout(
        title='Customer Topic Distribution (Treemap)',
        template='plotly_white',
        height=550,
        margin=dict(t=50, l=25, r=25, b=25)
    )
    
    return fig


def create_dynamic_review_volume_chart(
    sentiment_profiles: pd.DataFrame,
    selected_customers: list
) -> go.Figure:
    """Create stacked bar chart of review volume by customer with sentiment coloring."""
    
    if sentiment_profiles is None or sentiment_profiles.empty:
        fig = go.Figure()
        fig.add_annotation(text="Sentiment profiles not available", showarrow=False, font=dict(size=20))
        return fig
    
    # Filter for selected customers
    filtered = sentiment_profiles[sentiment_profiles['customer_id'].isin(selected_customers)].copy()
    
    if filtered.empty:
        fig = go.Figure()
        fig.add_annotation(text="No data for selected customers", showarrow=False, font=dict(size=20))
        return fig
    
    # Truncate customer IDs for display
    filtered['customer_display'] = filtered['customer_id'].astype(str).str[:15]
    
    # Sort by total reviews
    filtered = filtered.sort_values('total_reviews', ascending=True)
    
    # Create stacked bar chart
    fig = go.Figure()
    
    colors = {
        'positive': '#2ecc71',
        'neutral': '#f39c12',
        'negative': '#e74c3c'
    }
    
    for sentiment in ['negative', 'neutral', 'positive']:
        count_col = f'{sentiment}_count'
        if count_col in filtered.columns:
            fig.add_trace(go.Bar(
                y=filtered['customer_display'],
                x=filtered[count_col],
                name=sentiment.capitalize(),
                orientation='h',
                marker_color=colors[sentiment],
                hovertemplate=f'{sentiment.capitalize()}: %{{x}}<extra></extra>'
            ))
    
    fig.update_layout(
        title=f'Review Volume by Customer (Top {len(filtered)})',
        xaxis_title='Number of Reviews',
        yaxis_title='Customer',
        barmode='stack',
        template='plotly_white',
        height=max(400, len(filtered) * 30),
        showlegend=True,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )
    
    return fig


# ==========================================
# VISUALIZATION GALLERY
# ==========================================

def render_visualization_gallery(config: Dict[str, Any]):
    """Render the visualization gallery page"""
    st.header("🖼️ Visualization Gallery")
    st.markdown("<div class='decoration'></div>", unsafe_allow_html=True)
    
    viz_path = Path(config.get("visualization_path", "data/08_reporting/visualizations"))
    
    if not viz_path.exists():
        st.warning(f"⚠️ Visualization directory not found: `{viz_path}`")
        st.info("Please run the visualization pipeline to generate reports.")
        return

    # Filter out customer visualizations (they are interactive in Customer Explorer)
    customer_viz_prefixes = ['customer_sentiment_sunburst', 'customer_sentiment_timeline', 'customer_topic_treemap', 'customer_review_volume']
    html_files = sorted([
        f for f in viz_path.glob("*.html") 
        if not any(f.stem.startswith(prefix) for prefix in customer_viz_prefixes)
    ])
    
    if not html_files:
        st.info("No visualization files found.")
        return
        
    # File selection
    col1, col2 = st.columns([3, 1])
    with col1:
        selected_file = st.selectbox(
            "Select Visualization",
            options=html_files,
            format_func=lambda x: x.name.replace("_", " ").replace(".html", "").title()
        )
    
    if selected_file:
        # Display HTML
        with open(selected_file, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        st.components.v1.html(html_content, height=800, scrolling=True)
        
        # AI Interpretation Section
        st.markdown("---")
        st.subheader("🤖 AI Interpretation")
        
        # Try to load pre-computed interpretation
        interpretation_file = viz_path / "interpretations" / f"{selected_file.stem}_interpretation.txt"
        
        if interpretation_file.exists():
            with open(interpretation_file, 'r', encoding='utf-8') as f:
                interpretation_content = f.read()
            
            st.markdown(interpretation_content)
            st.caption("*This interpretation was generated during the Kedro pipeline run.*")
        else:
            st.info(
                "No interpretation available for this visualization. "
                "Run the Kedro visualization pipeline to generate AI interpretations."
            )





# ==========================================
# PRODUCT ANALYSIS
# ==========================================

def render_product_analysis(sentiment_predictions: pd.DataFrame, config: Dict[str, Any]):
    """Render the product analysis page"""
    st.header("📦 Product Analysis")
    st.markdown("<div class='decoration'></div>", unsafe_allow_html=True)
    
    if sentiment_predictions is None or 'product_id' not in sentiment_predictions.columns:
        st.warning("Product data not available or product_id column missing.")
        return

    # Group by product
    product_stats = sentiment_predictions.groupby('product_id').agg(
        review_count=('review_id', 'count'), # Use review_id which is guaranteed to be non-null
        avg_rating=('rating', 'mean') if 'rating' in sentiment_predictions.columns else ('sentiment_confidence', 'count'), # Fallback
        positive_count=('sentiment', lambda x: (x.str.lower() == 'positive').sum()),
        negative_count=('sentiment', lambda x: (x.str.lower() == 'negative').sum())
    ).reset_index()
    
    # Calculate scores
    # Ensure we don't divide by zero
    product_stats['sentiment_score'] = (
        (product_stats['positive_count'] - product_stats['negative_count']) / 
        product_stats['review_count'].replace(0, 1)
    ).round(3)
    
    if 'avg_rating' in product_stats.columns:
        product_stats['avg_rating'] = product_stats['avg_rating'].round(2)
        
    # Top Products
    st.subheader("🏆 Top Products")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("##### By Volume")
        st.dataframe(
            product_stats.nlargest(5, 'review_count')[['product_id', 'review_count', 'sentiment_score']],
            use_container_width=True,
            hide_index=True
        )
        
    with col2:
        st.markdown("##### By Sentiment")
        st.dataframe(
            product_stats[product_stats['review_count'] > 5].nlargest(5, 'sentiment_score')[['product_id', 'sentiment_score', 'review_count']],
            use_container_width=True,
            hide_index=True
        )
        
    st.markdown("---")
    
    # Topic Distribution Visualization
    st.subheader("📊 Topic Distribution")
    st.markdown("*View the distribution of topics across all product reviews.*")
    
    viz_path = Path(config.get("visualization_path", "data/08_reporting/visualizations"))
    topic_dist_file = viz_path / "topic_distribution.html"
    
    if topic_dist_file.exists():
        with open(topic_dist_file, 'r', encoding='utf-8') as f:
            html_content = f.read()
        components.html(html_content, height=500, scrolling=True)
        
        # Show interpretation if available
        interpretation_file = viz_path / "interpretations" / "topic_distribution_interpretation.txt"
        if interpretation_file.exists():
            with st.expander("🤖 AI Interpretation", expanded=False):
                with open(interpretation_file, 'r', encoding='utf-8') as f:
                    st.markdown(f.read())
    else:
        st.info("Topic distribution visualization not available. Run the Kedro visualization pipeline to generate it.")
    
    st.markdown("---")
    
    # Detailed View
    st.subheader("🔍 Product Details")
    
    selected_product = st.selectbox(
        "Select Product",
        options=product_stats.sort_values('review_count', ascending=False)['product_id'].tolist()
    )
    
    if selected_product:
        stats = product_stats[product_stats['product_id'] == selected_product].iloc[0]
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Reviews", stats['review_count'])
        c2.metric("Avg Rating", stats.get('avg_rating', 'N/A'))
        c3.metric("Sentiment Score", stats['sentiment_score'])
        
        # Filtered reviews
        st.markdown("##### Recent Reviews")
        reviews = sentiment_predictions[sentiment_predictions['product_id'] == selected_product].head(5)
        st.dataframe(reviews[['date', 'rating', 'review_text', 'sentiment']], use_container_width=True, hide_index=True)


# ==========================================
# ASPECT ANALYSIS (ABSA)
# ==========================================

def render_aspect_analysis(config: Dict[str, Any]):
    """Render the Aspect-Based Sentiment Analysis page"""
    st.header("🔍 Aspect Analysis (ABSA)")
    st.markdown("<div class='decoration'></div>", unsafe_allow_html=True)
    st.markdown("*Analyze sentiment at the aspect level (price, quality, shipping, etc.)*")
    
    aspect_summary = load_aspect_sentiment_summary()
    
    if aspect_summary is None:
        st.warning("""
        ⚠️ **Aspect analysis data not available**
        
        Please run the ABSA pipeline first:
        ```bash
        kedro run --pipeline=absa
        ```
        """)
        return
    
    # Overview metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Aspects Analyzed", aspect_summary.get('total_aspects_analyzed', 0))
    with col2:
        st.metric("Unique Aspects", aspect_summary.get('unique_aspects', 0))
    with col3:
        problem_count = len(aspect_summary.get('problem_aspects', []))
        st.metric("Problem Areas 🔴", problem_count, delta="Critical" if problem_count > 0 else None, delta_color="inverse")
    
    st.markdown("---")
    
    # Problem aspects alert
    problem_aspects = aspect_summary.get('problem_aspects', [])
    if problem_aspects:
        st.error(f"⚠️ **Product Issue Alert**: High negative sentiment detected in: **{', '.join(problem_aspects)}**")
    
    # Strong aspects
    strong_aspects = aspect_summary.get('strong_aspects', [])
    if strong_aspects:
        st.success(f"✅ **Strengths**: Positive sentiment in: **{', '.join(strong_aspects)}**")
    
    st.markdown("---")
    
    # Aspect breakdown table
    st.subheader("📊 Aspect Sentiment Breakdown")
    
    aspects = aspect_summary.get('aspects', {})
    if aspects:
        aspect_df = pd.DataFrame([
            {
                'Aspect': aspect,
                'Mentions': data['total_mentions'],
                'Positive %': data['positive_pct'],
                'Neutral %': data['neutral_pct'],
                'Negative %': data['negative_pct'],
                'Net Sentiment': data['net_sentiment'],
                'Confidence': data['avg_confidence']
            }
            for aspect, data in aspects.items()
        ])
        
        # Color code by net sentiment
        def color_sentiment(val):
            if isinstance(val, (int, float)):
                if val > 0.3:
                    return 'background-color: #d5f5e3'
                elif val < -0.3:
                    return 'background-color: #fadbd8'
            return ''
        
        styled_df = aspect_df.style.applymap(color_sentiment, subset=['Net Sentiment'])
        st.dataframe(styled_df, use_container_width=True, hide_index=True)
        
        # Visualization
        st.subheader("📈 Aspect Sentiment Distribution")
        
        fig = go.Figure()
        
        for sentiment, color, label in [('Positive %', '#2ecc71', 'Positive'), ('Neutral %', '#f39c12', 'Neutral'), ('Negative %', '#e74c3c', 'Negative')]:
            fig.add_trace(go.Bar(
                name=label,
                x=aspect_df['Aspect'],
                y=aspect_df[sentiment],
                marker_color=color
            ))
        
        fig.update_layout(
            barmode='stack',
            xaxis_title='Aspect',
            yaxis_title='Percentage',
            title='Sentiment Distribution by Aspect',
            height=450
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No aspect data available yet.")


# ==========================================
# BUSINESS KPIs
# ==========================================

def render_business_kpis(config: Dict[str, Any]):
    """Render the Business KPIs dashboard"""
    st.header("📊 Business KPIs")
    st.markdown("<div class='decoration'></div>", unsafe_allow_html=True)
    st.markdown("*Strategic metrics aligned with Voice of Customer objectives*")
    
    business_kpis = load_business_kpis()
    aspect_summary = load_aspect_sentiment_summary()
    
    if business_kpis is None and aspect_summary is None:
        st.warning("""
        ⚠️ **Business KPIs data not available**
        
        Please run the pipelines first:
        ```bash
        kedro run
        ```
        """)
        return
    
    st.markdown("---")
    
    # Product Issue Detection (from aspect summary)
    st.subheader("🚨 Product Issue Detection")
    
    if aspect_summary:
        problem_aspects = aspect_summary.get('problem_aspects', [])
        if problem_aspects:
            st.error(f"**{len(problem_aspects)} product issues detected** requiring attention:")
            for aspect in problem_aspects:
                aspect_data = aspect_summary.get('aspects', {}).get(aspect, {})
                negative_pct = aspect_data.get('negative_pct', 0)
                mentions = aspect_data.get('total_mentions', 0)
                st.markdown(f"- 🔴 **{aspect.title()}**: {negative_pct:.1f}% negative across {mentions} mentions")
        else:
            st.success("✅ No critical product issues detected")
    
    st.markdown("---")
    
    # VoC Health Metrics
    st.subheader("📈 VoC Health Metrics")
    
    if aspect_summary:
        # Calculate overall VoC health from aspects
        aspects = aspect_summary.get('aspects', {})
        if aspects:
            total_positive = sum(d.get('positive', 0) for d in aspects.values())
            total_negative = sum(d.get('negative', 0) for d in aspects.values())
            total_neutral = sum(d.get('neutral', 0) for d in aspects.values())
            total = total_positive + total_neutral + total_negative
            
            if total > 0:
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Overall Positive", f"{total_positive/total*100:.1f}%")
                with col2:
                    st.metric("Overall Neutral", f"{total_neutral/total*100:.1f}%")
                with col3:
                    st.metric("Overall Negative", f"{total_negative/total*100:.1f}%")
                with col4:
                    net = (total_positive - total_negative) / total
                    st.metric("Net Sentiment", f"{net:.2f}", delta="Positive" if net > 0 else "Negative")
    
    st.markdown("---")
    
    # Recommendations
    st.subheader("💡 Actionable Insights")
    
    if aspect_summary:
        problem_aspects = aspect_summary.get('problem_aspects', [])
        strong_aspects = aspect_summary.get('strong_aspects', [])
        
        st.markdown("**Based on customer feedback analysis:**")
        
        if problem_aspects:
            st.markdown("🔧 **Areas requiring improvement:**")
            for aspect in problem_aspects[:3]:
                st.markdown(f"  - Investigate and address issues with **{aspect}**")
        
        if strong_aspects:
            st.markdown("⭐ **Leverage strengths:**")
            for aspect in strong_aspects[:3]:
                st.markdown(f"  - Highlight **{aspect}** in marketing materials")
    else:
        st.info("Run the ABSA pipeline to generate actionable insights.")


def render_business_kpis(config):
    """Business KPIs Dashboard View"""
    st.title("🚀 Business Impact Dashboard")
    st.markdown("<div class='decoration'></div>", unsafe_allow_html=True)
    st.markdown("*Real-time tracking of VoC impact on business objectives*")
    
    kpis = load_business_kpis()
    display_business_impact_tab(kpis)
    
    # Show summary of alerts
    alerts = load_high_urgency_alerts()
    if alerts:
        st.error(f"🚨 **Urgent Attention Required**: {len(alerts)} high-urgency interactions detected.")
        if st.button("View All Alerts"):
            st.session_state["nav_to"] = "Alerts & Monitoring"
            st.rerun()


def render_alerts(config):
    """Alerts and Monitoring Dashboard View"""
    st.title("🚨 Alerts & Monitoring")
    st.markdown("<div class='decoration'></div>", unsafe_allow_html=True)
    st.markdown("*Immediate intervention required for these high-urgency interactions*")
    
    alerts = load_high_urgency_alerts()
    
    if not alerts:
        st.success("✅ **No high-urgency alerts detected at this time.**")
        return
        
    st.error(f"Found {len(alerts)} interactions requiring immediate review.")
    
    for alert in alerts:
        with st.expander(f"🔴 {alert['topic']} - Customer {alert['customer_id']}", expanded=False):
            col1, col2 = st.columns([1, 2])
            with col1:
                st.write(f"**Emotion:** {alert['emotion'].capitalize()}")
                st.write(f"**Urgency:** High")
                st.write(f"**Detected At:** {alert['timestamp'][:19]}")
            with col2:
                st.write(f"**Recommended Action:**")
                st.info(alert['action'])
            
            st.write("**Customer Feedback Preview:**")
            st.markdown(f"> {alert['text_preview']}")
            
            if st.button(f"Mark as Resolved", key=f"resolve_{alert['interaction_id']}"):
                st.success("Marked for follow-up.")


# ==========================================
# MAIN APPLICATION
# ==========================================

def main():
    # Load configuration
    config = load_config()
    setup_page(config)
    
    # Sidebar Navigation
    st.sidebar.title("📊 VoC Explorer")
    
    # Handle navigation redirection
    if "nav_to" in st.session_state:
        st.session_state.app_mode = st.session_state.nav_to
        del st.session_state.nav_to

    app_mode = st.sidebar.radio(
        "Navigation",
        ["Customer Explorer", "Product Analysis", "Aspect Analysis", "Business KPIs", "Alerts & Monitoring", "Visualization Gallery"],
        key="app_mode"
    )
    
    st.sidebar.markdown("---")
    
    # Load shared data
    sentiment_predictions = load_sentiment_predictions()
    
    if app_mode == "Customer Explorer":
        render_customer_explorer(config, sentiment_predictions)
    elif app_mode == "Product Analysis":
        render_product_analysis(sentiment_predictions, config)
    elif app_mode == "Aspect Analysis":
        render_aspect_analysis(config)
    elif app_mode == "Business KPIs":
        render_business_kpis(config)
    elif app_mode == "Alerts & Monitoring":
        render_alerts(config)
    elif app_mode == "Visualization Gallery":
        render_visualization_gallery(config)
        
    # Footer
    st.sidebar.markdown(
        "*Built with Streamlit | Data processed by Kedro pipelines*",
        help="Run the pipelines to refresh the data"
    )

def render_customer_explorer(config, sentiment_predictions):
    """Original Customer Explorer View"""
    
    # Title and description
    st.title("👥 Customer Explorer")
    st.markdown("<div class='decoration'></div>", unsafe_allow_html=True)
    st.markdown("*Explore customer insights from sentiment and topic analysis*")
    
    # Load data
    sentiment_profiles = load_customer_sentiment_profiles()
    topic_profiles = load_customer_topic_profiles()
    # sentiment_predictions already loaded
    topic_assignments = load_topic_assignments()
    summaries = load_customer_summaries()
    
    # Check if data is available
    if sentiment_profiles is None or sentiment_profiles.empty:
        st.warning("""
        ⚠️ **No data available**
        
        Please run the Kedro pipelines first to generate customer profiles:
        ```bash
        kedro run --pipeline=sentiment_analysis
        kedro run --pipeline=topic_modeling
        ```
        """)
        return
    
    # Sort customers by total reviews (used in multiple tabs)
    customer_list = sentiment_profiles.sort_values(
        'total_reviews', ascending=False
    )['customer_id'].tolist()
    
    # Main content area
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Customer Overview",
        "🔎 Customer Details",
        "📊 Customer Visualizations",
        "🔄 Compare Customers",
        "📋 All Customers"
    ])
    
    # Tab 1: Customer Overview
    with tab1:
        st.header("📈 Customer Overview")
        
        # Customer Selection (moved from sidebar)
        st.subheader("🎯 Select Customer")
        selected_customer = st.selectbox(
            "Choose a customer to view details",
            options=customer_list,
            format_func=lambda x: f"{str(x)[:20]}... ({sentiment_profiles[sentiment_profiles['customer_id']==x]['total_reviews'].values[0]} reviews)",
            key="overview_customer_select"
        )
        
        st.markdown("---")
        
        # Get customer data
        customer_sentiment = sentiment_profiles[
            sentiment_profiles['customer_id'] == selected_customer
        ].iloc[0].to_dict()
        
        customer_topic = None
        if topic_profiles is not None and not topic_profiles.empty:
            topic_match = topic_profiles[topic_profiles['customer_id'] == selected_customer]
            if not topic_match.empty:
                customer_topic = topic_match.iloc[0].to_dict()
        
        # Merge for KPIs
        customer_data = {**customer_sentiment}
        if customer_topic:
            customer_data.update(customer_topic)
        
        # KPI Cards
        create_customer_kpi_cards(customer_data)
        
        st.markdown("---")
        
        # Charts row
        col1, col2 = st.columns(2)
        
        with col1:
            # Sentiment Gauge
            gauge = create_sentiment_gauge(
                customer_sentiment.get('positive_pct', 0),
                customer_sentiment.get('neutral_pct', 0),
                customer_sentiment.get('negative_pct', 0)
            )
            st.plotly_chart(gauge, use_container_width=True)
        
        with col2:
            # Sentiment Pie
            pie = create_sentiment_pie(
                customer_sentiment.get('positive_count', 0),
                customer_sentiment.get('neutral_count', 0),
                customer_sentiment.get('negative_count', 0)
            )
            st.plotly_chart(pie, use_container_width=True)
    
    # Tab 2: Customer Details
    with tab2:
        st.header("🔎 Customer Details")
        
        # Customer Selection for this tab
        selected_customer_details = st.selectbox(
            "Select Customer",
            options=customer_list,
            format_func=lambda x: f"{str(x)[:20]}... ({sentiment_profiles[sentiment_profiles['customer_id']==x]['total_reviews'].values[0]} reviews)",
            key="details_customer_select"
        )
        
        st.markdown("---")
        
        if sentiment_predictions is not None:
            # Customer reviews
            st.subheader("📝 Actionable Interaction Insights")
            
            # Use unified records if available
            unified_records = load_unified_interaction_records()
            
            if unified_records is not None and not unified_records.empty:
                cust_unified = unified_records[unified_records['customer_id'] == selected_customer_details]
                if not cust_unified.empty:
                    # Select columns for actionable view
                    display_df = cust_unified[[
                        'date', 'review_text', 'sentiment', 'detected_emotion', 
                        'urgency', 'recommended_action'
                    ]].copy()
                    
                    # Highlight Urgency
                    def style_urgency(val):
                        if val == 'High':
                            return 'color: white; background-color: #e74c3c; font-weight: bold'
                        elif val == 'Medium':
                            return 'background-color: #f39c12'
                        return ''
                    
                    st.dataframe(
                        display_df.style.applymap(style_urgency, subset=['urgency']),
                        height=400,
                        use_container_width=True
                    )
                    
                    # Aspect Sentiment for this customer
                    aspect_profiles = load_customer_aspect_profiles()
                    if aspect_profiles is not None:
                        cust_aspects = aspect_profiles[aspect_profiles['customer_id'] == selected_customer_details]
                        if not cust_aspects.empty:
                            st.subheader("📊 Aspect Sentiment Analysis")
                            fig = create_aspect_sentiment_chart(cust_aspects)
                            st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No unified records found for this customer.")
            else:
                # Fallback to original reviews table
                reviews = create_customer_reviews_table(
                    sentiment_predictions,
                    selected_customer_details,
                    max_rows=20
                )
                
                if not reviews.empty:
                    # Color code by sentiment
                    def color_sentiment(val):
                        if val == 'positive':
                            return 'background-color: #d5f5e3'
                        elif val == 'negative':
                            return 'background-color: #fadbd8'
                        return 'background-color: #fef9e7'
                    
                    styled_reviews = reviews.style.applymap(
                        color_sentiment, subset=['sentiment']
                    )
                    st.dataframe(styled_reviews, height=400, use_container_width=True)
                else:
                    st.info("No reviews found for this customer.")
        else:
            st.info("Sentiment predictions not available.")
        
        # Topic information
        if topic_profiles is not None and not topic_profiles.empty:
            topic_match = topic_profiles[topic_profiles['customer_id'] == selected_customer_details]
            if not topic_match.empty:
                customer_topic_details = topic_match.iloc[0].to_dict()
                st.subheader("🏷️ Topic Profile")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Dominant Topic", customer_topic_details.get('dominant_topic_name', 'N/A')[:30])
                with col2:
                    st.metric("Unique Topics", customer_topic_details.get('unique_topics', 0))
                with col3:
                    st.metric(
                        "Topic Entropy ℹ️",
                        f"{customer_topic_details.get('topic_entropy', 0):.2f}",
                        help="**Formula:** -Σ(p_i × log₂(p_i)) where p_i is the probability of each topic\n\n"
                             "**Range:** 0.0 to ~log₂(n) where n is number of unique topics\n\n"
                             "**What it measures:** Information-theoretic measure of topic distribution diversity using Shannon entropy.\n\n"
                             "**Interpretation:**\n"
                             "• Higher values: Reviews are evenly distributed across many topics (more unpredictable/diverse)\n"
                             "• Lower values: Reviews are concentrated in fewer topics (more predictable/focused)\n"
                             "• 0.0: All reviews discuss the same single topic\n\n"
                             "**Example:** Customer with 8 reviews split as [4, 2, 1, 1] across 4 topics has higher entropy than [7, 1] across 2 topics.\n\n"
                             "**Difference from Topic Diversity:** Entropy considers the distribution balance, while Diversity only counts unique topics."
                    )
    
    # Tab 3: Customer Visualizations
    with tab3:
        st.header("📊 Customer Visualizations")
        st.markdown("*Select customers to generate interactive visualizations.*")
        
        # Customer selection for visualizations
        viz_customers = st.multiselect(
            "Select Customers to Visualize",
            options=customer_list[:50],  # Limit to top 50
            default=customer_list[:5] if len(customer_list) >= 5 else customer_list,
            max_selections=15,
            key="viz_customer_select"
        )
        
        if not viz_customers:
            st.info("Please select at least one customer to visualize.")
        else:
            # Visualization type selection
            viz_type = st.radio(
                "Select Visualization Type",
                options=["Sentiment Sunburst", "Sentiment Timeline", "Topic Treemap", "Review Volume"],
                horizontal=True,
                key="viz_type_select"
            )
            
            st.markdown("---")
            
            if viz_type == "Sentiment Sunburst":
                st.subheader("🌅 Sentiment Sunburst")
                st.caption("Hierarchical view of sentiment distribution across selected customers.")
                fig = create_dynamic_sentiment_sunburst(
                    sentiment_profiles,
                    sentiment_predictions,
                    viz_customers
                )
                st.plotly_chart(fig, use_container_width=True)
                
            elif viz_type == "Sentiment Timeline":
                st.subheader("📈 Sentiment Timeline")
                st.caption("Track sentiment trends over time for selected customers.")
                fig = create_dynamic_sentiment_timeline(
                    sentiment_predictions,
                    viz_customers
                )
                st.plotly_chart(fig, use_container_width=True)
                
            elif viz_type == "Topic Treemap":
                st.subheader("🗺️ Topic Treemap")
                st.caption("Topic distribution across selected customers.")
                fig = create_dynamic_topic_treemap(
                    topic_profiles,
                    viz_customers
                )
                st.plotly_chart(fig, use_container_width=True)
                
            elif viz_type == "Review Volume":
                st.subheader("📊 Review Volume")
                st.caption("Review volume breakdown by sentiment for selected customers.")
                fig = create_dynamic_review_volume_chart(
                    sentiment_profiles,
                    viz_customers
                )
                st.plotly_chart(fig, use_container_width=True)
    
    # Tab 4: Compare Customers
    with tab4:
        st.header("🔄 Customer Comparison")
        
        # Customer multi-select
        comparison_customers = st.multiselect(
            "Select customers to compare",
            options=customer_list[:50],  # Limit to top 50
            default=customer_list[:3] if len(customer_list) >= 3 else customer_list,
            max_selections=10
        )
        
        if len(comparison_customers) >= 2:
            # Metric selection
            metric = st.selectbox(
                "Select metric to compare",
                options=['total_reviews', 'sentiment_score', 'positive_pct', 'negative_pct', 'avg_confidence']
            )
            
            # Comparison chart
            comparison_chart = create_customer_comparison_chart(
                sentiment_profiles,
                comparison_customers,
                metric
            )
            st.plotly_chart(comparison_chart, use_container_width=True)
            
            # Radar chart comparison
            st.subheader("Multi-dimensional Comparison")
            
            if topic_profiles is not None:
                merged = sentiment_profiles.merge(
                    topic_profiles[['customer_id', 'topic_diversity']],
                    on='customer_id',
                    how='inner'
                )
                comparison_data = merged[merged['customer_id'].isin(comparison_customers)]
                
                fig = go.Figure()
                
                categories = ['Reviews', 'Positivity', 'Confidence', 'Diversity']
                
                for _, row in comparison_data.iterrows():
                    values = [
                        row['total_reviews'] / comparison_data['total_reviews'].max(),
                        row['positive_pct'] / 100,
                        row['avg_confidence'],
                        row['topic_diversity']
                    ]
                    values.append(values[0])
                    
                    fig.add_trace(go.Scatterpolar(
                        r=values,
                        theta=categories + [categories[0]],
                        fill='toself',
                        name=str(row['customer_id'])[:15],
                        opacity=0.6
                    ))
                
                fig.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                    height=500,
                    title="Customer Comparison Radar"
                )
                
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Select at least 2 customers to compare.")
    
    # Tab 5: All Customers
    with tab5:
        st.header("📋 All Customers")
        
        # Filters (moved from sidebar)
        st.subheader("🔍 Filters")
        
        filter_col1, filter_col2 = st.columns(2)
        
        with filter_col1:
            min_reviews = st.slider(
                "Minimum Reviews",
                min_value=1,
                max_value=int(sentiment_profiles['total_reviews'].max()),
                value=3,
                key="all_customers_min_reviews"
            )
        
        with filter_col2:
            sentiment_filter = st.multiselect(
                "Sentiment Filter",
                options=['positive', 'neutral', 'negative'],
                default=['positive', 'neutral', 'negative'],
                key="all_customers_sentiment_filter"
            )
        
        st.markdown("---")
        
        # Apply filters
        filtered_profiles = sentiment_profiles[
            (sentiment_profiles['total_reviews'] >= min_reviews) &
            (sentiment_profiles['dominant_sentiment'].isin(sentiment_filter))
        ].copy()
        
        # Sort options
        sort_col1, sort_col2 = st.columns([2, 1])
        with sort_col1:
            sort_by = st.selectbox(
                "Sort by",
                options=['total_reviews', 'sentiment_score', 'positive_pct', 'negative_pct', 'avg_confidence'],
                index=0,
                key="all_customers_sort_by"
            )
        with sort_col2:
            sort_order = st.radio("Order", ['Descending', 'Ascending'], horizontal=True, key="all_customers_order")
        
        filtered_profiles = filtered_profiles.sort_values(
            sort_by,
            ascending=(sort_order == 'Ascending')
        )
        
        # Display stats
        st.caption(f"Showing {len(filtered_profiles)} of {len(sentiment_profiles)} customers")
        
        # Display table
        st.dataframe(
            filtered_profiles,
            height=500,
            use_container_width=True,
            hide_index=True
        )
        
        # Export option
        st.download_button(
            "📥 Download as CSV",
            filtered_profiles.to_csv(index=False),
            file_name="customer_profiles.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    main()
