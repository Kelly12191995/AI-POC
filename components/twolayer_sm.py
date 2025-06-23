import streamlit as st
import pandas as pd
import io
import logging
from typing import Dict, Any, Optional
import os
from datetime import datetime
from openpyxl import load_workbook
from openpyxl.styles import PatternFill, Font, Alignment, Border
from openpyxl.utils.dataframe import dataframe_to_rows
import copy


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AIClient:
    """Initial the Connection to Gemini"""
    def __init__(self):
        self.available = False
        try:
            from google import genai
            self.client = genai.Client(
                vertexai=True, 
                project='prj-exp-sbx-apr-xqs7fz4g', 
                location='us-central1'
            )
            self.available = True
            pass
        except Exception as e:
            logger.error(f"Failed to initialize AI client: {e}")
            st.warning("AI client not configured")
    
    def generate_content(self, prompt: str) -> str:
        """Generate AI content - replace with actual implementation"""
        if self.available:
            response = self.client.models.generate_content(
                model='gemini-2.0-flash-001',
                contents=[prompt]
            )
            return response.text


class ReportConfig:
    """Configuration class for different report types with two-layer AI instructions"""
    
    @staticmethod
    def get_all_reports() -> Dict[str, Dict[str, Any]]:
        """Returns all available report configurations"""
        return {
            "Digital Sales Report": ReportConfig.get_sales_config(),  
            "Digital Billboard Report": ReportConfig.get_billboard_config(),
            # "PN": ReportConfig.get_pn_config(),
            # "transaction": ReportConfig.get_trx_config(),
        }
    
    @staticmethod
    def get_sales_config() -> Dict[str, Any]:
        """Configuration for Digital Sales Report with two-layer analysis structure"""
        return {
            "file": "Digital Sales report.xlsx",  ###-- changed to align with Monday's distribution name
            "description": "Weekly sales and traffic report analyzing gross sales trends, customer segmentation, and channel performance.",
            "focus_areas": [
                "Gross Sales Trends – Compare week-over-week sales fluctuations. Identify key drivers: new vs existing customers, Mobility vs Residential sales.",
                "Overall Shop Traffic – The number of total visits on Shop pages.",
                "Overall Channel Mix – Evaluate the proportion of sales driven by online channels compared to other sales avenues.",
                "Close Rate – Measure the close rate by calculating sales divided by total shop traffic."
            ],
            "ai_instructions": {
                "executive_summary": {
                    "instruction": """
                    As a professional Data Analyst, provide a data-driven executive overview focusing on the most recent week's performance.
                    
                    **STRUCTURE REQUIREMENTS:**
                    1. **Start with Key Performance Numbers**: Begin by presenting actual volume for the most recent week and the week-over-week (WoW) percentage change
                    2. **Use Bullet Point Format**: Provide the rest of your response in structured bullet points
                    
                    **Focus Areas (in bullet points):**
                    • **Gross Sales Performance**: the most recent week vs previous week comparison with key drivers (Mobility vs. Residential, New vs Existing Customers, ON vs QC)
                    • **Traffic & Close Rate Trends**: Shop traffic changes and close rate impacts for the most recent week
                    • **Channel Mix Evolution**: Online vs. other sales channels performance and week-over-week shifts
                
                    
                    Keep the tone executive-level, data-driven, and focused on the most recent week's strategic implications.
                    """,
                    "default_sentences": 6
                },

                "detailed_analysis": {
                    "instruction": """
                    As a Senior Data Analyst, provide detailed data-driven insights focusing on:
                    
                    1. Granular Performance Metrics: Analyze Trend according to the past 8 weeks
                    2. Segmentation Analysis: Deep dive into customer type new vs exisiting, sales type Add a line vs Cross Sales vs Add Rgu, regional performance ON vs QC, LOB(line of Bussines) performace Mobility vs Residentail.
                    3. Traffic Mix: Identify the traffic mix trend, Direct vs Search vs Digital/Social vs Other traffic Mix
                    
                    Always reference specific metrics and provide actual data volume.
                    """,
                    "default_sentences": 2
                }
            },
            "sample_questions": {
                "What is the Activation% for the most recent week?": {
                    "analysis_type": "detailed_analysis",
                    "context": "Analyze the Activation% row and compare it to previous periods. Focus on activation trends."
                },
                "Which line of business experienced fluctuations for most recent week, Mobility Sales or Residential Sales?": {
                    "analysis_type": "detailed_analysis", 
                    "context": "Compare Mobility vs Residential sales of most recent week vs other weeks. Identify the biggest change driver and its impact."
                },
                "How did close rates (vs Shop Traffic) change week-over-week?": {
                    "analysis_type": "detailed_analysis",
                    "context": "Examine close rates (vs Shop Traffic) for the most recent week vs previous week. Highlight major fluctuations and their business implications."
                }
            }
        }

    @staticmethod
    def get_billboard_config() -> Dict[str, Any]:
        """Configuration for Digital Billboard Report with two-layer analysis structure"""
        return {
            "file": "Digital_Billboard_Report.xlsx",
            "description": "Digital Billboard is personalized offer for customer at the Logged in space, available on Web and App",
            "focus_areas": [
                "Impressions Volume – The number of times a digital billboard rendered to a customer",
                "Clicks Volume – The number of clicks on a digital billboard",
                "Sales Volume – Post-click-sale happened in all sales channel, same-day, 7day, 30day sales. Key item to track the impact of digital billboard",
                "Transaction Volume – Post-click-transaction happened in online channel"
            ],
            "ai_instructions": {
                "executive_summary": {
                    "instruction": """
                    As a Digital Marketing Director, provide a strategic overview focusing on the most recent week's digital billboard performance (WE May 17).
                    
                    **STRUCTURE REQUIREMENTS:**
                    1. **Start with Key Performance Numbers**: Begin by presenting actual impression volume, click volume, and sales volume for the most recent week with week-over-week (WoW) percentage changes
                    2. **Use Bullet Point Format**: Provide the rest of your response in structured bullet points
                    
                    **Focus Areas (in bullet points):**
                    • **Impression Performance**: WE May 17 vs WE May 10 impression volume and growth drivers
                    • **Click Performance**: Click volume changes and click-through rate trends
                    • **Sales Impact**: Post-click sales performance and conversion effectiveness 
                    • **Channel Comparison**: Web vs App performance differences and strategic implications
                    • **Strategic Recommendations**: Key optimization opportunities for leadership
                    
                    Keep the tone strategic and accessible to marketing leadership stakeholders.
                    **Format: Start with actual numbers, then bullet points for structured insights.**
                    """,
                    "default_sentences": 7
                },
                "detailed_analysis": {
                    "instruction": """
                    As a Digital Advertising Specialist, provide detailed analysis focusing on:
                    
                    1. Performance Metrics: Analyze impression volume, click-through rates, conversion metrics
                    2. Channel Comparison: Deep dive into Web vs App performance, user behavior differences
                    3. Funnel Analysis: Examine impression-to-click-to-sale conversion patterns
                    4. Temporal Patterns: Identify peak performance times, day-of-week trends, seasonal effects
                    5. Optimization Insights: Provide specific recommendations for campaign improvements
                    
                    Always reference specific metrics and provide quantitative insights.
                    Focus on actionable optimizations and performance improvements.
                    """,
                    "default_sentences": 2
                }
            },
            "sample_questions": {
                "Which channels contributed most to impression growth?": {
                    "analysis_type": "detailed_analysis",
                    "context": "Analyze Web and App channel impressions volume WE May 17 vs WE May 10. Focus on channel performance and growth drivers."
                },
                "What was the click-through rate performance across channels?": {
                    "analysis_type": "detailed_analysis",
                    "context": "Calculate and compare click-through rates between Web and App channels. Analyze performance differences and implications."
                },
                "How did post-click sales conversion perform?": {
                    "analysis_type": "detailed_analysis", 
                    "context": "Examine the conversion from clicks to sales across different time periods (same-day, 7-day, 30-day). Identify conversion patterns."
                }
            }
        }

class ReportAnalyzer:
    """Main class for handling report analysis with configurable two-layer AI prompting"""
    
    def __init__(self):
        self.ai_client = AIClient()
        self.reports_config = ReportConfig.get_all_reports()
    
    @st.cache_data
    def load_report_data(_self, file_path: str, report_type: str) -> Optional[pd.DataFrame]:
        """Load report data for analysis"""
        try:
            if os.path.exists(file_path):
                if file_path.endswith('.xlsx') or file_path.endswith('.xls'):
                    df = pd.read_excel(f"data/{file_path}", header=0)
                elif file_path.endswith('.csv'):
                    df = pd.read_csv(f"data/{file_path}")
                else:
                    raise ValueError("Unsupported file format")
                
                logger.info(f"Successfully loaded {file_path} with {len(df)} rows")
                return df
            else:
                st.error(f"File {file_path} not found. Please add the file to continue.")
                return None
                
        except Exception as e:
            logger.error(f"Error loading file {file_path}: {e}")
            st.error(f"Error loading file: {e}")
            return None
    
    def get_original_excel_bytes(self, file_path: str) -> bytes:
        """Get original Excel file as bytes for download"""
        try:
            with open(file_path, "rb") as file:
                return file.read()
        except Exception as e:
            logger.error(f"Error reading original Excel file: {e}")
            return b""
    
    def create_excel_download(self, report_config: Dict) -> bytes:
        """Get original Excel file as bytes for download - preserves all formatting"""
        file_path = report_config['file']
        try:
            with open(file_path, "rb") as file:
                return file.read()
        except Exception as e:
            logger.error(f"Error reading original Excel file {file_path}: {e}")
            st.error(f"Error reading Excel file for download: {e}")
            return b""
    
    def generate_ai_content(self, report_config: Dict, analysis_type: str = "detailed_analysis", num_sentences: int = None, data_context: str = "", question_context: str = "") -> str:
        """Generate AI content using only predefined instructions"""
        
        ai_instructions = report_config.get("ai_instructions", {})
        analysis_config = ai_instructions.get(analysis_type, {})
        
        if isinstance(analysis_config, dict):
            specific_instruction = analysis_config.get("instruction", "")
            default_sentences = analysis_config.get("default_sentences", 2)
        else:
            specific_instruction = analysis_config
            default_sentences = 2
        
        sentence_count = num_sentences if num_sentences is not None else default_sentences
        
        sentence_instruction = f"\n\nIMPORTANT: Provide your response in exactly {sentence_count} sentences."
        
        # Build the prompt with instruction, optional question context, and data
        prompt_parts = [specific_instruction]
        if question_context:
            prompt_parts.append(f"\nSpecific Context: {question_context}")
        
        prompt_parts.append(sentence_instruction)
        prompt_parts.append(f"\nData:\n{data_context}")
        
        enhanced_prompt = "".join(prompt_parts)
        
        try:
            return self.ai_client.generate_content(enhanced_prompt)
        except Exception as e:
            logger.error(f"AI generation failed: {e}")
            return f"Unable to generate AI insights: {e}"

def main():
    """Main application"""
    st.set_page_config(
        page_title="AI-powered Report Insight Hub",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    analyzer = ReportAnalyzer()
    
    st.title("Customer Ops AI Insights Dashboard")
    st.markdown("*AI-powered insights with configurable response length*")
    
    # Sidebar to choose Report
    with st.sidebar:
        st.header("📋 Report Selection")
        selected_report = st.selectbox(
            "Choose Report Type:",
            list(analyzer.reports_config.keys()),
            help="Each report type has specialized executive and detailed analysis"
        )
        
        report_config = analyzer.reports_config[selected_report]
        
        # Sentence count 2-10 sentences -should i change to token count?
        st.header("🎯 Interactive Analysis Settings")
        sentence_count = st.slider(
            "Response Length (sentences):",
            min_value=1,
            max_value=10,
            value=2,
            help="Control the length of detailed analysis responses"
        )
        
        # Show report info - different info for different content
        st.subheader("📈 Report Info")
        st.write(f"**Description:** {report_config['description']}")
        
        st.subheader("🎯 Focus Areas")
        for area in report_config['focus_areas']:
            st.write(f"• {area}")
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    # Load data
    df = analyzer.load_report_data(report_config['file'], selected_report)
    
    if df is not None:
        with col1:
            st.subheader(f"📊 {selected_report} - Data Overview")
            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            with col_m1:
                st.metric("Records", len(df))
            with col_m2:
                st.metric("Columns", len(df.columns))
            with col_m3:
                numeric_cols = df.select_dtypes(include=['number']).columns
                st.metric("Numeric Fields", len(numeric_cols))
            with col_m4:
                if not df.empty:
                    date_cols = df.select_dtypes(include=['datetime64']).columns
                    st.metric("Date Fields", len(date_cols))
            
            # Download the selected report
            excel_data = analyzer.create_excel_download(report_config)
            st.download_button(
                label="📥 Download Original Excel Report",
                data=excel_data,
                file_name=f"{selected_report.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                help="Download original Excel file with all formatting preserved"
            )
            
            # show either detailed report or 20 recordes
            if st.checkbox("Show detailed data"):
                st.dataframe(df, use_container_width=True)
            else:
                st.dataframe(df.head(20), use_container_width=True)
                st.caption(f"Showing first 20 of {len(df)} rows")
    
        with col2:
            st.subheader("📌 Executive Summary")
            st.caption("Focus: Most recent week's performance with bullet points")
            if st.button("🔄 Generate Executive Summary", use_container_width=True):
                with st.spinner("Generating executive summary..."):
                    data_context = df.to_string()
                    
                    exec_sentences = report_config["ai_instructions"]["executive_summary"]["default_sentences"]
                    summary = analyzer.generate_ai_content(
                        report_config, 
                        "executive_summary",
                        exec_sentences,
                        data_context
                    )
                    st.session_state.executive_summary = summary
            
            if hasattr(st.session_state, 'executive_summary'):
                st.write(st.session_state.executive_summary)
    else:
        st.warning(f"Please ensure the data file '{report_config['file']}' is available in the current directory.")
    
    st.markdown("---")
    st.subheader("💬 Interactive Analysis (Detailed Layer)")
    col_settings1, col_settings2 = st.columns(2)
    with col_settings1:
        st.info("**Analysis Type:** Detailed Analysis")
    with col_settings2:
        st.info(f"**Response Length:** {sentence_count} sentences")
    
    # Sample questions
    st.write("**💡 Suggested Questions (Based on Detailed Analysis Layer):**")
    sample_questions = report_config.get('sample_questions', {})
    
    if sample_questions:
        cols = st.columns(2)
        for i, (question, config) in enumerate(sample_questions.items()):
            col_idx = i % 2
            with cols[col_idx]:
                if st.button(f"❓ {question}", key=f"q_{i}"):
                    st.session_state.user_question = question
                    st.session_state.question_config = config
    
    # Chat input
    user_input = st.chat_input(f"Ask about {selected_report} (detailed analysis layer)...")
    
    # Handle button click
    if hasattr(st.session_state, 'user_question'):
        user_input = st.session_state.user_question
        question_config = getattr(st.session_state, 'question_config', {})
        delattr(st.session_state, 'user_question')
        if hasattr(st.session_state, 'question_config'):
            delattr(st.session_state, 'question_config')
    else:
        question_config = {}
    
    if user_input and df is not None:
        with st.chat_message("user"):
            st.write(user_input)
        
        with st.chat_message("assistant"):
            with st.spinner("Analyzing..."):
                data_context = df.to_string()
                
                # For sample questions, the prompt includeing Detailed_analysis layer + the given context(should i just keep contect?)
                if question_config:
                    question_analysis_type = question_config.get('analysis_type', 'detailed_analysis')
                    question_context = question_config.get('context', '')
                    
                    response = analyzer.generate_ai_content(
                        report_config,
                        question_analysis_type,
                        sentence_count,
                        data_context,
                        question_context
                    )
                else:
                    # For user random input, the prompt is following the detailed analysis layer instuctions
                    response = analyzer.generate_ai_content(
                        report_config,
                        "detailed_analysis",
                        sentence_count,
                        data_context
                    )
                
                st.write(response)
    elif user_input and df is None:
        with st.chat_message("assistant"):
            st.write("Please load a data file first before asking questions about the report.")

if __name__ == "__main__":
    main()