import streamlit as st
import pandas as pd
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Any, Dict, List, Optional, Tuple

def main():
    st.set_page_config(
        page_title="AI Resume Optimizer",
        page_icon="📄",
        layout="wide"
    )
    
    st.title("🚀 AI-Powered Resume Optimizer")
    st.markdown("Upload your resume and get comprehensive ATS compatibility analysis with actionable insights.")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")
        
        job_field = st.selectbox(
            "Select Job Field",
            options=[field.value for field in JobField],
            index=0
        )
        
        analysis_type = st.selectbox(
            "Analysis Type",
            options=["Comprehensive", "Quick", "ATS Focus"],
            index=0
        )
        
        st.header("About")
        st.info("This tool analyzes your resume for ATS compatibility, keyword optimization, and professional formatting.")
    
    # Main content area
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.header("Upload Resume")
        uploaded_file = st.file_uploader(
            "Choose your resume file",
            type=['pdf', 'docx', 'txt'],
            help="Upload a PDF, DOCX, or TXT file (max 10MB)"
        )
        
        if uploaded_file:
            # Validate file
            is_valid, validation_message = FileHandler.validate_file(uploaded_file)
            
            if is_valid:
                st.success(validation_message)
                
                if st.button("Analyze Resume", type="primary"):
                    with st.spinner("Analyzing your resume..."):
                        # Extract text from file
                        resume_text = FileHandler.extract_text(uploaded_file)
                        
                        if resume_text:
                            # Initialize analyzer
                            analyzer = EnhancedResumeOptimizationCrew(job_field=job_field)
                            
                            # Analyze resume
                            results = analyzer.analyze_resume(resume_text)
                            
                            if results.get("success"):
                                st.session_state['analysis_results'] = results
                                st.session_state['resume_text'] = resume_text
                                st.rerun()
                            else:
                                st.error(f"Analysis failed: {results.get('error', 'Unknown error')}")
                        else:
                            st.error("Could not extract text from the uploaded file.")
            else:
                st.error(validation_message)
    
    with col2:
        if 'analysis_results' in st.session_state:
            display_results(st.session_state['analysis_results'])
        else:
            st.header("Analysis Results")
            st.info("Upload a resume and click 'Analyze Resume' to see results here.")

def display_results(results: Dict[str, Any]) -> None:
    """Display analysis results"""
    st.header("📊 Analysis Results")
    
    # ATS Compatibility Score
    scores = results.get('ats_scores', {})
    total_score = scores.get('total_score', 0)
    
    # Score gauge
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = total_score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "ATS Compatibility Score"},
        delta = {'reference': 80},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 50], 'color': "lightgray"},
                {'range': [50, 80], 'color': "yellow"},
                {'range': [80, 100], 'color': "green"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Score breakdown
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Keyword Relevance", f"{scores.get('keyword_relevance', 0):.1f}%")
        st.metric("Section Structure", f"{scores.get('section_structure', 0):.1f}%")
    
    with col2:
        st.metric("Format Compatibility", f"{scores.get('format_compatibility', 0):.1f}%")
        st.metric("Readability", f"{scores.get('readability', 0):.1f}%")
    
    with col3:
        st.metric("Completeness", f"{scores.get('completeness', 0):.1f}%")
    
    # Detailed analysis tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📝 Sections", "🔑 Keywords", "📊 Readability", "💡 Recommendations"])
    
    with tab1:
        st.subheader("Resume Sections Analysis")
        sections = results.get('sections', {})
        
        if sections:
            for section_name, content in sections.items():
                with st.expander(f"{section_name.title()} Section"):
                    st.text_area("Content", content, height=100, disabled=True)
        else:
            st.warning("No standard sections detected. Consider organizing your resume with clear section headers.")
    
    with tab2:
        st.subheader("Keyword Analysis")
        keyword_analysis = results.get('keyword_analysis', {})
        
        if keyword_analysis:
            # Create keyword dataframe
            keyword_data = []
            for keyword, data in keyword_analysis.items():
                keyword_data.append({
                    'Keyword': keyword,
                    'Count': data['count'],
                    'Density': f"{data['density']*100:.2f}%"
                })
            
            df = pd.DataFrame(keyword_data)
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No keyword analysis available.")
    
    with tab3:
        st.subheader("Readability Metrics")
        readability = results.get('readability', {})
        
        if readability:
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Word Count", readability.get('word_count', 0))
                st.metric("Sentence Count", readability.get('sentence_count', 0))
                st.metric("Avg Sentence Length", f"{readability.get('avg_sentence_length', 0):.1f}")
            
            with col2:
                st.metric("Flesch Reading Ease", f"{readability.get('flesch_reading_ease', 0):.1f}")
                st.metric("Flesch-Kincaid Grade", f"{readability.get('flesch_kincaid_grade', 0):.1f}")
                st.metric("Automated Readability Index", f"{readability.get('automated_readability_index', 0):.1f}")
    
    with tab4:
        st.subheader("Optimization Recommendations")
        recommendations = results.get('recommendations', [])
        
        if recommendations:
            for i, rec in enumerate(recommendations, 1):
                st.write(f"{i}. {rec}")
        else:
            st.success("Great job! No major improvements needed.")
        
        # Contact info analysis
        contact_info = results.get('contact_info', {})
        if contact_info:
            st.subheader("Contact Information Found")
            for key, value in contact_info.items():
                st.write(f"**{key.title()}:** {value}")

if __name__ == "__main__":
    main()
