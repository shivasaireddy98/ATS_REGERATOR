# workflow.py - Enhanced LangGraph Workflow
import json
import logging
from typing import Dict, List, Optional, Any, TypedDict
from datetime import datetime
from dataclasses import dataclass, asdict

from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

from config import ATSConfig, INDUSTRY_KEYWORDS, ATS_SCORING_CRITERIA
from utils import TextProcessor, ResumeValidator

# Configure logging
logger = logging.getLogger(__name__)

# State management for LangGraph
class ResumeWorkflowState(TypedDict):
    """State schema for the resume optimization workflow"""
    # Input data
    original_content: str
    job_field: str
    user_preferences: Dict[str, Any]
    
    # Processing stages
    current_step: str
    progress: float
    errors: List[str]
    warnings: List[str]
    
    # Parsed content
    parsed_sections: Dict[str, str]
    contact_info: Dict[str, str]
    content_metadata: Dict[str, Any]
    
    # Analysis results
    ats_analysis: Dict[str, Any]
    keyword_analysis: Dict[str, Any]
    format_analysis: Dict[str, Any]
    overall_score: float
    
    # Optimization results
    optimized_content: str
    improvements_made: List[str]
    recommendations: List[str]
    
    # Final output
    final_resume: str
    optimization_summary: Dict[str, Any]

@dataclass
class WorkflowConfig:
    """Configuration for workflow execution"""
    max_iterations: int = 5
    timeout_seconds: int = 300
    enable_caching: bool = True
    detailed_logging: bool = True

class ResumeOptimizationWorkflow:
    """Enhanced LangGraph workflow for resume optimization"""
    
    def __init__(self, llm: ChatOpenAI, job_field: str = "Technology", config: WorkflowConfig = None):
        self.llm = llm
        self.job_field = job_field
        self.config = config or WorkflowConfig()
        self.ats_config = ATSConfig()
        
        # Build the workflow graph
        self.graph = self._build_enhanced_graph()
    
    def _build_enhanced_graph(self) -> StateGraph:
        """Build the enhanced LangGraph workflow"""
        
        workflow = StateGraph(ResumeWorkflowState)
        
        # Add workflow nodes
        workflow.add_node("initialize", self._initialize_workflow)
        workflow.add_node("validate_input", self._validate_input)
        workflow.add_node("parse_content", self._parse_content)
        workflow.add_node("analyze_ats", self._analyze_ats_compatibility)
        workflow.add_node("analyze_keywords", self._analyze_keywords)
        workflow.add_node("analyze_format", self._analyze_format)
        workflow.add_node("generate_recommendations", self._generate_recommendations)
        workflow.add_node("optimize_content", self._optimize_content)
        workflow.add_node("validate_optimization", self._validate_optimization)
        workflow.add_node("finalize_resume", self._finalize_resume)
        workflow.add_node("handle_error", self._handle_error)
        
        # Define workflow edges
        workflow.add_edge("initialize", "validate_input")
        workflow.add_conditional_edges(
            "validate_input",
            self._should_continue_after_validation,
            {
                "continue": "parse_content",
                "error": "handle_error"
            }
        )
        workflow.add_edge("parse_content", "analyze_ats")
        workflow.add_edge("analyze_ats", "analyze_keywords")
        workflow.add_edge("analyze_keywords", "analyze_format")
        workflow.add_edge("analyze_format", "generate_recommendations")
        workflow.add_edge("generate_recommendations", "optimize_content")
        workflow.add_conditional_edges(
            "optimize_content",
            self._should_validate_optimization,
            {
                "validate": "validate_optimization",
                "finalize": "finalize_resume"
            }
        )
        workflow.add_conditional_edges(
            "validate_optimization",
            self._should_reoptimize,
            {
                "reoptimize": "optimize_content",
                "finalize": "finalize_resume"
            }
        )
        workflow.add_edge("finalize_resume", END)
        workflow.add_edge("handle_error", END)
        
        # Set entry point
        workflow.set_entry_point("initialize")
        
        return workflow.compile()
    
    def _initialize_workflow(self, state: ResumeWorkflowState) -> ResumeWorkflowState:
        """Initialize the workflow with default values"""
        logger.info("Initializing resume optimization workflow")
        
        state.update({
            "current_step": "initializing",
            "progress": 0.0,
            "errors": [],
            "warnings": [],
            "parsed_sections": {},
            "contact_info": {},
            "content_metadata": {},
            "ats_analysis": {},
            "keyword_analysis": {},
            "format_analysis": {},
            "overall_score": 0.0,
            "optimized_content": "",
            "improvements_made": [],
            "recommendations": [],
            "final_resume": "",
            "optimization_summary": {}
        })
        
        return state
    
    def _validate_input(self, state: ResumeWorkflowState) -> ResumeWorkflowState:
        """Validate input content and requirements"""
        logger.info("Validating input content")
        
        state["current_step"] = "validating_input"
        state["progress"] = 0.1
        
        try:
            content = state.get("original_content", "")
            
            if not content or len(content.strip()) < 50:
                state["errors"].append("Content too short or empty")
                return state
            
            # Basic validation using ResumeValidator
            validation_result = ResumeValidator.validate_resume_structure(content)
            
            if not validation_result["is_valid"]:
                state["errors"].extend(validation_result["issues"])
            
            if validation_result["warnings"]:
                state["warnings"].extend(validation_result["warnings"])
            
            # Store validation metadata
            state["content_metadata"]["validation"] = validation_result
            state["content_metadata"]["word_count"] = len(content.split())
            state["content_metadata"]["char_count"] = len(content)
            
        except Exception as e:
            logger.error(f"Input validation error: {e}")
            state["errors"].append(f"Validation error: {str(e)}")
        
        return state
    
    def _parse_content(self, state: ResumeWorkflowState) -> ResumeWorkflowState:
        """Parse resume content into structured sections"""
        logger.info("Parsing resume content")
        
        state["current_step"] = "parsing_content"
        state["progress"] = 0.2
        
        try:
            content = state["original_content"]
            
            # Extract contact information
            contact_info = {
                "email": TextProcessor.extract_email(content),
                "phone": TextProcessor.extract_phone(content),
                "urls": TextProcessor.extract_urls(content)
            }
            state["contact_info"] = contact_info
            
            # Parse sections using LLM
            parsing_prompt = f"""
            Parse the following resume content and extract the main sections.
            Identify and extract:
            1. Professional Summary/Objective
            2. Work Experience
            3. Education
            4. Skills
            5. Any additional relevant sections
            
            Resume Content:
            {content}
            
            Return a JSON object with section names as keys and content as values.
            """
            
            messages = [
                SystemMessage(content="You are a resume parsing expert. Extract and structure resume sections accurately."),
                HumanMessage(content=parsing_prompt)
            ]
            
            response = self.llm.invoke(messages)
            
            try:
                # Try to parse JSON response
                parsed_sections = json.loads(response.content)
                state["parsed_sections"] = parsed_sections
            except json.JSONDecodeError:
                # Fallback to simple text parsing
                state["parsed_sections"] = {"full_content": content}
                state["warnings"].append("Could not parse sections with LLM, using full content")
            
        except Exception as e:
            logger.error(f"Content parsing error: {e}")
            state["errors"].append(f"Parsing error: {str(e)}")
        
        return state
    
    def _analyze_ats_compatibility(self, state: ResumeWorkflowState) -> ResumeWorkflowState:
        """Analyze ATS compatibility"""
        logger.info("Analyzing ATS compatibility")
        
        state["current_step"] = "analyzing_ats"
        state["progress"] = 0.4
        
        try:
            content = state["original_content"]
            
            ats_prompt = f"""
            Analyze the following resume for ATS (Applicant Tracking System) compatibility.
            
            Check for:
            1. Formatting issues that could prevent ATS parsing
            2. Section header recognition
            3. Bullet point consistency
            4. Special characters that might cause problems
            5. Overall structure and organization
            
            Provide a compatibility score (0-100) and list specific issues and recommendations.
            
            Resume Content:
            {content}
            
            Return analysis in JSON format with keys: score, issues, recommendations, format_problems.
            """
            
            messages = [
                SystemMessage(content="You are an ATS compatibility expert. Analyze resumes for parsing compatibility."),
                HumanMessage(content=ats_prompt)
            ]
            
            response = self.llm.invoke(messages)
            
            try:
                ats_analysis = json.loads(response.content)
                state["ats_analysis"] = ats_analysis
            except json.JSONDecodeError:
                # Fallback analysis
                state["ats_analysis"] = {
                    "score": 70,
                    "issues": ["Unable to perform detailed analysis"],
                    "recommendations": ["Review formatting manually"],
                    "format_problems": []
                }
                state["warnings"].append("Could not parse ATS analysis response")
            
        except Exception as e:
            logger.error(f"ATS analysis error: {e}")
            state["errors"].append(f"ATS analysis error: {str(e)}")
        
        return state
    
    def _analyze_keywords(self, state: ResumeWorkflowState) -> ResumeWorkflowState:
        """Analyze keyword usage and optimization"""
        logger.info("Analyzing keywords")
        
        state["current_step"] = "analyzing_keywords"
        state["progress"] = 0.6
        
        try:
            content = state["original_content"]
            job_field = state.get("job_field", "Technology")
            
            # Get industry keywords
            from config import JobField
            job_field_enum = getattr(JobField, job_field.upper(), JobField.TECHNOLOGY)
            industry_keywords = INDUSTRY_KEYWORDS.get(job_field_enum, {})
            
            keyword_prompt = f"""
            Analyze the keyword usage in this resume for the {job_field} industry.
            
            Focus on:
            1. Current keyword density and usage
            2. Missing industry-relevant keywords
            3. Keyword integration opportunities
            4. Skills alignment with industry standards
            
            Industry Context: {job_field}
            Available Industry Keywords: {json.dumps(industry_keywords, indent=2)}
            
            Resume Content:
            {content}
            
            Return analysis in JSON format with keys: current_keywords, missing_keywords, 
            keyword_density, optimization_suggestions, skills_gaps.
            """
            
            messages = [
                SystemMessage(content=f"You are a keyword optimization expert specializing in {job_field} resumes."),
                HumanMessage(content=keyword_prompt)
            ]
            
            response = self.llm.invoke(messages)
            
            try:
                keyword_analysis = json.loads(response.content)
                state["keyword_analysis"] = keyword_analysis
            except json.JSONDecodeError:
                # Fallback analysis
                state["keyword_analysis"] = {
                    "current_keywords": [],
                    "missing_keywords": list(industry_keywords.get("programming", [])[:5]),
                    "keyword_density": 0.02,
                    "optimization_suggestions": ["Add more industry-specific keywords"],
                    "skills_gaps": []
                }
                state["warnings"].append("Could not parse keyword analysis response")
            
        except Exception as e:
            logger.error(f"Keyword analysis error: {e}")
            state["errors"].append(f"Keyword analysis error: {str(e)}")
        
        return state
    
    def _analyze_format(self, state: ResumeWorkflowState) -> ResumeWorkflowState:
        """Analyze formatting and layout"""
        logger.info("Analyzing format and layout")
        
        state["current_step"] = "analyzing_format"
        state["progress"] = 0.7
        
        try:
            content = state["original_content"]
            
            format_prompt = f"""
            Analyze the formatting and layout of this resume.
            
            Check for:
            1. Consistent bullet points and spacing
            2. Clear section headers
            3. Professional formatting
            4. Length optimization
            5. Visual hierarchy and readability
            
            Resume Content:
            {content}
            
            Return analysis in JSON format with keys: format_score, layout_issues, 
            spacing_problems, header_consistency, improvement_suggestions.
            """
            
            messages = [
                SystemMessage(content="You are a resume formatting expert focused on professional presentation."),
                HumanMessage(content=format_prompt)
            ]
            
            response = self.llm.invoke(messages)
            
            try:
                format_analysis = json.loads(response.content)
                state["format_analysis"] = format_analysis
            except json.JSONDecodeError:
                # Fallback analysis
                state["format_analysis"] = {
                    "format_score": 75,
                    "layout_issues": ["Inconsistent spacing"],
                    "spacing_problems": [],
                    "header_consistency": "Good",
                    "improvement_suggestions": ["Standardize bullet points"]
                }
                state["warnings"].append("Could not parse format analysis response")
            
        except Exception as e:
            logger.error(f"Format analysis error: {e}")
            state["errors"].append(f"Format analysis error: {str(e)}")
        
        return state
    
    def _generate_recommendations(self, state: ResumeWorkflowState) -> ResumeWorkflowState:
        """Generate comprehensive optimization recommendations"""
        logger.info("Generating optimization recommendations")
        
        state["current_step"] = "generating_recommendations"
        state["progress"] = 0.8
        
        try:
            # Compile all analysis results
            ats_analysis = state.get("ats_analysis", {})
            keyword_analysis = state.get("keyword_analysis", {})
            format_analysis = state.get("format_analysis", {})
            
            recommendations_prompt = f"""
            Based on the comprehensive analysis of this resume, generate specific, actionable 
            optimization recommendations.
            
            Analysis Results:
            ATS Analysis: {json.dumps(ats_analysis, indent=2)}
            Keyword Analysis: {json.dumps(keyword_analysis, indent=2)}
            Format Analysis: {json.dumps(format_analysis, indent=2)}
            
            Generate:
            1. Priority recommendations (top 5 most impactful changes)
            2. Quick wins (easy improvements with immediate impact)
            3. Long-term improvements (strategic enhancements)
            4. Industry-specific suggestions for {state.get('job_field', 'Technology')}
            
            Return recommendations in JSON format with keys: priority_recommendations, 
            quick_wins, long_term_improvements, industry_specific, overall_strategy.
            """
            
            messages = [
                SystemMessage(content="You are a resume optimization strategist providing actionable recommendations."),
                HumanMessage(content=recommendations_prompt)
            ]
            
            response = self.llm.invoke(messages)
            
            try:
                recommendations = json.loads(response.content)
                state["recommendations"] = recommendations.get("priority_recommendations", [])
                state["optimization_summary"]["recommendations"] = recommendations
            except json.JSONDecodeError:
                # Fallback recommendations
                fallback_recommendations = [
                    "Add a professional summary section",
                    "Include more industry-specific keywords",
                    "Standardize bullet point formatting",
                    "Quantify achievements with metrics",
                    "Optimize for single-page layout"
                ]
                state["recommendations"] = fallback_recommendations
                state["warnings"].append("Could not parse recommendations response")
            
        except Exception as e:
            logger.error(f"Recommendations generation error: {e}")
            state["errors"].append(f"Recommendations error: {str(e)}")
        
        return state
    
    def _optimize_content(self, state: ResumeWorkflowState) -> ResumeWorkflowState:
        """Optimize resume content based on recommendations"""
        logger.info("Optimizing resume content")
        
        state["current_step"] = "optimizing_content"
        state["progress"] = 0.9
        
        try:
            original_content = state["original_content"]
            recommendations = state.get("recommendations", [])
            job_field = state.get("job_field", "Technology")
            
            optimization_prompt = f"""
            Optimize the following resume based on the specific recommendations provided.
            Apply the improvements while maintaining the candidate's authentic voice and experience.
            
            Original Resume:
            {original_content}
            
            Specific Recommendations to Apply:
            {json.dumps(recommendations, indent=2)}
            
            Job Field: {job_field}
            
            Requirements:
            1. Maintain all factual information
            2. Improve ATS compatibility
            3. Enhance keyword usage naturally
            4. Optimize formatting and structure
            5. Keep within appropriate length
            
            Return the optimized resume content as plain text, followed by a brief summary 
            of improvements made (separate with "---IMPROVEMENTS---").
            """
            
            messages = [
                SystemMessage(content=f"You are an expert resume writer specializing in {job_field} resumes and ATS optimization."),
                HumanMessage(content=optimization_prompt)
            ]
            
            response = self.llm.invoke(messages)
            
            # Parse response to separate optimized content and improvements
            response_parts = response.content.split("---IMPROVEMENTS---")
            
            if len(response_parts) >= 2:
                optimized_content = response_parts[0].strip()
                improvements_text = response_parts[1].strip()
                improvements_made = [imp.strip() for imp in improvements_text.split('\n') if imp.strip()]
            else:
                optimized_content = response.content.strip()
                improvements_made = ["Content optimized based on analysis"]
            
            state["optimized_content"] = optimized_content
            state["improvements_made"] = improvements_made
            
        except Exception as e:
            logger.error(f"Content optimization error: {e}")
            state["errors"].append(f"Optimization error: {str(e)}")
            state["optimized_content"] = state["original_content"]  # Fallback to original
        
        return state
    
    def _validate_optimization(self, state: ResumeWorkflowState) -> ResumeWorkflowState:
        """Validate the optimized resume"""
        logger.info("Validating optimization results")
        
        state["current_step"] = "validating_optimization"
        
        try:
            optimized_content = state.get("optimized_content", "")
            original_content = state.get("original_content", "")
            
            # Basic validation checks
            validation_checks = {
                "length_appropriate": len(optimized_content.split()) <= len(original_content.split()) * 1.2,
                "contact_preserved": bool(TextProcessor.extract_email(optimized_content)),
                "structure_maintained": len(optimized_content) > len(original_content) * 0.8,
                "no_hallucination": True  # Placeholder for more sophisticated check
            }
            
            # Calculate improvement score
            original_word_count = len(original_content.split())
            optimized_word_count = len(optimized_content.split())
            
            state["optimization_summary"]["validation"] = validation_checks
            state["optimization_summary"]["word_count_change"] = optimized_word_count - original_word_count
            state["optimization_summary"]["improvements_applied"] = len(state.get("improvements_made", []))
            
            # Determine if reoptimization is needed
            failed_checks = [k for k, v in validation_checks.items() if not v]
            if failed_checks:
                state["warnings"].extend([f"Validation failed: {check}" for check in failed_checks])
            
        except Exception as e:
            logger.error(f"Optimization validation error: {e}")
            state["warnings"].append(f"Validation error: {str(e)}")
        
        return state
    
    def _finalize_resume(self, state: ResumeWorkflowState) -> ResumeWorkflowState:
        """Finalize the optimized resume"""
        logger.info("Finalizing resume optimization")
        
        state["current_step"] = "completed"
        state["progress"] = 1.0
        
        try:
            # Set final resume content
            state["final_resume"] = state.get("optimized_content", state["original_content"])
            
            # Calculate overall improvement score
            ats_score = state.get("ats_analysis", {}).get("score", 70)
            keyword_score = min(100, len(state.get("keyword_analysis", {}).get("current_keywords", [])) * 10)
            format_score = state.get("format_analysis", {}).get("format_score", 75)
            
            overall_score = (ats_score + keyword_score + format_score) / 3
            state["overall_score"] = overall_score
            
            # Complete optimization summary
            state["optimization_summary"].update({
                "overall_score": overall_score,
                "ats_score": ats_score,
                "keyword_score": keyword_score,
                "format_score": format_score,
                "total_improvements": len(state.get("improvements_made", [])),
                "processing_time": datetime.now().isoformat(),
                "warnings_count": len(state.get("warnings", [])),
                "errors_count": len(state.get("errors", []))
            })
            
        except Exception as e:
            logger.error(f"Finalization error: {e}")
            state["errors"].append(f"Finalization error: {str(e)}")
            state["final_resume"] = state.get("optimized_content", state["original_content"])
        
        return state
    
    def _handle_error(self, state: ResumeWorkflowState) -> ResumeWorkflowState:
        """Handle workflow errors"""
        logger.error("Handling workflow error")
        
        state["current_step"] = "error"
        state["final_resume"] = state.get("original_content", "")
        state["optimization_summary"] = {
            "status": "error",
            "errors": state.get("errors", []),
            "partial_results": True
        }
        
        return state
    
    # Conditional edge functions
    def _should_continue_after_validation(self, state: ResumeWorkflowState) -> str:
        """Determine if workflow should continue after validation"""
        return "error" if state.get("errors") else "continue"
    
    def _should_validate_optimization(self, state: ResumeWorkflowState) -> str:
        """Determine if optimization should be validated"""
        return "validate" if state.get("optimized_content") else "finalize"
    
    def _should_reoptimize(self, state: ResumeWorkflowState) -> str:
        """Determine if reoptimization is needed"""
        validation = state.get("optimization_summary", {}).get("validation", {})
        failed_checks = [k for k, v in validation.items() if not v]
        
        # Allow one reoptimization attempt
        if failed_checks and not state.get("reoptimization_attempted"):
            state["reoptimization_attempted"] = True
            return "reoptimize"
        
        return "finalize"
    
    def run_workflow(self, resume_content: str, job_field: str = None, user_preferences: Dict = None) -> Dict[str, Any]:
        """Run the complete optimization workflow"""
        
        # Prepare initial state
        initial_state = ResumeWorkflowState(
            original_content=resume_content,
            job_field=job_field or self.job_field,
            user_preferences=user_preferences or {},
            current_step="starting",
            progress=0.0,
            errors=[],
            warnings=[]
        )
        
        try:
            # Execute the workflow
            final_state = self.graph.invoke(initial_state)
            
            return {
                "success": True,
                "final_state": final_state,
                "optimized_resume": final_state.get("final_resume", ""),
                "improvements": final_state.get("improvements_made", []),
                "recommendations": final_state.get("recommendations", []),
                "scores": {
                    "overall": final_state.get("overall_score", 0),
                    "ats": final_state.get("ats_analysis", {}).get("score", 0),
                    "keywords": final_state.get("keyword_analysis", {}).get("keyword_density", 0) * 100,
                    "format": final_state.get("format_analysis", {}).get("format_score", 0)
                },
                "summary": final_state.get("optimization_summary", {}),
                "warnings": final_state.get("warnings", []),
                "errors": final_state.get("errors", [])
            }
            
        except Exception as e:
            logger.error(f"Workflow execution error: {e}")
            return {
                "success": False,
                "error": str(e),
                "optimized_resume": resume_content,  # Return original on error
                "improvements": [],
                "recommendations": [],
                "scores": {"overall": 0, "ats": 0, "keywords": 0, "format": 0},
                "summary": {"status": "failed", "error": str(e)},
                "warnings": [],
                "errors": [str(e)]
            }
    
    def get_workflow_status(self, state: ResumeWorkflowState) -> Dict[str, Any]:
        """Get current workflow status"""
        return {
            "current_step": state.get("current_step", "unknown"),
            "progress": state.get("progress", 0.0),
            "errors": state.get("errors", []),
            "warnings": state.get("warnings", []),
            "estimated_completion": self._estimate_completion_time(state)
        }
    
    def _estimate_completion_time(self, state: ResumeWorkflowState) -> int:
        """Estimate remaining completion time in seconds"""
        progress = state.get("progress", 0.0)
        if progress >= 1.0:
            return 0
        
        # Rough estimation based on current progress
        total_estimated_time = 60  # 60 seconds total
        remaining_progress = 1.0 - progress
        return int(total_estimated_time * remaining_progress)