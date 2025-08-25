# agents.py - Enhanced CrewAI Agents with Specialized Tools
import re
import json
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import nltk
from collections import Counter
import textstat

from crewai import Agent, Task, Crew, Process
from langchain.tools import Tool
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage

from config import (
    ATSConfig, CrewAIConfig, INDUSTRY_KEYWORDS, 
    ATS_FORMATTING_RULES, ATS_SCORING_CRITERIA, AGENT_PROMPTS
)

# Custom tools for agents
class ResumeAnalysisTools:
    """Collection of specialized tools for resume analysis"""
    
    @staticmethod
    def extract_contact_info(text: str) -> Dict[str, str]:
        """Extract contact information using regex patterns"""
        contact_info = {}
        
        # Email extraction
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = re.findall(email_pattern, text)
        if emails:
            contact_info['email'] = emails[0]
        
        # Phone extraction
        phone_pattern = r'(\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})'
        phones = re.findall(phone_pattern, text)
        if phones:
            contact_info['phone'] = ''.join(phones[0])
        
        # LinkedIn extraction
        linkedin_pattern = r'linkedin\.com/in/[\w-]+'
        linkedin = re.findall(linkedin_pattern, text, re.IGNORECASE)
        if linkedin:
            contact_info['linkedin'] = linkedin[0]
        
        return contact_info
    
    @staticmethod
    def analyze_sections(text: str) -> Dict[str, str]:
        """Identify and extract resume sections"""
        sections = {}
        
        # Common section headers
        section_patterns = {
            'summary': r'(professional summary|summary|profile|objective)',
            'experience': r'(work experience|professional experience|experience|employment)',
            'education': r'(education|academic background)',
            'skills': r'(skills|technical skills|core competencies)',
            'certifications': r'(certifications|certificates|licenses)',
            'projects': r'(projects|key projects|notable projects)'
        }
        
        text_lower = text.lower()
        lines = text.split('\n')
        
        for section_name, pattern in section_patterns.items():
            for i, line in enumerate(lines):
                if re.search(pattern, line.lower()) and len(line.strip()) < 50:
                    # Found section header, extract content until next section
                    section_content = []
                    for j in range(i+1, len(lines)):
                        next_line = lines[j].strip()
                        if not next_line:
                            continue
                        
                        # Check if this is another section header
                        is_header = any(re.search(p, next_line.lower()) 
                                      for p in section_patterns.values())
                        if is_header and len(next_line) < 50:
                            break
                        
                        section_content.append(next_line)
                    
                    sections[section_name] = '\n'.join(section_content)
                    break
        
        return sections
    
    @staticmethod
    def calculate_keyword_density(text: str, keywords: List[str]) -> Dict[str, float]:
        """Calculate keyword density for given keywords"""
        text_lower = text.lower()
        words = re.findall(r'\b\w+\b', text_lower)
        total_words = len(words)
        
        keyword_counts = {}
        for keyword in keywords:
            keyword_lower = keyword.lower()
            count = text_lower.count(keyword_lower)
            density = count / total_words if total_words > 0 else 0
            keyword_counts[keyword] = {
                'count': count,
                'density': density
            }
        
        return keyword_counts
    
    @staticmethod
    def assess_readability(text: str) -> Dict[str, float]:
        """Assess text readability using various metrics"""
        return {
            'flesch_reading_ease': textstat.flesch_reading_ease(text),
            'flesch_kincaid_grade': textstat.flesch_kincaid_grade(text),
            'automated_readability_index': textstat.automated_readability_index(text),
            'coleman_liau_index': textstat.coleman_liau_index(text)
        }

# Enhanced agents with specialized capabilities
class EnhancedResumeParser(Agent):
    """Enhanced content parser with advanced text analysis"""
    
    def __init__(self, llm, config: CrewAIConfig):
        self.tools = self._create_tools()
        
        super().__init__(
            role="Advanced Resume Content Parser",
            goal="Extract, structure, and analyze resume content with high precision",
            backstory=AGENT_PROMPTS["content_parser"] + """
            
            I use advanced text processing techniques to identify sections, extract contact
            information, and structure content for optimal analysis by other agents.
            """,
            verbose=config.verbose,
            allow_delegation=config.allow_delegation,
            llm=llm,
            tools=self.tools
        )
    
    def _create_tools(self) -> List[Tool]:
        """Create specialized tools for content parsing"""
        
        contact_extractor = Tool(
            name="extract_contact_info",
            description="Extract contact information (email, phone, LinkedIn) from text",
            func=ResumeAnalysisTools.extract_contact_info
        )
        
        section_analyzer = Tool(
            name="analyze_sections",
            description="Identify and extract resume sections (summary, experience, education, skills)",
            func=ResumeAnalysisTools.analyze_sections
        )
        
        return [contact_extractor, section_analyzer]

class EnhancedATSAnalyzer(Agent):
    """Enhanced ATS analyzer with comprehensive compatibility checking"""
    
    def __init__(self, llm, config: CrewAIConfig, ats_config: ATSConfig):
        self.ats_config = ats_config
        self.tools = self._create_tools()
        
        super().__init__(
            role="Advanced ATS Compatibility Specialist",
            goal="Provide comprehensive ATS compatibility analysis with actionable insights",
            backstory=AGENT_PROMPTS["ats_analyzer"] + """
            
            I perform deep analysis of resume compatibility with Applicant Tracking Systems,
            checking formatting, structure, keyword usage, and providing detailed scoring.
            """,
            verbose=config.verbose,
            allow_delegation=config.allow_delegation,
            llm=llm,
            tools=self.tools
        )
    
    def _create_tools(self) -> List[Tool]:
        """Create tools for ATS analysis"""
        
        def format_checker(text: str) -> Dict:
            """Check formatting compatibility with ATS systems"""
            issues = []
            score = 100
            
            # Check for problematic characters
            if re.search(r'[^\w\s\-\.\,\(\)\@\#\%\&\+\=\:\;\!\?]', text):
                issues.append("Contains special characters that may not parse correctly")
                score -= 15
            
            # Check section headers
            sections_found = len(ResumeAnalysisTools.analyze_sections(text))
            if sections_found < 3:
                issues.append("Missing standard resume sections")
                score -= 20
            
            # Check length
            word_count = len(text.split())
            if word_count > 800:
                issues.append("Resume may be too long for single page")
                score -= 10
            elif word_count < 200:
                issues.append("Resume content appears insufficient")
                score -= 15
            
            return {
                'format_score': max(0, score),
                'issues': issues,
                'sections_found': sections_found,
                'word_count': word_count
            }
        
        format_tool = Tool(
            name="check_ats_formatting",
            description="Check resume formatting for ATS compatibility",
            func=format_checker
        )
        
        return [format_tool]

class EnhancedKeywordOptimizer(Agent):
    """Enhanced keyword optimizer with industry-specific intelligence"""
    
    def __init__(self, llm, config: CrewAIConfig, job_field: str = "Technology"):
        self.job_field = job_field
        self.tools = self._create_tools()
        
        super().__init__(
            role="Advanced Keyword Optimization Expert",
            goal=f"Optimize resume keywords for {job_field} industry and ATS compatibility",
            backstory=AGENT_PROMPTS["keyword_optimizer"] + f"""
            
            I specialize in {job_field} industry keywords and understand how to naturally
            integrate them into resume content while maintaining readability and authenticity.
            """,
            verbose=config.verbose,
            allow_delegation=config.allow_delegation,
            llm=llm,
            tools=self.tools
        )
    
    def _create_tools(self) -> List[Tool]:
        """Create keyword optimization tools"""
        
        def keyword_analyzer(text: str) -> Dict:
            """Analyze keyword usage and suggest improvements"""
            from config import JobField
            
            # Get industry keywords
            job_field_enum = getattr(JobField, self.job_field.upper(), JobField.TECHNOLOGY)
            industry_keywords = INDUSTRY_KEYWORDS.get(job_field_enum, {})
            
            all_keywords = []
            for category, keywords in industry_keywords.items():
                all_keywords.extend(keywords)
            
            # Analyze current keyword usage
            keyword_analysis = ResumeAnalysisTools.calculate_keyword_density(text, all_keywords)
            
            # Identify missing high-value keywords
            used_keywords = [kw for kw, data in keyword_analysis.items() if data['count'] > 0]
            missing_keywords = [kw for kw in all_keywords if kw not in used_keywords]
            
            # Calculate overall keyword score
            total_density = sum(data['density'] for data in keyword_analysis.values())
            keyword_score = min(100, (total_density / 0.03) * 100)  # Target 3% density
            
            return {
                'keyword_score': keyword_score,
                'total_density': total_density,
                'used_keywords': used_keywords,
                'missing_keywords': missing_keywords[:10],  # Top 10 missing
                'keyword_analysis': keyword_analysis
            }
        
        keyword_tool = Tool(
            name="analyze_keywords",
            description="Analyze keyword usage and density for industry relevance",
            func=keyword_analyzer
        )
        
        return [keyword_tool]

class EnhancedFormatOptimizer(Agent):
    """Enhanced format optimizer with professional layout expertise"""
    
    def __init__(self, llm, config: CrewAIConfig):
        self.tools = self._create_tools()
        
        super().__init__(
            role="Advanced Resume Format Specialist",
            goal="Optimize resume formatting for professional appearance and ATS compatibility",
            backstory=AGENT_PROMPTS["format_optimizer"] + """
            
            I ensure resumes have professional formatting, consistent styling, optimal layout,
            and perfect compatibility with both ATS systems and human reviewers.
            """,
            verbose=config.verbose,
            allow_delegation=config.allow_delegation,
            llm=llm,
            tools=self.tools
        )
    
    def _create_tools(self) -> List[Tool]:
        """Create formatting optimization tools"""
        
        def layout_analyzer(text: str) -> Dict:
            """Analyze and suggest layout improvements"""
            lines = text.split('\n')
            issues = []
            suggestions = []
            
            # Check for consistent bullet points
            bullet_patterns = []
            for line in lines:
                stripped = line.strip()
                if stripped.startswith(('•', '-', '*', '◦')):
                    bullet_patterns.append(stripped[0])
            
            if len(set(bullet_patterns)) > 1:
                issues.append("Inconsistent bullet point formatting")
                suggestions.append("Use consistent bullet points throughout (• recommended)")
            
            # Check for excessive whitespace
            empty_lines = sum(1 for line in lines if not line.strip())
            if empty_lines > len(lines) * 0.3:
                issues.append("Excessive whitespace")
                suggestions.append("Reduce empty lines to improve space utilization")
            
            # Check line length for readability
            long_lines = sum(1 for line in lines if len(line) > 80)
            if long_lines > len(lines) * 0.2:
                issues.append("Some lines may be too long")
                suggestions.append("Consider breaking long lines for better readability")
            
            format_score = max(0, 100 - len(issues) * 15)
            
            return {
                'format_score': format_score,
                'issues': issues,
                'suggestions': suggestions,
                'bullet_consistency': len(set(bullet_patterns)) <= 1,
                'line_count': len(lines),
                'empty_line_ratio': empty_lines / len(lines) if lines else 0
            }
        
        layout_tool = Tool(
            name="analyze_layout",
            description="Analyze resume layout and formatting for optimization opportunities",
            func=layout_analyzer
        )
        
        return [layout_tool]

# Enhanced Crew with comprehensive analysis
class EnhancedResumeOptimizationCrew:
    """Enhanced crew with specialized agents and comprehensive analysis"""
    
    def __init__(self, llm, job_field: str = "Technology"):
        self.llm = llm
        self.job_field = job_field
        self.config = CrewAIConfig()
        self.ats_config = ATSConfig()
        
        self.agents = self._create_enhanced_agents()
        self.tasks = self._create_comprehensive_tasks()
        self.crew = self._create_crew()
    
    def _create_enhanced_agents(self) -> Dict[str, Agent]:
        """Create enhanced agents with specialized capabilities"""
        
        parser = EnhancedResumeParser(self.llm, self.config)
        ats_analyzer = EnhancedATSAnalyzer(self.llm, self.config, self.ats_config)
        keyword_optimizer = EnhancedKeywordOptimizer(self.llm, self.config, self.job_field)
        format_optimizer = EnhancedFormatOptimizer(self.llm, self.config)
        
        return {
            "parser": parser,
            "ats_analyzer": ats_analyzer,
            "keyword_optimizer": keyword_optimizer,
            "format_optimizer": format_optimizer
        }
    
    def _create_comprehensive_tasks(self) -> Dict[str, Task]:
        """Create comprehensive tasks for enhanced analysis"""
        
        parse_task = Task(
            description="""
            Perform comprehensive parsing of the resume content:
            1. Extract and structure all sections (contact, summary, experience, education, skills)
            2. Identify the resume format and layout structure
            3. Extract contact information using advanced pattern recognition
            4. Analyze content organization and completeness
            5. Provide detailed section breakdown with content analysis
            """,
            agent=self.agents["parser"],
            expected_output="Structured resume analysis with extracted sections and metadata"
        )
        
        ats_analysis_task = Task(
            description=f"""
            Conduct thorough ATS compatibility analysis:
            1. Check formatting compatibility with major ATS systems
            2. Analyze keyword usage and density for {self.job_field} industry
            3. Evaluate section structure and header recognition
            4. Assess overall parseability and data extraction potential
            5. Provide compatibility score (0-100) with detailed breakdown
            6. List specific issues and improvement recommendations
            """,
            agent=self.agents["ats_analyzer"],
            expected_output="Comprehensive ATS compatibility report with scores and recommendations"
        )
        
        keyword_optimization_task = Task(
            description=f"""
            Perform advanced keyword optimization for {self.job_field}:
            1. Analyze current keyword usage and density
            2. Identify industry-relevant keywords that are missing
            3. Suggest natural integration points for new keywords
            4. Optimize keyword density for ATS effectiveness (target 2-5%)
            5. Ensure keywords align with job market demands
            6. Provide specific keyword placement recommendations
            """,
            agent=self.agents["keyword_optimizer"],
            expected_output="Detailed keyword optimization strategy with specific recommendations"
        )
        
        format_optimization_task = Task(
            description="""
            Optimize resume formatting for maximum impact:
            1. Analyze current layout and formatting consistency
            2. Ensure ATS-friendly formatting standards
            3. Optimize bullet points, spacing, and section headers
            4. Suggest improvements for professional appearance
            5. Ensure single-page optimization where appropriate
            6. Provide specific formatting guidelines and corrections
            """,
            agent=self.agents["format_optimizer"],
            expected_output="Complete formatting optimization guide with specific improvements"
        )
        
        return {
            "parse": parse_task,
            "ats_analysis": ats_analysis_task,
            "keyword_optimization": keyword_optimization_task,
            "format_optimization": format_optimization_task
        }
    
    def _create_crew(self) -> Crew:
        """Create the enhanced crew"""
        return Crew(
            agents=list(self.agents.values()),
            tasks=list(self.tasks.values()),
            verbose=2,
            process=Process.sequential
        )
    
    def analyze_resume(self, resume_content: str) -> Dict:
        """Run comprehensive resume analysis"""
        try:
            # Prepare inputs for the crew
            inputs = {
                "resume_content": resume_content,
                "job_field": self.job_field,
                "analysis_depth": "comprehensive"
            }
            
            # Execute the crew analysis
            result = self.crew.kickoff(inputs=inputs)
            
            return {
                "success": True,
                "result": result,
                "job_field": self.job_field,
                "agents_used": list(self.agents.keys())
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "job_field": self.job_field
            }
    
    def get_agent_status(self) -> Dict[str, str]:
        """Get current status of all agents"""
        return {
            agent_name: "Ready" for agent_name in self.agents.keys()
        }