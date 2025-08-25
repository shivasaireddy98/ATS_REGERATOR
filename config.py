from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Optional

class JobField(Enum):
    TECHNOLOGY = "Technology"
    MARKETING = "Marketing"
    FINANCE = "Finance"
    HEALTHCARE = "Healthcare"
    EDUCATION = "Education"

@dataclass
class ATSConfig:
    min_keyword_density: float = 0.02
    max_keyword_density: float = 0.05
    target_sections: List[str] = None
    
    def __post_init__(self):
        if self.target_sections is None:
            self.target_sections = ["contact", "summary", "experience", "education", "skills"]

@dataclass
class CrewAIConfig:
    verbose: bool = True
    allow_delegation: bool = False
    temperature: float = 0.7

# Industry-specific keywords
INDUSTRY_KEYWORDS = {
    JobField.TECHNOLOGY: {
        "programming": ["Python", "JavaScript", "Java", "React", "Node.js", "SQL", "AWS", "Docker"],
        "skills": ["API", "Machine Learning", "Data Analysis", "Cloud Computing", "DevOps"],
        "methodologies": ["Agile", "Scrum", "CI/CD", "Microservices", "REST"]
    },
    JobField.MARKETING: {
        "digital": ["SEO", "SEM", "Social Media", "Google Analytics", "Content Marketing"],
        "strategy": ["Brand Management", "Campaign Management", "Market Research"],
        "tools": ["HubSpot", "Salesforce", "Adobe Creative Suite", "Google Ads"]
    },
    JobField.FINANCE: {
        "analysis": ["Financial Modeling", "Risk Analysis", "Investment Analysis"],
        "tools": ["Excel", "Bloomberg", "SAP", "QuickBooks", "Tableau"],
        "regulations": ["GAAP", "SOX", "Compliance", "Audit"]
    },
    JobField.HEALTHCARE: {
        "clinical": ["Patient Care", "Medical Records", "HIPAA", "Clinical Research"],
        "technical": ["EMR", "Medical Imaging", "Laboratory", "Pharmacy"],
        "certifications": ["CPR", "BLS", "ACLS", "Licensed"]
    },
    JobField.EDUCATION: {
        "teaching": ["Curriculum Development", "Lesson Planning", "Assessment", "Classroom Management"],
        "technology": ["Learning Management Systems", "Educational Technology", "Online Learning"],
        "skills": ["Student Engagement", "Differentiated Instruction", "Data Analysis"]
    }
}

ATS_FORMATTING_RULES = {
    "avoid_tables": "Tables may not parse correctly in ATS systems",
    "use_standard_fonts": "Use Arial, Calibri, or Times New Roman",
    "simple_formatting": "Avoid complex formatting, graphics, or unusual layouts",
    "standard_sections": "Use standard section headers like 'Experience', 'Education', 'Skills'",
    "consistent_dates": "Use consistent date formats throughout"
}

ATS_SCORING_CRITERIA = {
    "keyword_relevance": 30,
    "format_compatibility": 25,
    "section_structure": 20,
    "readability": 15,
    "completeness": 10
}

AGENT_PROMPTS = {
    "content_parser": """You are an expert resume content parser with advanced text analysis capabilities. 
    Your role is to accurately extract, structure, and analyze resume content to provide detailed insights 
    for optimization.""",
    
    "ats_analyzer": """You are an ATS compatibility specialist with deep knowledge of how Applicant 
    Tracking Systems parse and rank resumes. You provide actionable insights to improve ATS compatibility.""",
    
    "keyword_optimizer": """You are a keyword optimization expert who understands industry-specific 
    terminology and how to naturally integrate keywords for maximum ATS effectiveness.""",
    
    "format_optimizer": """You are a professional resume formatting specialist who ensures resumes 
    have optimal layout, consistent styling, and perfect compatibility with both ATS systems and 
    human reviewers."""
}
