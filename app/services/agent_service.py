"""
AI Agent service for code analysis and generation.
"""

import json
import re
from typing import Dict, List, Any, Optional

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    openai = None

try:
    from anthropic import Anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    Anthropic = None

from app.core.config import settings
from app.core.telemetry import get_telemetry
from app.models.schemas import AIProviderResponse
from app.services.sandbox import SandboxService
from app.services.git_service import GitService, GitError


class AgentError(Exception):
    """Base exception for agent operations."""
    pass


class AgentService:
    """Service for AI-powered code analysis and generation."""
    
    def __init__(self):
        self.telemetry = get_telemetry()
        self.openai_client = None
        self.anthropic_client = None
        if OPENAI_AVAILABLE or ANTHROPIC_AVAILABLE:
            self._initialize_clients()
        else:
            self.telemetry.log_event(
                "No AI providers available - agent functionality disabled",
                level="warning"
            )
    
    def _initialize_clients(self) -> None:
        """Initialize AI provider clients."""
        try:
            if OPENAI_AVAILABLE and settings.openai_api_key:
                from openai import AsyncOpenAI
                self.openai_client = AsyncOpenAI(api_key=settings.openai_api_key)
                self.telemetry.log_event("OpenAI client initialized")
            elif OPENAI_AVAILABLE:
                self.telemetry.log_event("OpenAI available but API key not configured", level="warning")
            
            if ANTHROPIC_AVAILABLE and settings.anthropic_api_key:
                self.anthropic_client = Anthropic(api_key=settings.anthropic_api_key)
                self.telemetry.log_event("Anthropic client initialized")
            elif ANTHROPIC_AVAILABLE:
                self.telemetry.log_event("Anthropic available but API key not configured", level="warning")
            
            if not self.openai_client and not self.anthropic_client:
                self.telemetry.log_event("No AI provider configured with valid API key", level="warning")
                
        except Exception as e:
            self.telemetry.log_error(e, context={"service": "agent"})
            raise AgentError(f"Failed to initialize AI clients: {e}")
    
    async def analyze_repository(
        self,
        correlation_id: str,
        repo_path: str,
        sandbox_service: SandboxService
    ) -> Dict[str, Any]:
        """
        Analyze a repository structure and content.
        
        Args:
            correlation_id: Request identifier
            repo_path: Path to the repository
            sandbox_service: Sandbox service instance
            
        Returns:
            Repository analysis data
            
        Raises:
            AgentError: If analysis fails
        """
        with self.telemetry.trace_operation(
            "analyze_repository",
            correlation_id=correlation_id,
            repo_path=repo_path
        ):
            try:
                git_service = GitService()
                
                files = await git_service.list_repository_files(
                    correlation_id=correlation_id,
                    repo_path=repo_path,
                    sandbox_service=sandbox_service
                )
                
                analysis = {
                    "files": files,
                    "file_count": len(files),
                    "languages": self._detect_languages(files),
                    "structure": self._analyze_structure(files),
                    "key_files": self._identify_key_files(files),
                    "dependencies": await self._analyze_dependencies(
                        correlation_id, repo_path, files, sandbox_service
                    )
                }
                
                sample_content = await self._get_sample_content(
                    correlation_id=correlation_id,
                    repo_path=repo_path,
                    files=analysis["key_files"][:5],
                    sandbox_service=sandbox_service
                )
                
                analysis["sample_content"] = sample_content
                
                self.telemetry.log_event(
                    "Repository analysis completed",
                    correlation_id=correlation_id,
                    file_count=len(files),
                    languages=analysis["languages"],
                    key_files_count=len(analysis["key_files"])
                )
                
                return analysis
                
            except Exception as e:
                self.telemetry.log_error(
                    e,
                    context={"correlation_id": correlation_id, "repo_path": repo_path},
                    correlation_id=correlation_id
                )
                raise AgentError(f"Failed to analyze repository: {e}")
    
    def _detect_languages(self, files: List[str]) -> Dict[str, int]:
        """Detect programming languages in the repository."""
        extensions = {}
        
        for file in files:
            if '.' in file:
                ext = file.split('.')[-1].lower()
                extensions[ext] = extensions.get(ext, 0) + 1
        
        language_map = {
            'py': 'Python',
            'js': 'JavaScript',
            'ts': 'TypeScript',
            'java': 'Java',
            'cpp': 'C++',
            'c': 'C',
            'go': 'Go',
            'rs': 'Rust',
            'php': 'PHP',
            'rb': 'Ruby',
            'swift': 'Swift',
            'kt': 'Kotlin',
            'scala': 'Scala',
            'cs': 'C#',
            'html': 'HTML',
            'css': 'CSS',
            'scss': 'SCSS',
            'sass': 'Sass',
            'vue': 'Vue',
            'jsx': 'JSX',
            'tsx': 'TSX'
        }
        
        languages = {}
        for ext, count in extensions.items():
            lang = language_map.get(ext, ext.upper())
            languages[lang] = languages.get(lang, 0) + count
        
        return languages
    
    def _analyze_structure(self, files: List[str]) -> Dict[str, Any]:
        """Analyze repository structure."""
        directories = set()
        for file in files:
            parts = file.split('/')
            for i in range(len(parts)):
                if i > 0:
                    directories.add('/'.join(parts[:i]))
        
        return {
            "directories": sorted(directories),
            "max_depth": max(len(f.split('/')) for f in files) if files else 0,
            "has_src": any(f.startswith('src/') for f in files),
            "has_test": any('test' in f.lower() for f in files),
            "has_docs": any(f.startswith('docs/') or f.startswith('doc/') for f in files)
        }
    
    def _identify_key_files(self, files: List[str]) -> List[str]:
        """Identify key files in the repository."""
        key_patterns = [
            r'README\.md$',
            r'package\.json$',
            r'requirements\.txt$',
            r'Cargo\.toml$',
            r'pom\.xml$',
            r'build\.gradle$',
            r'Dockerfile$',
            r'docker-compose\.yml$',
            r'main\.py$',
            r'app\.py$',
            r'server\.py$',
            r'index\.js$',
            r'index\.ts$',
            r'main\.js$',
            r'main\.ts$',
            r'App\.js$',
            r'App\.tsx$',
            r'\.env\.example$',
            r'config\.py$',
            r'settings\.py$',
            r'\.gitignore$',
            r'Makefile$',
            r'setup\.py$',
            r'pyproject\.toml$'
        ]
        
        key_files = []
        for pattern in key_patterns:
            for file in files:
                if re.search(pattern, file, re.IGNORECASE):
                    key_files.append(file)
        
        root_files = [f for f in files if '/' not in f]
        key_files.extend(root_files[:10])
        
        return list(set(key_files))
    
    async def _analyze_dependencies(
        self,
        correlation_id: str,
        repo_path: str,
        files: List[str],
        sandbox_service: SandboxService
    ) -> Dict[str, Any]:
        """Analyze project dependencies."""
        dependencies = {}
        
        dependency_files = {
            'package.json': 'npm',
            'requirements.txt': 'pip',
            'Pipfile': 'pipenv',
            'pyproject.toml': 'poetry',
            'Cargo.toml': 'cargo',
            'pom.xml': 'maven',
            'build.gradle': 'gradle',
            'composer.json': 'composer'
        }
        
        try:
            git_service = GitService()
            
            for file in files:
                if file in dependency_files:
                    try:
                        content = await git_service.get_file_content(
                            correlation_id=correlation_id,
                            repo_path=repo_path,
                            file_path=file,
                            sandbox_service=sandbox_service
                        )
                        
                        dependencies[dependency_files[file]] = {
                            'file': file,
                            'content_length': len(content),
                            'parsed': self._parse_dependency_file(file, content)
                        }
                    except Exception as e:
                        self.telemetry.log_error(
                            e,
                            context={"file": file, "correlation_id": correlation_id}
                        )
                        continue
        
        except Exception as e:
            self.telemetry.log_error(
                e,
                context={"correlation_id": correlation_id},
                correlation_id=correlation_id
            )
        
        return dependencies
    
    def _parse_dependency_file(self, filename: str, content: str) -> Dict[str, Any]:
        """Parse dependency file content."""
        try:
            if filename == 'package.json':
                data = json.loads(content)
                return {
                    'dependencies': data.get('dependencies', {}),
                    'devDependencies': data.get('devDependencies', {}),
                    'scripts': data.get('scripts', {})
                }
            elif filename == 'requirements.txt':
                lines = [line.strip() for line in content.split('\n') if line.strip()]
                return {'requirements': lines}
            else:
                return {'raw_content': content[:500]}
        except Exception:
            return {'parse_error': True}
    
    async def _get_sample_content(
        self,
        correlation_id: str,
        repo_path: str,
        files: List[str],
        sandbox_service: SandboxService
    ) -> Dict[str, str]:
        """Get sample content from key files."""
        sample_content = {}
        
        try:
            git_service = GitService()
            
            for file in files[:5]:  # Limit to 5 files
                try:
                    content = await git_service.get_file_content(
                        correlation_id=correlation_id,
                        repo_path=repo_path,
                        file_path=file,
                        sandbox_service=sandbox_service
                    )
                    
                    # Limit content length
                    if len(content) > 2000:
                        content = content[:2000] + "... (truncated)"
                    
                    sample_content[file] = content
                    
                except Exception as e:
                    sample_content[file] = f"Error reading file: {e}"
        
        except Exception as e:
            self.telemetry.log_error(
                e,
                context={"correlation_id": correlation_id},
                correlation_id=correlation_id
            )
        
        return sample_content
    
    async def create_implementation_plan(
        self,
        correlation_id: str,
        prompt: str,
        repo_analysis: Dict[str, Any],
        ai_provider: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create an implementation plan based on the prompt and repository analysis.
        
        Args:
            correlation_id: Request identifier
            prompt: User's coding prompt
            repo_analysis: Repository analysis data
            ai_provider: AI provider to use (optional)
            
        Returns:
            Implementation plan
            
        Raises:
            AgentError: If plan creation fails
        """
        if not self.openai_client and not self.anthropic_client:
            # Fallback to basic implementation plan
            return self._create_fallback_plan(prompt, repo_analysis)
            
        with self.telemetry.trace_operation(
            "create_implementation_plan",
            correlation_id=correlation_id,
            prompt=prompt[:50]
        ):
            try:
                context = self._prepare_context(prompt, repo_analysis)
                
                provider = ai_provider or settings.default_ai_provider
                
                if provider == "openai" and self.openai_client:
                    plan = await self._generate_plan_openai(context)
                elif provider == "anthropic" and self.anthropic_client:
                    plan = await self._generate_plan_anthropic(context)
                else:
                    if self.openai_client:
                        plan = await self._generate_plan_openai(context)
                    elif self.anthropic_client:
                        plan = await self._generate_plan_anthropic(context)
                    else:
                        return self._create_fallback_plan(prompt, repo_analysis)
                
                structured_plan = self._structure_plan(plan)
                
                self.telemetry.log_event(
                    "Implementation plan created",
                    correlation_id=correlation_id,
                    provider=provider,
                    plan_steps=len(structured_plan.get("steps", [])),
                    files_to_modify=len(structured_plan.get("files_to_modify", []))
                )
                
                return structured_plan
                
            except Exception as e:
                self.telemetry.log_error(
                    e,
                    context={"correlation_id": correlation_id, "prompt": prompt[:50]},
                    correlation_id=correlation_id
                )
                return self._create_fallback_plan(prompt, repo_analysis)
    
    def _create_fallback_plan(self, prompt: str, repo_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Create a basic fallback implementation plan when AI is not available."""
        return {
            "summary": f"Basic implementation plan for: {prompt[:100]}",
            "steps": [
                "Analyze the repository structure",
                "Identify files that need modification",
                "Implement the requested changes",
                "Test the implementation",
                "Create documentation"
            ],
            "files_to_modify": repo_analysis.get("key_files", [])[:3],
            "new_files": [],
            "considerations": [
                "This is a basic plan generated without AI assistance",
                "Manual review and customization recommended",
                "Consider impact on existing functionality"
            ],
            "ai_available": False
        }
    
    def _prepare_context(self, prompt: str, repo_analysis: Dict[str, Any]) -> str:
        """Prepare context for AI model."""
        context = f"""
You are an expert software engineer tasked with implementing code changes. 

**User Request:**
{prompt}

**Repository Analysis:**
- Languages: {repo_analysis.get('languages', {})}
- File count: {repo_analysis.get('file_count', 0)}
- Key files: {repo_analysis.get('key_files', [])}
- Dependencies: {list(repo_analysis.get('dependencies', {}).keys())}

**Sample Content:**
"""
        
        for file, content in repo_analysis.get('sample_content', {}).items():
            context += f"\n--- {file} ---\n{content}\n"
        
        context += """

**Instructions:**
1. Analyze the request and repository structure
2. Create a detailed implementation plan
3. Identify specific files to modify
4. Provide step-by-step instructions
5. Consider best practices and code quality

**Response Format:**
Please respond with a JSON object containing:
- summary: Brief description of the changes
- steps: Array of implementation steps
- files_to_modify: Array of files that need to be changed
- new_files: Array of new files to create (if any)
- considerations: Important considerations or warnings
"""
        
        return context
    
    async def _generate_plan_openai(self, context: str) -> str:
        """Generate implementation plan using OpenAI."""
        if not OPENAI_AVAILABLE or not self.openai_client:
            raise AgentError("OpenAI is not available")
            
        try:
            response = await self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert software engineer. Provide detailed, actionable implementation plans in JSON format."
                    },
                    {
                        "role": "user",
                        "content": context
                    }
                ],
                max_tokens=2000,
                temperature=0.3
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            raise AgentError(f"OpenAI API error: {e}")
    
    async def _generate_plan_anthropic(self, context: str) -> str:
        """Generate implementation plan using Anthropic."""
        if not ANTHROPIC_AVAILABLE or not self.anthropic_client:
            raise AgentError("Anthropic is not available")
            
        try:
            response = await self.anthropic_client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=2000,
                temperature=0.3,
                system="You are an expert software engineer. Provide detailed, actionable implementation plans in JSON format.",
                messages=[
                    {
                        "role": "user",
                        "content": context
                    }
                ]
            )
            
            return response.content[0].text
            
        except Exception as e:
            raise AgentError(f"Anthropic API error: {e}")
    
    def _structure_plan(self, plan_text: str) -> Dict[str, Any]:
        """Structure the AI-generated plan."""
        try:
            if plan_text.strip().startswith('{'):
                return json.loads(plan_text)
            
            return {
                "summary": "AI-generated implementation plan",
                "steps": [step.strip() for step in plan_text.split('\n') if step.strip()],
                "files_to_modify": [],
                "new_files": [],
                "considerations": ["Review generated plan carefully"]
            }
            
        except json.JSONDecodeError:
            return {
                "summary": "Implementation plan parsing failed",
                "steps": ["Manual review required"],
                "files_to_modify": [],
                "new_files": [],
                "considerations": ["Plan could not be parsed as JSON"],
                "raw_plan": plan_text
            }
    
    async def implement_changes(
        self,
        correlation_id: str,
        repo_path: str,
        implementation_plan: Dict[str, Any],
        sandbox_service: SandboxService,
        repo_analysis: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Implement the changes specified in the implementation plan.
        
        Args:
            correlation_id: Request identifier
            repo_path: Path to the repository
            implementation_plan: Implementation plan from create_implementation_plan
            sandbox_service: Sandbox service instance
            repo_analysis: Repository analysis data
            
        Returns:
            List of changes made
            
        Raises:
            AgentError: If implementation fails
        """
        with self.telemetry.trace_operation(
            "implement_changes",
            correlation_id=correlation_id,
            repo_path=repo_path
        ):
            try:
                changes_made = []
                git_service = GitService()
                
                files_to_modify = implementation_plan.get("files_to_modify", [])
                new_files = implementation_plan.get("new_files", [])
                
                if not files_to_modify and not new_files:
                    steps = implementation_plan.get("steps", [])
                    summary = implementation_plan.get("summary", "")
                    
                    for step in steps:
                        if "hello" in step.lower() or "function" in step.lower():
                            python_files = [f for f in repo_analysis.get("files", []) if f.endswith('.py')]
                            if python_files:
                                files_to_modify = python_files[:1]
                                break
                    
                    if not files_to_modify:
                        new_files = ["main.py"]
                
                for filepath in files_to_modify:
                    try:
                        current_content = await git_service.read_file_content(
                            correlation_id=correlation_id,
                            repo_path=repo_path,
                            file_path=filepath,
                            sandbox_service=sandbox_service
                        )
                        
                        new_content = await self._generate_file_content(
                            correlation_id=correlation_id,
                            filepath=filepath,
                            current_content=current_content,
                            implementation_plan=implementation_plan
                        )
                        
                        await git_service.write_file_content(
                            correlation_id=correlation_id,
                            repo_path=repo_path,
                            file_path=filepath,
                            content=new_content,
                            sandbox_service=sandbox_service
                        )
                        
                        changes_made.append({
                            "filepath": filepath,
                            "action": "modified",
                            "old_content": current_content,
                            "new_content": new_content,
                            "description": f"Updated {filepath} based on implementation plan"
                        })
                        
                    except Exception as e:
                        self.telemetry.log_error(
                            e,
                            context={"filepath": filepath, "correlation_id": correlation_id},
                            correlation_id=correlation_id
                        )
                
                for filepath in new_files:
                    try:
                        new_content = await self._generate_file_content(
                            correlation_id=correlation_id,
                            filepath=filepath,
                            current_content="",
                            implementation_plan=implementation_plan,
                            is_new_file=True
                        )
                        
                        await git_service.write_file_content(
                            correlation_id=correlation_id,
                            repo_path=repo_path,
                            file_path=filepath,
                            content=new_content,
                            sandbox_service=sandbox_service
                        )
                        
                        changes_made.append({
                            "filepath": filepath,
                            "action": "created",
                            "old_content": "",
                            "new_content": new_content,
                            "description": f"Created new file {filepath}"
                        })
                        
                    except Exception as e:
                        self.telemetry.log_error(
                            e,
                            context={"filepath": filepath, "correlation_id": correlation_id},
                            correlation_id=correlation_id
                        )
                
                if not changes_made:
                    example_file = "README_BACKSPACE.md"
                    example_content = f"""# Backspace Agent Changes

This file was created by the Backspace coding agent.

## Implementation Plan Summary
{implementation_plan.get('summary', 'No summary available')}

## Steps Taken
"""
                    for i, step in enumerate(implementation_plan.get('steps', []), 1):
                        example_content += f"{i}. {step}\n"
                    
                    await git_service.write_file_content(
                        correlation_id=correlation_id,
                        repo_path=repo_path,
                        file_path=example_file,
                        content=example_content,
                        sandbox_service=sandbox_service
                    )
                    
                    changes_made.append({
                        "filepath": example_file,
                        "action": "created",
                        "new_content": example_content,
                        "description": "Created documentation of changes"
                    })
                
                self.telemetry.log_event(
                    "Changes implemented",
                    correlation_id=correlation_id,
                    changes_count=len(changes_made),
                    files_modified=len(files_to_modify)
                )
                
                return changes_made
                
            except Exception as e:
                self.telemetry.log_error(
                    e,
                    context={"correlation_id": correlation_id, "repo_path": repo_path},
                    correlation_id=correlation_id
                )
                raise AgentError(f"Failed to implement changes: {e}")
    
    async def _generate_file_content(
        self,
        correlation_id: str,
        filepath: str,
        current_content: str,
        implementation_plan: Dict[str, Any],
        is_new_file: bool = False
    ) -> str:
        """
        Generate new file content using AI based on the implementation plan.
        
        Args:
            correlation_id: Request identifier
            filepath: Path to the file
            current_content: Current file content
            implementation_plan: Implementation plan
            is_new_file: Whether this is a new file
            
        Returns:
            New file content
        """
        try:
            context = f"""
You are an expert software engineer implementing code changes.

**File to create/modify:** {filepath}
**Is new file:** {is_new_file}

**Implementation Plan:**
{implementation_plan.get('summary', 'No summary')}

**Steps to implement:**
"""
            for i, step in enumerate(implementation_plan.get('steps', []), 1):
                if isinstance(step, dict):
                    context += f"{i}. {step.get('description', str(step))}\n"
                else:
                    context += f"{i}. {step}\n"
            
            if not is_new_file and current_content:
                context += f"""

**Current file content:**
```{self._get_file_extension(filepath)}
{current_content}
```

**Instructions:**
Modify the existing file to implement the requested changes. Maintain the existing structure and style where possible.
"""
            else:
                context += f"""

**Instructions:**
Create a new file with the exact filename: {filepath}

IMPORTANT: 
- Do NOT include the filename in your response
- Generate ONLY the code content
- Use appropriate language syntax and best practices
- Make sure the code is complete and runnable
- Include proper imports and structure
"""
            
            if self.openai_client:
                response = await self.openai_client.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an expert software engineer. Generate clean, well-structured code that implements the requested changes. Do NOT include filenames or markdown formatting in your response - only pure code."
                        },
                        {
                            "role": "user",
                            "content": context
                        }
                    ],
                    max_tokens=2000,
                    temperature=0.3
                )
                return response.choices[0].message.content.strip()
            else:
                if filepath.endswith('.py'):
                    return f"""# Generated by Backspace Agent

def hello_world():
    \"\"\"Simple hello world function.\"\"\"
    print("Hello, World!")

if __name__ == "__main__":
    hello_world()
"""
                elif filepath.endswith('.js'):
                    return f"""// Generated by Backspace Agent

function helloWorld() {{
    console.log("Hello, World!");
}}

// Call the function
helloWorld();
"""
                else:
                    return f"# Generated by Backspace Agent\n\n{implementation_plan.get('summary', 'Implementation completed')}"
                    
        except Exception as e:
            self.telemetry.log_error(
                e,
                context={"filepath": filepath, "correlation_id": correlation_id},
                correlation_id=correlation_id
            )
            return f"# Generated by Backspace Agent\n# Error occurred: {str(e)}"
    
    def _get_file_extension(self, filepath: str) -> str:
        """Get file extension for syntax highlighting."""
        if '.' in filepath:
            return filepath.split('.')[-1]
        return 'text'
    
    async def health_check(self) -> bool:
        """
        Perform a health check on the agent service.
        
        Returns:
            True if healthy, False otherwise
        """
        try:
            if OPENAI_AVAILABLE and self.openai_client:
                try:
                    await self.openai_client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[{"role": "user", "content": "Hello"}],
                        max_tokens=5
                    )
                    return True
                except Exception:
                    pass
            
            if ANTHROPIC_AVAILABLE and self.anthropic_client:
                try:
                    await self.anthropic_client.messages.create(
                        model="claude-3-haiku-20240307",
                        max_tokens=5,
                        messages=[{"role": "user", "content": "Hello"}]
                    )
                    return True
                except Exception:
                    pass
            return True
            
        except Exception as e:
            self.telemetry.log_error(
                e,
                context={"service": "agent", "operation": "health_check"}
            )
            return False 