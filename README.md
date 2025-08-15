   # AI Coding Agent

   An AI-powered coding assistant that analyzes repositories, implements features, and creates pull requests automatically. Just describe what you want, and it'll write the code and integrate it properly.

   ## 🛠️ **Running Locally**

   ### **What You Need**
   - Python 3.11+
   - Docker (for sandboxing)
   - GitHub Personal Access Token ([create one here](https://github.com/settings/tokens))
   - OpenAI API key

   ### **Setup Steps**

   1. **Clone and configure:**
      ```bash
      git clone https://github.com/your-username/Backspace.git
      cd Backspace
      cp .env.example .env
      ```

   2. **Edit your `.env` file:**
      ```bash

         # REQUIRED CONFIGURATION
         GITHUB_TOKEN=
         OPENAI_API_KEY=
         DEFAULT_AI_PROVIDER=openai

         # Option 2: Anthropic (alternative to OpenAI)
         # ANTHROPIC_API_KEY=your_anthropic_api_key_here
         # DEFAULT_AI_PROVIDER=anthropic

       
         # SERVER CONFIGURATION
         # Application Settings
         APP_NAME=Backspace Coding Agent
         APP_VERSION=0.1.0
         DEBUG=true
         HOST=0.0.0.0
         PORT=8000

         # Logging Configuration
         LOG_LEVEL=INFO
         LOG_FORMAT=json

         # SANDBOX CONFIGURATION

         # Docker Settings
         SANDBOX_TIMEOUT=300
         MAX_CONCURRENT_JOBS=5
         MAX_REPO_SIZE_MB=100

         # SECURITY CONFIGURATION
         # Rate Limiting
         RATE_LIMIT_REQUESTS=10
         RATE_LIMIT_WINDOW=60

   
         # OBSERVABILITY CONFIGURATION
         # OpenTelemetry Tracing
         ENABLE_TRACING=false
         JAEGER_ENDPOINT=jaeger:14268

         # Langsmith Integration (Optional)
         LANGSMITH_API_KEY=
         # LANGSMITH_TRACING=true
         LANGSMITH_ENDPOINT="https://api.smith.langchain.com"
         LANGSMITH_PROJECT=backspace-agent
         # ENABLE_LANGSMITH=false
         DOCKER_IMAGE=python:3.11

      ```

   3. **Install and run:**
      ```bash
      # Install dependencies
      pip install -r requirements.txt
      
      # Start the application
      python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
      ```

   4. **Open your browser:** `http://localhost:8000`

   ## 🏗️ **Architecture Overview**

   Here's how I built the system using LangGraph + Docker sandboxing:

   ```mermaid
   graph TB
      A[Client Request] --> B[FastAPI Server]
      B --> C[Streaming Service]
      B --> D[LangGraph Agent]
      
      D --> E[Repository Analysis]
      D --> F[Planning Engine]
      D --> G[Implementation Engine]
      D --> H[Git Operations]
      
      G --> I[Docker Sandbox]
      I --> J[Tool Execution]
      J --> K[File Operations]
      J --> L[Command Execution]
      
      H --> M[GitHub API]
      M --> N[Pull Request Creation]
      
      C --> O[Server-Sent Events]
      O --> P[Real-time UI Updates]
      
      Q[LangSmith] --> D
      R[Telemetry] --> B
      
      style I fill:#e74c3c,stroke:#2c3e50,stroke-width:2px,color:#fff
      style D fill:#3498db,stroke:#2c3e50,stroke-width:2px,color:#fff
      style M fill:#27ae60,stroke:#2c3e50,stroke-width:2px,color:#fff
   ```

   ### **Component Breakdown**

   **FastAPI Server:** Handles HTTP requests, manages streaming connections, and orchestrates the overall workflow

   **LangGraph Agent:** The core AI agent that contains the entire 6-step workflow:
   - `analyze_repository` - Understands codebase structure and dependencies
   - `create_plan` - Generates detailed implementation plans
   - `implement_changes` - Writes code and executes changes using tools
   - `commit_changes` - Creates meaningful git commits
   - `push_changes` - Pushes to remote repository
   - `create_pull_request` - Generates PRs with detailed descriptions

   **Streaming Service:** Provides real-time progress updates via Server-Sent Events

   **Docker Sandbox:** Isolated execution environment where all code runs safely

   **Tools:** 7 specialized tools that LangGraph nodes can call:
   - `analyze_repository`, `read_file`, `write_file`, `execute_command`, `create_branch`, `commit_changes`, `push_changes`

   **Services:** Core services that power the system:
   - `SandboxService` - Manages Docker containers and command execution
   - `GitService` - Handles all git operations and GitHub API calls
   - `StreamingService` - Manages real-time event streaming

   **LangSmith Integration:** Traces every decision and tool call for debugging and optimization

   ### **Why This Architecture?**

   I chose the **LangGraph state machine approach** over other patterns for specific reasons:

   **The Problem:** Traditional AI agents often fail in production because they:
   - Make unreliable decisions without proper state management
   - Can't recover from failures gracefully  
   - Don't ensure complete task execution (might create files but forget integration)
   - Are difficult to debug when something goes wrong
   - Don't handle network failures or API rate limits well

   **Why LangGraph State Machine:**

   1. **Deterministic Execution:** Each step must complete successfully before moving to the next. No partial implementations or inconsistent states.

   2. **Robust Error Handling:** If any step fails, the `handle_error` node can retry the entire workflow up to 3 times. This handles network issues, API rate limits, and temporary failures automatically.

   3. **Complete Task Validation:** The `implement_changes` node uses an internal conversation loop to ensure BOTH file creation AND proper integration happen. Traditional agents often create files but forget to integrate them.

   4. **Full Observability:** Every decision, tool call, and state transition is traced through LangSmith. When something goes wrong, you can see exactly why the agent made specific choices.

   5. **Production Ready:** The state machine approach provides the reliability needed for production use cases where partial failures aren't acceptable.

   This architecture focuses on building an AI agent that reliably understands existing codebases, makes intelligent changes, integrates them properly, and handles failures gracefully.

   ### **LangGraph Workflow Architecture**

   Our agent uses a state-machine approach with LangGraph, ensuring deterministic execution:

   ```mermaid
   ---
   config:
     flowchart:
       curve: linear
   ---
   graph TD;
           __start__([<p>__start__</p>]):::first
           analyze_repository(analyze_repository)
           create_plan(create_plan)
           implement_changes(implement_changes)
           commit_changes(commit_changes)
           push_changes(push_changes)
           create_pull_request(create_pull_request)
           handle_error(handle_error)
           __end__([<p>__end__</p>]):::last
           __start__ --> analyze_repository;
           analyze_repository -. &nbsp;continue&nbsp; .-> create_plan;
           analyze_repository -. &nbsp;error&nbsp; .-> handle_error;
           commit_changes -. &nbsp;error&nbsp; .-> handle_error;
           commit_changes -. &nbsp;continue&nbsp; .-> push_changes;
           create_plan -. &nbsp;error&nbsp; .-> handle_error;
           create_plan -. &nbsp;continue&nbsp; .-> implement_changes;
           create_pull_request -. &nbsp;continue&nbsp; .-> __end__;
           create_pull_request -. &nbsp;error&nbsp; .-> handle_error;
           handle_error -. &nbsp;end&nbsp; .-> __end__;
           handle_error -. &nbsp;retry&nbsp; .-> analyze_repository;
           implement_changes -. &nbsp;continue&nbsp; .-> commit_changes;
           implement_changes -. &nbsp;error&nbsp; .-> handle_error;
           push_changes -. &nbsp;continue&nbsp; .-> create_pull_request;
           push_changes -. &nbsp;error&nbsp; .-> handle_error;
           classDef default fill:#f2f0ff,line-height:1.2
           classDef first fill-opacity:0
           classDef last fill:#bfb6fc
   ```

   ### **How the LangGraph Workflow Works**

   I designed the agent as a state machine with these sequential nodes:

   1. **`analyze_repository`** - Clones the repo and understands its structure, dependencies, and patterns
   2. **`create_plan`** - Generates a detailed implementation plan based on the user's request  
   3. **`implement_changes`** - Uses tools to actually write code, install packages, and make changes (includes internal logic for ensuring both new file creation and integration)
   4. **`commit_changes`** - Creates meaningful git commits
   5. **`push_changes`** - Pushes to remote repository  
   6. **`create_pull_request`** - Generates detailed PR descriptions and creates the pull request

   **Error Handling**: Each node has conditional edges - if any step fails, it routes to the `handle_error` node. This node can retry the entire workflow up to 3 times before giving up, making the system robust against network issues or temporary failures.

   **Integration Logic**: The complex integration validation (ensuring new files are created AND properly integrated into existing code) happens internally within the `implement_changes` node through a conversation loop with the LLM, not as separate workflow nodes.

   ### **Tool Arsenal**

   I built 7 tools that the agent can use:

   - **`analyze_repository`** - Understands codebase structure and existing dependencies
   - **`read_file`** - Safely reads files with validation
   - **`write_file`** - Creates and modifies files with proper integration checks
   - **`execute_command`** - Runs shell commands in sandboxed containers
   - **`commit_changes`** - Creates git commits with descriptive messages
   - **`push_changes`** - Pushes changes to remote repository
   - **`create_pull_request`** - Generates PRs with detailed descriptions

   Each tool is isolated, testable, and has specific security boundaries.

   ## 🐳 **Docker Sandbox Implementation**

   Every request creates an isolated Docker container with this configuration:

   ```python
   # Container configuration from app/services/sandbox.py
   container_config = {
       "image": "alpine:latest",
       "name": f"sandbox_{unique_id}",
       "volumes": {temp_dir: {"bind": "/workspace", "mode": "rw"}},
       "working_dir": "/workspace",
       "user": "root",
       "network_mode": "bridge",
       "mem_limit": "512m",
       "cpu_quota": 50000,
       "cpu_period": 100000,
       "pids_limit": 100,
       "read_only": False,
       "tmpfs": {"/tmp": "rw,noexec,nosuid,size=100m"},
       "environment": {
           "PYTHONUNBUFFERED": "1",
           "PYTHONDONTWRITEBYTECODE": "1",
           "HOME": "/workspace"
       },
       "command": ["sleep", "infinity"]
   }
   ```

   Commands use different users based on need:

   ```python
   # User selection logic from execute_command()
   if user is None:
       user = "root" if "apk" in command or "apt-get" in command else "1000:1000"
   ```

   **Design Decisions:**
   - **Alpine Linux**: Lightweight (5MB) but includes package manager for installing Git, Python, Node.js
   - **Root container + user switching**: Allows package installation while running user code safely  
   - **Resource limits**: 512MB RAM and 50% CPU prevent resource exhaustion
   - **Process limits**: 100 max processes prevent fork bombs
   - **Writable filesystem**: Required for package installs and builds, but `/tmp` uses `noexec,nosuid` to prevent execution of malicious files

   **Container Lifecycle:**
   - Creates new container for each request
   - Installs packages as root for system-level dependencies  
   - Runs user code as non-privileged user (1000:1000) to prevent system access
   - **Automatically destroys container after request completes (success or failure)**
   - Cleans up temporary directories and resources

   ---
