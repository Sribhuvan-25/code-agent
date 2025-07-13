# 🚀 Simple Production Deployment Guide

This guide will walk you through deploying your Backspace AI Coding Agent using Docker Compose. This approach is simple but includes important production features like health checks, resource limits, and proper networking.

## 📋 Prerequisites

Before you start, make sure you have:

- **Docker** installed and running
- **Docker Compose** installed
- **Git** installed
- **API Keys** ready:
  - OpenAI API Key OR Anthropic API Key
  - GitHub Personal Access Token

## 🏗️ Architecture Overview

Your deployment will include:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Nginx         │    │  Backspace      │    │   Jaeger        │
│   (Load Balancer)│────│   Agent         │────│   (Telemetry)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────┐
                    │   Redis         │
                    │   (Caching)     │
                    └─────────────────┘
```

## 🚀 Step-by-Step Deployment

### Step 1: Prepare Your Environment

1. **Clone your repository** (if not already done):
   ```bash
   git clone <your-repo-url>
   cd Backspace
   ```

2. **Set up environment variables**:
   ```bash
   # Copy the example environment file
   cp env.prod.example .env.prod
   
   # Edit the file with your actual API keys
   nano .env.prod
   ```

3. **Configure your API keys** in `.env.prod`:
   ```bash
   # Required: Choose one AI provider
   OPENAI_API_KEY=your_openai_api_key_here
   # OR
   ANTHROPIC_API_KEY=your_anthropic_api_key_here
   DEFAULT_AI_PROVIDER=openai
   
   # Required: GitHub token
   GITHUB_TOKEN=your_github_personal_access_token_here
   ```

### Step 2: Deploy with One Command

Use the deployment script for easy deployment:

```bash
# Make the script executable (if not already)
chmod +x deploy.sh

# Deploy everything
./deploy.sh
```

The script will:
- ✅ Check Docker and Docker Compose
- ✅ Verify environment configuration
- ✅ Build the application
- ✅ Start all services
- ✅ Check service health
- ✅ Show you the results

### Step 3: Verify Deployment

After deployment, you should see:

```
🎉 Deployment completed successfully!

📊 Service Information:
  • Main Application: http://localhost:8000
  • Nginx Load Balancer: http://localhost:80
  • Jaeger Telemetry: http://localhost:16686
  • Redis: localhost:6379
```

## 🔧 What Each Service Does

### 1. **Backspace Agent** (`backspace-agent`)
- **Purpose**: Your main AI coding agent application
- **Port**: 8000
- **Features**: 
  - LangGraph workflow execution
  - Docker sandbox management
  - GitHub API integration
  - Real-time streaming

### 2. **Nginx** (`nginx`)
- **Purpose**: Load balancer and reverse proxy
- **Port**: 80 (HTTP)
- **Features**:
  - Rate limiting (10 requests/second)
  - Health check routing
  - Static file serving
  - Request forwarding

### 3. **Jaeger** (`jaeger`)
- **Purpose**: Distributed tracing and telemetry
- **Port**: 16686 (UI), 14268 (API)
- **Features**:
  - Trace all LLM operations
  - Monitor performance
  - Debug issues

### 4. **Redis** (`redis`)
- **Purpose**: Caching and session management
- **Port**: 6379
- **Features**:
  - Store streaming session data
  - Cache frequently accessed data
  - Memory optimization

## 🛠️ Management Commands

### View Logs
```bash
# All services
./deploy.sh logs

# Specific service
docker-compose -f docker-compose.prod.yml logs -f backspace-agent
```

### Check Status
```bash
./deploy.sh status
```

### Restart Services
```bash
./deploy.sh restart
```

### Stop Everything
```bash
./deploy.sh stop
```

## 🔍 Monitoring Your Application

### 1. **Application Health**
```bash
curl http://localhost:8000/health
```

### 2. **Jaeger Telemetry**
- Open: http://localhost:16686
- View traces of all LLM operations
- Monitor performance bottlenecks

### 3. **Service Logs**
```bash
# Real-time logs
docker-compose -f docker-compose.prod.yml logs -f

# Specific service
docker-compose -f docker-compose.prod.yml logs -f backspace-agent
```

## 🚨 Troubleshooting

### Common Issues

#### 1. **Docker Socket Permission**
```bash
# If you get permission errors
sudo chmod 666 /var/run/docker.sock
```

#### 2. **Port Already in Use**
```bash
# Check what's using the port
lsof -i :8000

# Kill the process or change the port in docker-compose.prod.yml
```

#### 3. **Environment Variables Missing**
```bash
# Check if .env.prod exists and has required values
cat .env.prod | grep -E "(OPENAI_API_KEY|ANTHROPIC_API_KEY|GITHUB_TOKEN)"
```

#### 4. **Services Not Starting**
```bash
# Check service status
docker-compose -f docker-compose.prod.yml ps

# View detailed logs
docker-compose -f docker-compose.prod.yml logs
```

## 🔒 Security Considerations

### 1. **Environment Variables**
- Never commit `.env.prod` to version control
- Use strong, unique API keys
- Rotate keys regularly

### 2. **Network Security**
- Services communicate over internal Docker network
- Only necessary ports are exposed
- Rate limiting prevents abuse

### 3. **Docker Security**
- Non-root user in containers
- Resource limits prevent DoS
- Health checks ensure service availability

## 📈 Scaling Considerations

### Horizontal Scaling
To scale your application:

1. **Add more instances** in `docker-compose.prod.yml`:
   ```yaml
   backspace-agent:
     deploy:
       replicas: 3  # Run 3 instances
   ```

2. **Update Nginx configuration** in `nginx.conf`:
   ```nginx
   upstream backspace_backend {
       server backspace-agent:8000;
       server backspace-agent:8001;
       server backspace-agent:8002;
   }
   ```

### Resource Optimization
Monitor resource usage:
```bash
# Check container resource usage
docker stats

# Monitor specific service
docker stats backspace-agent
```

## 🎯 Production Checklist

Before going live, ensure:

- [ ] All API keys are configured
- [ ] GitHub token has required permissions
- [ ] Docker socket permissions are correct
- [ ] Health checks are passing
- [ ] Logs are being generated
- [ ] Telemetry is working
- [ ] Rate limiting is configured
- [ ] Backup strategy is in place

## 🚀 Next Steps

Once deployed, you can:

1. **Test the API**:
   ```bash
   curl -X POST http://localhost:8000/api/v1/code \
     -H "Content-Type: application/json" \
     -d '{"repo_url": "https://github.com/user/repo", "prompt": "Add a README"}'
   ```

2. **Monitor performance** via Jaeger UI

3. **Scale as needed** based on usage

4. **Set up CI/CD** for automated deployments

This deployment approach gives you a production-ready system that's both simple to understand and technically robust! 🎉 