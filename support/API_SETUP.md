# API Setup Instructions

## Groq API Configuration

This project uses Groq API for AI-powered medical analysis. To enable this feature:

### 1. Get Groq API Key
- Visit [Groq Console](https://console.groq.com/)
- Sign up for a free account
- Generate an API key

### 2. Configure API Key
Replace the placeholder in `main.py`:

```python
# Line 15 in main.py
GROQ_API_KEY = "your_groq_api_key_here"  # Replace with your actual key
```

### 3. Alternative: Environment Variable
For better security, use environment variable:

```bash
# Set environment variable
export GROQ_API_KEY="your_actual_api_key"
```

Then modify `main.py`:
```python
import os
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "your_groq_api_key_here")
```

### 4. Without API Key
The application will work without Groq API, but AI analysis will show:
"AI explanation unavailable: [error message]"

All other features (segmentation, visualization, measurements) work normally.