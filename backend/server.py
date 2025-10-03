from fastapi import FastAPI, APIRouter, HTTPException, UploadFile, File
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import uuid
from datetime import datetime
import json
import re
import asyncio
from bs4 import BeautifulSoup
import lxml
import pyppeteer
from pyppeteer import launch
import openai

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

app = FastAPI()
api_router = APIRouter(prefix="/api")

# Web features baseline data - extensive list of modern web APIs and features
BASELINE_FEATURES = {
    # CSS Properties
    "display: grid": {"baseline": True, "support": "Widely Supported", "category": "CSS Grid"},
    "display: flex": {"baseline": True, "support": "Widely Supported", "category": "CSS Flexbox"},
    "gap": {"baseline": True, "support": "Widely Supported", "category": "CSS Layout"},
    "grid-template-columns": {"baseline": True, "support": "Widely Supported", "category": "CSS Grid"},
    "grid-template-rows": {"baseline": True, "support": "Widely Supported", "category": "CSS Grid"},
    "flex-direction": {"baseline": True, "support": "Widely Supported", "category": "CSS Flexbox"},
    "align-items": {"baseline": True, "support": "Widely Supported", "category": "CSS Alignment"},
    "justify-content": {"baseline": True, "support": "Widely Supported", "category": "CSS Alignment"},
    "transform": {"baseline": True, "support": "Widely Supported", "category": "CSS Transforms"},
    "transition": {"baseline": True, "support": "Widely Supported", "category": "CSS Animations"},
    "box-shadow": {"baseline": True, "support": "Widely Supported", "category": "CSS Styling"},
    "border-radius": {"baseline": True, "support": "Widely Supported", "category": "CSS Styling"},
    "background-clip": {"baseline": True, "support": "Widely Supported", "category": "CSS Background"},
    
    # CSS Properties - Non-baseline
    "subgrid": {"baseline": False, "support": "Limited Support", "category": "CSS Grid"},
    "container-queries": {"baseline": False, "support": "Limited Support", "category": "CSS Container Queries"},
    ":has()": {"baseline": False, "support": "Limited Support", "category": "CSS Selectors"},
    "accent-color": {"baseline": False, "support": "Limited Support", "category": "CSS Form Styling"},
    "color-mix()": {"baseline": False, "support": "Limited Support", "category": "CSS Color"},
    "view-transition-name": {"baseline": False, "support": "Experimental", "category": "CSS Transitions"},
    
    # JavaScript APIs - Baseline
    "fetch": {"baseline": True, "support": "Widely Supported", "category": "JavaScript Network"},
    "Promise": {"baseline": True, "support": "Widely Supported", "category": "JavaScript Async"},
    "async/await": {"baseline": True, "support": "Widely Supported", "category": "JavaScript Async"},
    "Array.prototype.includes": {"baseline": True, "support": "Widely Supported", "category": "JavaScript Array"},
    "Object.entries": {"baseline": True, "support": "Widely Supported", "category": "JavaScript Object"},
    "Map": {"baseline": True, "support": "Widely Supported", "category": "JavaScript Collections"},
    "Set": {"baseline": True, "support": "Widely Supported", "category": "JavaScript Collections"},
    "WeakMap": {"baseline": True, "support": "Widely Supported", "category": "JavaScript Collections"},
    "Symbol": {"baseline": True, "support": "Widely Supported", "category": "JavaScript Primitives"},
    "Proxy": {"baseline": True, "support": "Widely Supported", "category": "JavaScript Meta"},
    "Reflect": {"baseline": True, "support": "Widely Supported", "category": "JavaScript Meta"},
    "classList": {"baseline": True, "support": "Widely Supported", "category": "JavaScript DOM"},
    "querySelector": {"baseline": True, "support": "Widely Supported", "category": "JavaScript DOM"},
    "addEventListener": {"baseline": True, "support": "Widely Supported", "category": "JavaScript Events"},
    "IntersectionObserver": {"baseline": True, "support": "Widely Supported", "category": "JavaScript Observer APIs"},
    "MutationObserver": {"baseline": True, "support": "Widely Supported", "category": "JavaScript Observer APIs"},
    "ResizeObserver": {"baseline": True, "support": "Widely Supported", "category": "JavaScript Observer APIs"},
    
    # JavaScript APIs - Non-baseline
    "SharedArrayBuffer": {"baseline": False, "support": "Limited Support", "category": "JavaScript Threading"},
    "Atomics": {"baseline": False, "support": "Limited Support", "category": "JavaScript Threading"},
    "OffscreenCanvas": {"baseline": False, "support": "Limited Support", "category": "JavaScript Canvas"},
    "requestIdleCallback": {"baseline": False, "support": "Limited Support", "category": "JavaScript Performance"},
    "Performance.measureUserAgentSpecificMemory": {"baseline": False, "support": "Experimental", "category": "JavaScript Performance"},
    "CSS.paintWorklet": {"baseline": False, "support": "Limited Support", "category": "JavaScript Worklets"},
    "AudioWorklet": {"baseline": False, "support": "Limited Support", "category": "JavaScript Audio"},
    "File System Access API": {"baseline": False, "support": "Limited Support", "category": "JavaScript File System"},
    "Web Locks API": {"baseline": False, "support": "Limited Support", "category": "JavaScript Concurrency"},
    "BroadcastChannel": {"baseline": False, "support": "Limited Support", "category": "JavaScript Communication"},
    
    # HTML Elements - Baseline
    "<dialog>": {"baseline": True, "support": "Widely Supported", "category": "HTML Interactive"},
    "<details>": {"baseline": True, "support": "Widely Supported", "category": "HTML Interactive"},
    "<summary>": {"baseline": True, "support": "Widely Supported", "category": "HTML Interactive"},
    "<picture>": {"baseline": True, "support": "Widely Supported", "category": "HTML Media"},
    "<source>": {"baseline": True, "support": "Widely Supported", "category": "HTML Media"},
    "<video>": {"baseline": True, "support": "Widely Supported", "category": "HTML Media"},
    "<audio>": {"baseline": True, "support": "Widely Supported", "category": "HTML Media"},
    "<canvas>": {"baseline": True, "support": "Widely Supported", "category": "HTML Graphics"},
    
    # HTML Elements - Non-baseline
    "<search>": {"baseline": False, "support": "Limited Support", "category": "HTML Form"},
    "<selectlist>": {"baseline": False, "support": "Experimental", "category": "HTML Form"},
    "<portal>": {"baseline": False, "support": "Experimental", "category": "HTML Navigation"},
    
    # Web APIs - Baseline
    "Geolocation API": {"baseline": True, "support": "Widely Supported", "category": "Web APIs"},
    "WebStorage (localStorage)": {"baseline": True, "support": "Widely Supported", "category": "Web APIs"},
    "IndexedDB": {"baseline": True, "support": "Widely Supported", "category": "Web APIs"},
    "History API": {"baseline": True, "support": "Widely Supported", "category": "Web APIs"},
    "File API": {"baseline": True, "support": "Widely Supported", "category": "Web APIs"},
    "Drag and Drop API": {"baseline": True, "support": "Widely Supported", "category": "Web APIs"},
    "Fullscreen API": {"baseline": True, "support": "Widely Supported", "category": "Web APIs"},
    "Notification API": {"baseline": True, "support": "Widely Supported", "category": "Web APIs"},
    "Vibration API": {"baseline": False, "support": "Limited Support", "category": "Web APIs"},
    
    # Web APIs - Non-baseline
    "WebXR Device API": {"baseline": False, "support": "Limited Support", "category": "Web APIs"},
    "WebCodecs API": {"baseline": False, "support": "Limited Support", "category": "Web APIs"},
    "WebGPU": {"baseline": False, "support": "Limited Support", "category": "Web APIs"},
    "WebAssembly SIMD": {"baseline": False, "support": "Limited Support", "category": "Web APIs"},
    "Origin Private File System API": {"baseline": False, "support": "Limited Support", "category": "Web APIs"},
    "Trusted Types": {"baseline": False, "support": "Limited Support", "category": "Web APIs"},
    "Web Streams API": {"baseline": False, "support": "Limited Support", "category": "Web APIs"},
}

# Models
class BaselineFeature(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    baseline: bool
    support: str
    category: str
    description: Optional[str] = None

class AnalysisResult(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    source_type: str  # 'code', 'file', 'url'
    source_content: str
    features_found: List[BaselineFeature]
    compliance_score: float
    badge_level: str  # Bronze, Silver, Gold
    ai_suggestions: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)

class CodeAnalysisRequest(BaseModel):
    code: str
    analysis_type: str = "full"  # full, css, js, html

class URLAnalysisRequest(BaseModel):
    url: str
    scan_method: str = "simple"  # simple (cheerio) or advanced (puppeteer)

# Analysis Functions
def extract_css_features(content: str) -> List[BaselineFeature]:
    """Extract CSS features from content"""
    features = []
    css_patterns = {
        r'display\s*:\s*grid': 'display: grid',
        r'display\s*:\s*flex': 'display: flex',
        r'gap\s*:': 'gap',
        r'grid-template-columns\s*:': 'grid-template-columns',
        r'grid-template-rows\s*:': 'grid-template-rows',
        r'flex-direction\s*:': 'flex-direction',
        r'align-items\s*:': 'align-items',
        r'justify-content\s*:': 'justify-content',
        r'transform\s*:': 'transform',
        r'transition\s*:': 'transition',
        r'box-shadow\s*:': 'box-shadow',
        r'border-radius\s*:': 'border-radius',
        r'background-clip\s*:': 'background-clip',
        r'subgrid': 'subgrid',
        r'@container': 'container-queries',
        r':has\s*\(': ':has()',
        r'accent-color\s*:': 'accent-color',
        r'color-mix\s*\(': 'color-mix()',
        r'view-transition-name\s*:': 'view-transition-name',
    }
    
    for pattern, feature_name in css_patterns.items():
        if re.search(pattern, content, re.IGNORECASE):
            feature_data = BASELINE_FEATURES.get(feature_name, {
                "baseline": False, 
                "support": "Unknown", 
                "category": "CSS Unknown"
            })
            features.append(BaselineFeature(
                name=feature_name,
                baseline=feature_data["baseline"],
                support=feature_data["support"],
                category=feature_data["category"]
            ))
    
    return features

def extract_js_features(content: str) -> List[BaselineFeature]:
    """Extract JavaScript features from content"""
    features = []
    js_patterns = {
        r'fetch\s*\(': 'fetch',
        r'\bnew\s+Promise\b': 'Promise',
        r'\basync\s+function\b|\bawait\b': 'async/await',
        r'\.includes\s*\(': 'Array.prototype.includes',
        r'Object\.entries\s*\(': 'Object.entries',
        r'\bnew\s+Map\b': 'Map',
        r'\bnew\s+Set\b': 'Set',
        r'\bnew\s+WeakMap\b': 'WeakMap',
        r'Symbol\s*\(': 'Symbol',
        r'\bnew\s+Proxy\b': 'Proxy',
        r'Reflect\.': 'Reflect',
        r'\.classList\b': 'classList',
        r'querySelector\s*\(': 'querySelector',
        r'addEventListener\s*\(': 'addEventListener',
        r'\bnew\s+IntersectionObserver\b': 'IntersectionObserver',
        r'\bnew\s+MutationObserver\b': 'MutationObserver',
        r'\bnew\s+ResizeObserver\b': 'ResizeObserver',
        r'SharedArrayBuffer': 'SharedArrayBuffer',
        r'Atomics\.': 'Atomics',
        r'OffscreenCanvas': 'OffscreenCanvas',
        r'requestIdleCallback\s*\(': 'requestIdleCallback',
        r'CSS\.paintWorklet': 'CSS.paintWorklet',
        r'AudioWorklet': 'AudioWorklet',
        r'showOpenFilePicker|showSaveFilePicker|showDirectoryPicker': 'File System Access API',
        r'navigator\.locks': 'Web Locks API',
        r'\bnew\s+BroadcastChannel\b': 'BroadcastChannel',
    }
    
    for pattern, feature_name in js_patterns.items():
        if re.search(pattern, content, re.IGNORECASE):
            feature_data = BASELINE_FEATURES.get(feature_name, {
                "baseline": False, 
                "support": "Unknown", 
                "category": "JavaScript Unknown"
            })
            features.append(BaselineFeature(
                name=feature_name,
                baseline=feature_data["baseline"],
                support=feature_data["support"],
                category=feature_data["category"]
            ))
    
    return features

def extract_html_features(content: str) -> List[BaselineFeature]:
    """Extract HTML features from content"""
    features = []
    html_patterns = {
        r'<dialog\b': '<dialog>',
        r'<details\b': '<details>',
        r'<summary\b': '<summary>',
        r'<picture\b': '<picture>',
        r'<source\b': '<source>',
        r'<video\b': '<video>',
        r'<audio\b': '<audio>',
        r'<canvas\b': '<canvas>',
        r'<search\b': '<search>',
        r'<selectlist\b': '<selectlist>',
        r'<portal\b': '<portal>',
    }
    
    for pattern, feature_name in html_patterns.items():
        if re.search(pattern, content, re.IGNORECASE):
            feature_data = BASELINE_FEATURES.get(feature_name, {
                "baseline": False, 
                "support": "Unknown", 
                "category": "HTML Unknown"
            })
            features.append(BaselineFeature(
                name=feature_name,
                baseline=feature_data["baseline"],
                support=feature_data["support"],
                category=feature_data["category"]
            ))
    
    return features

def extract_webapi_features(content: str) -> List[BaselineFeature]:
    """Extract Web API features from content"""
    features = []
    api_patterns = {
        r'navigator\.geolocation': 'Geolocation API',
        r'localStorage|sessionStorage': 'WebStorage (localStorage)',
        r'indexedDB|IDBRequest': 'IndexedDB',
        r'history\.(pushState|replaceState)': 'History API',
        r'FileReader|File\b': 'File API',
        r'ondragstart|ondrop|DataTransfer': 'Drag and Drop API',
        r'requestFullscreen|exitFullscreen': 'Fullscreen API',
        r'\bnew\s+Notification\b': 'Notification API',
        r'navigator\.vibrate': 'Vibration API',
        r'WebXR|XRSession': 'WebXR Device API',
        r'VideoEncoder|VideoDecoder|AudioEncoder|AudioDecoder': 'WebCodecs API',
        r'navigator\.gpu': 'WebGPU',
        r'WebAssembly\.instantiate.*simd': 'WebAssembly SIMD',
        r'navigator\.storage\.getDirectory': 'Origin Private File System API',
        r'trustedTypes': 'Trusted Types',
        r'ReadableStream|WritableStream': 'Web Streams API',
    }
    
    for pattern, feature_name in api_patterns.items():
        if re.search(pattern, content, re.IGNORECASE):
            feature_data = BASELINE_FEATURES.get(feature_name, {
                "baseline": False, 
                "support": "Unknown", 
                "category": "Web APIs Unknown"
            })
            features.append(BaselineFeature(
                name=feature_name,
                baseline=feature_data["baseline"],
                support=feature_data["support"],
                category=feature_data["category"]
            ))
    
    return features

def calculate_compliance_score(features: List[BaselineFeature]) -> tuple[float, str]:
    """Calculate compliance score and badge level"""
    if not features:
        return 100.0, "Gold"
    
    baseline_count = sum(1 for f in features if f.baseline)
    total_count = len(features)
    score = (baseline_count / total_count) * 100
    
    if score >= 90:
        badge = "Gold"
    elif score >= 70:
        badge = "Silver"
    else:
        badge = "Bronze"
    
    return score, badge

async def get_ai_suggestions(non_baseline_features: List[BaselineFeature]) -> str:
    """Get AI suggestions for non-baseline features using OpenAI"""
    if not non_baseline_features:
        return "All features used are baseline compatible! Great job!"
    
    try:
        # Initialize OpenAI client
        client = openai.AsyncOpenAI(api_key=os.environ.get('OPENAI_API_KEY'))
        
        features_text = "\n".join([f"- {f.name} ({f.category})" for f in non_baseline_features])
        
        response = await client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "You are a web development expert specializing in browser compatibility and baseline web features. Provide concise, actionable suggestions for developers."
                },
                {
                    "role": "user", 
                    "content": f"""The following non-baseline web features were detected in the code:

{features_text}

For each feature, provide:
1. Why it's not baseline (browser support issues)
2. Suggested fallback or alternative approach
3. Polyfill recommendations if available

Keep suggestions concise and practical for developers."""
                }
            ]
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        logging.error(f"AI suggestion error: {e}")
        return f"Unable to generate AI suggestions at this time. Non-baseline features detected: {', '.join([f.name for f in non_baseline_features])}"

async def analyze_url_simple(url: str) -> str:
    """Simple URL analysis using requests and BeautifulSoup"""
    try:
        import requests
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.text
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error fetching URL: {str(e)}")

async def analyze_url_advanced(url: str) -> str:
    """Advanced URL analysis using Puppeteer"""
    try:
        browser = await launch(headless=True, args=['--no-sandbox', '--disable-dev-shm-usage'])
        page = await browser.newPage()
        await page.goto(url, {'waitUntil': 'networkidle2', 'timeout': 30000})
        content = await page.content()
        await browser.close()
        return content
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error analyzing URL with Puppeteer: {str(e)}")

# API Routes
@api_router.get("/")
async def root():
    return {"message": "Baseline Compatibility Web Platform API"}

@api_router.get("/features", response_model=List[BaselineFeature])
async def get_all_features():
    """Get all supported baseline features"""
    features = []
    for name, data in BASELINE_FEATURES.items():
        features.append(BaselineFeature(
            name=name,
            baseline=data["baseline"],
            support=data["support"],
            category=data["category"]
        ))
    return features

@api_router.post("/analyze/code", response_model=AnalysisResult, status_code=201)
async def analyze_code(request: CodeAnalysisRequest):
    """Analyze code snippet for baseline compatibility"""
    # Validate input
    if not request.code.strip():
        raise HTTPException(status_code=422, detail="Code input cannot be empty")
    
    features = []
    
    # Extract different types of features
    if request.analysis_type in ["full", "css"]:
        features.extend(extract_css_features(request.code))
    if request.analysis_type in ["full", "js", "javascript"]:
        features.extend(extract_js_features(request.code))
        features.extend(extract_webapi_features(request.code))
    if request.analysis_type in ["full", "html"]:
        features.extend(extract_html_features(request.code))
    
    # Remove duplicates
    unique_features = []
    seen_names = set()
    for feature in features:
        if feature.name not in seen_names:
            unique_features.append(feature)
            seen_names.add(feature.name)
    
    # Calculate compliance
    score, badge = calculate_compliance_score(unique_features)
    
    # Get AI suggestions for non-baseline features
    non_baseline_features = [f for f in unique_features if not f.baseline]
    ai_suggestions = await get_ai_suggestions(non_baseline_features)
    
    # Create analysis result
    result = AnalysisResult(
        source_type="code",
        source_content=request.code[:500] + "..." if len(request.code) > 500 else request.code,
        features_found=unique_features,
        compliance_score=score,
        badge_level=badge,
        ai_suggestions=ai_suggestions
    )
    
    # Save to database
    await db.analysis_results.insert_one(result.dict())
    
    return result

@api_router.post("/analyze/url", response_model=AnalysisResult, status_code=201)
async def analyze_url(request: URLAnalysisRequest):
    """Analyze website URL for baseline compatibility"""
    # Fetch content based on method
    if request.scan_method == "advanced":
        content = await analyze_url_advanced(request.url)
    else:
        content = await analyze_url_simple(request.url)
    
    # Extract features from HTML, CSS, and JS
    features = []
    features.extend(extract_html_features(content))
    features.extend(extract_css_features(content))
    features.extend(extract_js_features(content))
    features.extend(extract_webapi_features(content))
    
    # Remove duplicates
    unique_features = []
    seen_names = set()
    for feature in features:
        if feature.name not in seen_names:
            unique_features.append(feature)
            seen_names.add(feature.name)
    
    # Calculate compliance
    score, badge = calculate_compliance_score(unique_features)
    
    # Get AI suggestions for non-baseline features
    non_baseline_features = [f for f in unique_features if not f.baseline]
    ai_suggestions = await get_ai_suggestions(non_baseline_features)
    
    # Create analysis result
    result = AnalysisResult(
        source_type="url",
        source_content=request.url,
        features_found=unique_features,
        compliance_score=score,
        badge_level=badge,
        ai_suggestions=ai_suggestions
    )
    
    # Save to database
    await db.analysis_results.insert_one(result.dict())
    
    return result

@api_router.post("/analyze/file", response_model=AnalysisResult, status_code=201)
async def analyze_file(file: UploadFile = File(...)):
    """Analyze uploaded file for baseline compatibility"""
    # Read file content
    content = await file.read()
    
    # Try to decode as text
    try:
        text_content = content.decode('utf-8')
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="File must be a text file (HTML, CSS, JS)")
    
    # Extract features based on file extension
    features = []
    file_ext = file.filename.lower().split('.')[-1] if file.filename else ""
    
    if file_ext in ['html', 'htm']:
        features.extend(extract_html_features(text_content))
        features.extend(extract_css_features(text_content))
        features.extend(extract_js_features(text_content))
        features.extend(extract_webapi_features(text_content))
    elif file_ext in ['css']:
        features.extend(extract_css_features(text_content))
    elif file_ext in ['js', 'jsx', 'ts', 'tsx']:
        features.extend(extract_js_features(text_content))
        features.extend(extract_webapi_features(text_content))
    else:
        # Analyze all types for unknown extensions
        features.extend(extract_html_features(text_content))
        features.extend(extract_css_features(text_content))
        features.extend(extract_js_features(text_content))
        features.extend(extract_webapi_features(text_content))
    
    # Remove duplicates
    unique_features = []
    seen_names = set()
    for feature in features:
        if feature.name not in seen_names:
            unique_features.append(feature)
            seen_names.add(feature.name)
    
    # Calculate compliance
    score, badge = calculate_compliance_score(unique_features)
    
    # Get AI suggestions for non-baseline features
    non_baseline_features = [f for f in unique_features if not f.baseline]
    ai_suggestions = await get_ai_suggestions(non_baseline_features)
    
    # Create analysis result
    result = AnalysisResult(
        source_type="file",
        source_content=f"{file.filename} ({len(text_content)} characters)",
        features_found=unique_features,
        compliance_score=score,
        badge_level=badge,
        ai_suggestions=ai_suggestions
    )
    
    # Save to database
    await db.analysis_results.insert_one(result.dict())
    
    return result

@api_router.get("/reports", response_model=List[AnalysisResult])
async def get_reports(limit: int = 50):
    """Get recent analysis reports"""
    reports = await db.analysis_results.find().sort("created_at", -1).limit(limit).to_list(length=None)
    return [AnalysisResult(**report) for report in reports]

@api_router.get("/reports/{report_id}", response_model=AnalysisResult)
async def get_report(report_id: str):
    """Get specific analysis report"""
    report = await db.analysis_results.find_one({"id": report_id})
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")
    return AnalysisResult(**report)

@api_router.delete("/reports/{report_id}")
async def delete_report(report_id: str):
    """Delete analysis report"""
    result = await db.analysis_results.delete_one({"id": report_id})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Report not found")
    return {"message": "Report deleted successfully"}

# Include the router in the main app
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)