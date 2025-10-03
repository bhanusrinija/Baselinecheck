# Baseline Compatibility Web Platform

A web platform to analyze and ensure baseline compatibility of modern web features across browsers and environments. Includes a **FastAPI backend** and a **React frontend** for checking code, files, and websites against baseline web standards.  


---

## Features

- **Code Analysis**: Check HTML, CSS, and JavaScript snippets.  
- **File Upload Analysis**: Analyze web files (HTML, CSS, JS).  
- **URL Analysis**: Scan websites and web pages for modern feature usage.  
- **AI-Powered Suggestions**: Intelligent recommendations for non-baseline features.  
- **Compliance Scoring**: Bronze, Silver, or Gold badges based on compliance.  
- **Comprehensive Feature Database**: CSS properties, JS APIs, HTML elements, Web APIs.  
- **MongoDB Integration**: Store and retrieve analysis results.  

---

## Tech Stack

**Backend:** FastAPI, MongoDB, Pydantic, BeautifulSoup4, lxml, Pyppeteer, OpenAI API  
**Frontend:** React, Vite, Tailwind CSS  

---

## Usage

- Input code snippets for analysis  
- Upload files for compatibility checking  
- Enter URLs for website scanning  
- View analysis results with compliance scores and AI suggestions  

**API Endpoints (for reference):**

- `GET /api/features` – Get all baseline features  
- `POST /api/analyze/code` – Analyze code snippets  
- `POST /api/analyze/url` – Scan website URL  
- `POST /api/analyze/file` – Analyze uploaded file  
- `GET/DELETE /api/reports` – Manage reports  

---

## Project Structure
baseline updated/
├── backend/
│ ├── server.py
│ ├── requirements.txt
│ └── .env
├── frontend/
│ ├── src/
│ │ ├── App.jsx
│ │ ├── index.jsx
│ │ └── components/
│ ├── package.json
│ ├── vite.config.js
│ └── tailwind.config.js
└── README.md

---

## Baseline Features Database

- **CSS Properties:** Grid, Flexbox, Transforms, Animations  
- **JavaScript APIs:** Promises, Fetch, Observers, Collections  
- **HTML Elements:** Dialog, Details, Canvas, Video  
- **Web APIs:** Geolocation, WebStorage, IndexedDB, Notifications  

Each feature is classified as **baseline** (widely supported) or **non-baseline**. 
