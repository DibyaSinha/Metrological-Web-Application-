# 🌤️ Meteorological Data WebApp

A full-stack web application built during my 30-day internship at **DRDO – Integrated Test Range (ITR), Chandipur**.  
This tool enables real-time analysis and visualization of atmospheric data based on altitude steps — tailored for defense research, high-altitude flight planning, and environmental profiling.

---

## 🔍 Overview

The **Meteorological Data WebApp** allows users to:

- Upload one or more CSV datasets containing atmospheric readings
- Enter an **altitude step value** (e.g., 200 meters)
- Automatically extract and display the closest real data for each step
- View **interactive graphs** and a **Wind Rose Plot**
- Download results as CSV or a ZIP file (including graphs as PDFs)

> This project was developed as part of a research-oriented internship under the mentorship of **Shri Debanshu Ball, Scientist F (DRDO ITR)**.

---

## ✨ Features

- 📂 Multi-file CSV upload with data validation
- 🧮 Altitude-step-based data extraction (e.g., 0m, 200m, 400m…)
- 📈 Dynamic Chart.js plots:
  - Temperature (K)
  - Pressure (hPa)
  - Humidity (%)
  - Wind Speed (knots)
  - Input vs Matched Altitude
- 🌪️ **Wind Rose Plot** for wind direction frequency (Matplotlib + Polar projection)
- 💾 Download processed results as:
  - ✅ CSV file
  - 📦 ZIP (includes results + PDF plots for each file)
- 🌗 Dark Mode toggle with local storage support
- 🖼️ DRDO-styled watermark, translucent tables, and responsive UI

---

## 🧠 Technologies Used

| Stack | Tools |
|-------|-------|
| **Frontend** | HTML5, CSS3, Bootstrap 5, JavaScript, Chart.js |
| **Backend** | Python (Flask), Pandas, NumPy, Matplotlib |
| **Session & File Handling** | Flask-Session, ZipFile, PDFPages |
| **Deployment** | Waitress Server (for local hosting), Browser auto-launch |
| **Visualization** | Matplotlib, Chart.js, Base64 Image Embedding |

---

## 📁 Project Structure
📦 meteorological-webapp/
├── static/ # CSS, logo, and assets
│ └── style.css
├── templates/
│ └── index.html # Main UI layout with Jinja2 templating
├── app.py # Flask server logic
└── README.md # Project documentation

