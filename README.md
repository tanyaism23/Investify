# EcoPolis - AI-Driven Urban Biodiversity Planner

An interactive web application that helps urban planners make biodiversity-conscious development decisions using machine learning, GIS data analysis, and sustainability recommendations.

## 🌿 Project Overview

EcoPolis is a tool that addresses the challenge of preserving urban biodiversity in the face of expanding urbanization. It combines GIS data, machine learning, and interactive visualization to help urban planners:
- Identify biodiversity hotspots
- Predict the impact of urban development on local ecosystems
- Generate actionable recommendations for sustainable urban planning

## 🚀 Features

### GIS-based Biodiversity Visualization
- Interactive mapping of urban areas, vegetation, and biodiversity hotspots
- Integration with QGIS and Geopandas for spatial data handling
- Zoom and click functionality for detailed biodiversity metrics

### Impact Prediction Model
- Machine learning model to predict development impact on local biodiversity
- Considers factors like climate data, land use changes, and vegetation cover
- Provides biodiversity risk scores for different development scenarios

### Sustainability Recommendation System
- Suggests optimal green corridors for wildlife movement
- Identifies areas for native vegetation preservation
- Recommends biodiversity-friendly infrastructure modifications

### User Interface
- Web-based interface for data upload and visualization
- Support for GIS files and climate data input
- Interactive scenario planning tools

## 🛠️ Technical Stack

- **Frontend**: React + Vite
- **Backend**: Python
- **ML Framework**: Jupyter notebooks for model development
- **Data Processing**: Geopandas, QGIS
- **Deployment**: Heroku/GitHub Pages compatible

## 📁 Project Structure

```
├── ML/                     # Machine learning models and notebooks
├── src/                    # Source code
├── .env                    # Environment variables
├── .gitignore             # Git ignore rules
├── README.md              # Project documentation
├── eslint.config.js       # ESLint configuration
├── index.html             # Main HTML file
├── models2.pkl            # Trained ML models
├── package.json           # Project dependencies
├── package-lock.json      # Dependency lock file
├── postcss.config.js      # PostCSS configuration
├── tailwind.config.js     # Tailwind CSS configuration
└── vite.config.js         # Vite configuration
```

## 🔍 Machine Learning Components

The `ML` directory contains:
- `All.ipynb`: Complete model training pipeline
- `Inference.ipynb`: Model inference and testing
- `LLM-baseline.ipynb`: Baseline model implementation
- `Reasoning_prompt.ipynb`: Prompt engineering for recommendations
- `ensemble_weights2.pkl`: Trained ensemble model weights
- `preprocessors2.pkl`: Data preprocessing pipelines

## 🚀 Getting Started

1. Clone the repository:
```bash
git clone https://github.com/your-username/ecopolis.git
cd ecopolis
```

2. Install dependencies:
```bash
npm install          # Frontend dependencies
pip install -r requirements.txt  # Python dependencies
```

3. Set up environment variables:
- Create a `.env` file based on `.env.example`
- Add your API keys for weather and biodiversity data services

4. Run the development server:
```bash
npm run dev
```

## 📊 Data Sources

- Urban and Biodiversity GIS Data: OpenStreetMap, USGS Earth Explorer
- Climate Data: NOAA, WorldClim
- Land Use Data: Various classification datasets

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- OpenStreetMap contributors for GIS data
- NOAA and WorldClim for climate data
- The open-source community for various tools and libraries used in this project
