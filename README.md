# 🌫️ Air Quality Index (AQI) Prediction – Delhi 2019

Dieses Projekt analysiert Luftqualitätsdaten aus **Delhi (2019)** und baut ein Machine-Learning-Modell zur Vorhersage der **AQI-Kategorie** ("Good", "Moderate", "Poor", "Very Poor", "Severe").

## 📊 Datenquelle
- Kaggle-Dataset: [Air Quality Data in India (2015–2020) – by Rohan Rao](https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india)  
- Verwendeter Ausschnitt: **Delhi, Jahr 2019 (stündliche Daten, 8.760 Zeilen)**  
- Inspiration: [AmanXai – Air Quality Index Analysis using Python](https://amanxai.com/2023/09/18/air-quality-index-analysis-using-python/)

## 🧪 Vorgehen
1. **Explorative Datenanalyse (EDA)**  
   - Zeitreihen der Schadstoffe (PM2.5, PM10, NO2, SO2, CO, O3, NH3)  
   - AQI-Berechnung & Kategorisierung (EPA-Skala)  
   - Korrelationen & wöchentliche Muster  

2. **Modellierung**  
   - Zielvariable: `AQI_Bucket` (offizielle Kategorien im Dataset)  
   - Features: Schadstoffkonzentrationen  
   - Modelle:  
     - **Gradient Boosting Classifier** → Accuracy ~ **69 %**  
     - **Logistische Regression (Baseline)** → Accuracy ~ **55 %** (CrossVal)  

3. **Ergebnisse**  
   - Beste Performance bei Gradient Boosting  
   - Hauptfehler zwischen benachbarten Klassen (z. B. *Moderate ↔ Poor*)  
   - Modell gespeichert als `aqi_bucket_model.pkl`

## 🚀 Streamlit-App
Eine interaktive App zur Vorhersage der AQI-Kategorie aus Eingabewerten der Schadstoffe.  

👉 **[Hier direkt ausprobieren (Hugging Face Space)](https://huggingface.co/spaces/emr7y/Air_Quality_Data_India)**  

## Lokal starten:
```bash
streamlit run app.py
```

## 📂 Projektstruktur
```
📂 Air_Quality_Data_India
 ├─ app.py                # Streamlit App
 ├─ aqi_bucket_model.pkl  # gespeichertes Modell
 ├─ requirements.txt      # Abhängigkeiten
 ├─ notebook.ipynb        # Analyse + Modelltraining
 └─ README.md             # Projektbeschreibung
```

## ❤️ Credits
- **Daten:** [Rohan Rao (Kaggle)](https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india)  
- **Inspiration:** [AmanXai Blog](https://amanxai.com/2023/09/18/air-quality-index-analysis-using-python/)  
