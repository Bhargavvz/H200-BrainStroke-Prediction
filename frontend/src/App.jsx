import React, { useState, useCallback, useRef } from 'react';
import Header from './components/Header';
import MRIUpload from './components/MRIUpload';
import ClinicalForm, { INITIAL_CLINICAL } from './components/ClinicalForm';
import PredictionResult from './components/PredictionResult';
import { predictStroke } from './api/predict';

export default function App() {
    const [mriFile, setMriFile] = useState(null);
    const [clinicalData, setClinicalData] = useState({ ...INITIAL_CLINICAL });
    const [result, setResult] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    const [mriPreview, setMriPreview] = useState(null);
    const resultRef = useRef(null);

    const handleFileChange = useCallback((file) => {
        setMriFile(file);
        setResult(null);
        setError(null);
        if (file) {
            const reader = new FileReader();
            reader.onload = (e) => setMriPreview(e.target.result);
            reader.readAsDataURL(file);
        } else {
            setMriPreview(null);
        }
    }, []);

    const handleSubmit = useCallback(async () => {
        if (!mriFile) {
            setError('Please upload a Brain MRI image first.');
            return;
        }

        setLoading(true);
        setError(null);
        setResult(null);

        try {
            const clinical = {
                age: parseFloat(clinicalData.age) || 50,
                gender: clinicalData.gender || 'Male',
                hypertension: parseInt(clinicalData.hypertension) || 0,
                heart_disease: parseInt(clinicalData.heart_disease) || 0,
                avg_glucose_level: parseFloat(clinicalData.avg_glucose_level) || 100,
                bmi: parseFloat(clinicalData.bmi) || 25,
                smoking_status: clinicalData.smoking_status || 'unknown',
                cholesterol: clinicalData.cholesterol || 'normal',
            };

            const response = await predictStroke(mriFile, clinical);
            setResult(response);

            // Scroll to result
            setTimeout(() => {
                resultRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' });
            }, 200);
        } catch (err) {
            console.error('Prediction error:', err);
            if (err.response?.data?.detail) {
                setError(err.response.data.detail);
            } else if (err.message.includes('Network Error')) {
                setError(
                    'Cannot connect to the backend server. Make sure it is running on http://localhost:8000'
                );
            } else {
                setError(`Prediction failed: ${err.message}`);
            }
        } finally {
            setLoading(false);
        }
    }, [mriFile, clinicalData]);

    return (
        <>
            <Header />

            <div className="main-grid">
                <div>
                    <MRIUpload file={mriFile} onFileChange={handleFileChange} />
                </div>
                <div>
                    <ClinicalForm clinicalData={clinicalData} onChange={setClinicalData} />
                </div>
            </div>

            <div className="submit-section">
                <button
                    className="btn-predict"
                    onClick={handleSubmit}
                    disabled={!mriFile || loading}
                >
                    {loading ? (
                        <>
                            <span className="spinner" />
                            Analyzing MRI...
                        </>
                    ) : (
                        <>
                            🧠 Predict Stroke Risk
                        </>
                    )}
                </button>

                {error && <div className="error-message">❌ {error}</div>}
            </div>

            <div ref={resultRef}>
                <PredictionResult result={result} originalPreview={mriPreview} />
            </div>

            <footer className="footer">
                <p>
                    Brain Stroke Prediction System · Hybrid AI (EfficientNet-B4 + Clinical DNN)
                    <br />
                    ⚠️ This is a research tool — not intended for clinical diagnosis.
                    Always consult a qualified medical professional.
                </p>
            </footer>
        </>
    );
}
