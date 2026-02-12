import React from 'react';

export default function Header() {
    return (
        <header className="header">
            <div className="header__badge">
                <span>🔬</span>
                <span>Hybrid AI · EfficientNet-B4 + Clinical DNN</span>
            </div>
            <h1 className="header__title">Brain Stroke Prediction</h1>
            <p className="header__subtitle">
                Upload a Brain MRI scan and enter clinical data to receive an
                AI-powered stroke risk assessment with explainability heatmaps.
            </p>
        </header>
    );
}
