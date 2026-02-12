import React from 'react';
import ConfidenceGauge from './ConfidenceGauge';
import GradCAMView from './GradCAMView';

function getPredictionClass(prediction) {
    const p = prediction.toLowerCase();
    if (p.includes('ischemi')) return 'ischemic';
    if (p.includes('hemorrh') || p.includes('bleed')) return 'hemorrhagic';
    return 'normal';
}

function getBarColor(cls) {
    const c = cls.toLowerCase();
    if (c.includes('normal') || c.includes('no stroke')) return 'fill-green';
    if (c.includes('ischemi')) return 'fill-orange';
    return 'fill-red';
}

export default function PredictionResult({ result, originalPreview }) {
    if (!result) return null;

    const {
        prediction,
        stroke_type,
        confidence,
        confidence_scores,
        risk_level,
        grad_cam_image,
    } = result;

    const predClass = getPredictionClass(prediction);
    const riskClass = risk_level?.toLowerCase() || 'low';

    return (
        <div className="result-section">
            <div className="glass-card">
                <div className="result-header">
                    <div className="result-header__label">AI Prediction</div>
                    <div className={`result-header__prediction ${predClass}`}>
                        {prediction}
                    </div>
                    {stroke_type && stroke_type !== 'None' && (
                        <div style={{ color: 'var(--text-secondary)', fontSize: '0.95rem', marginTop: 4 }}>
                            Stroke Type: <strong>{stroke_type}</strong>
                        </div>
                    )}
                    <div className={`risk-badge ${riskClass}`}>
                        {riskClass === 'high' && '⚠️ '}
                        {riskClass === 'medium' && '⚡ '}
                        {riskClass === 'low' && '✅ '}
                        {risk_level} Risk
                    </div>
                </div>

                {/* Confidence Gauge */}
                <ConfidenceGauge confidence={confidence} />

                {/* Per-class confidence bars */}
                {confidence_scores && (
                    <div className="confidence-bars">
                        {Object.entries(confidence_scores).map(([cls, score]) => (
                            <div className="confidence-bar" key={cls}>
                                <span className="confidence-bar__label">{cls}</span>
                                <div className="confidence-bar__track">
                                    <div
                                        className={`confidence-bar__fill ${getBarColor(cls)}`}
                                        style={{ width: `${(score * 100).toFixed(1)}%` }}
                                    />
                                </div>
                                <span className="confidence-bar__value">
                                    {(score * 100).toFixed(1)}%
                                </span>
                            </div>
                        ))}
                    </div>
                )}

                {/* Grad-CAM */}
                <GradCAMView gradCamImage={grad_cam_image} originalPreview={originalPreview} />
            </div>
        </div>
    );
}
