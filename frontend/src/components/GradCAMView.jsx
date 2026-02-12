import React, { useState } from 'react';

export default function GradCAMView({ gradCamImage, originalPreview }) {
    const [opacity, setOpacity] = useState(50);

    if (!gradCamImage) return null;

    return (
        <div className="gradcam-section">
            <div className="card-title" style={{ marginBottom: 16 }}>
                🔍 Grad-CAM Explainability
            </div>
            <div className="card-description">
                Regions highlighted in warm colors indicate areas the model focused on for its decision.
            </div>

            <div className="gradcam-grid">
                {originalPreview && (
                    <div className="gradcam-panel">
                        <div className="gradcam-panel__title">Original MRI</div>
                        <img src={originalPreview} alt="Original MRI" />
                    </div>
                )}
                <div className="gradcam-panel">
                    <div className="gradcam-panel__title">Grad-CAM Heatmap</div>
                    <img
                        src={`data:image/png;base64,${gradCamImage}`}
                        alt="Grad-CAM Heatmap"
                        style={{ opacity: opacity / 100 }}
                    />
                </div>
            </div>

            <div className="opacity-slider">
                <label>Heatmap Opacity</label>
                <input
                    type="range"
                    min="10"
                    max="100"
                    value={opacity}
                    onChange={(e) => setOpacity(Number(e.target.value))}
                />
                <span style={{ color: 'var(--text-muted)', fontSize: '0.82rem', minWidth: 36 }}>
                    {opacity}%
                </span>
            </div>
        </div>
    );
}
