import React, { useState } from 'react';

const INITIAL = {
    age: '',
    gender: 'Male',
    hypertension: '0',
    heart_disease: '0',
    avg_glucose_level: '',
    bmi: '',
    smoking_status: 'never_smoked',
    cholesterol: 'normal',
};

export default function ClinicalForm({ clinicalData, onChange }) {
    const [expanded, setExpanded] = useState(true);

    const handleChange = (field, value) => {
        onChange({ ...clinicalData, [field]: value });
    };

    return (
        <div className="glass-card">
            <div className="card-title">📋 Clinical Information</div>
            <div className="card-description">
                Optional — improves prediction accuracy with patient metadata
            </div>

            <button
                className="collapsible-toggle"
                onClick={() => setExpanded(!expanded)}
            >
                <span className={`chevron${expanded ? ' open' : ''}`}>▶</span>
                {expanded ? 'Hide clinical form' : 'Show clinical form'}
            </button>

            {expanded && (
                <div className="clinical-form">
                    {/* Age */}
                    <div className="form-group">
                        <label className="form-label">Age (years)</label>
                        <input
                            className="form-input"
                            type="number"
                            min="0" max="120"
                            placeholder="e.g. 65"
                            value={clinicalData.age}
                            onChange={(e) => handleChange('age', e.target.value)}
                        />
                    </div>

                    {/* Gender */}
                    <div className="form-group">
                        <label className="form-label">Gender</label>
                        <select
                            className="form-select"
                            value={clinicalData.gender}
                            onChange={(e) => handleChange('gender', e.target.value)}
                        >
                            <option value="Male">Male</option>
                            <option value="Female">Female</option>
                            <option value="Other">Other</option>
                        </select>
                    </div>

                    {/* Hypertension */}
                    <div className="form-group">
                        <label className="form-label">Hypertension</label>
                        <select
                            className="form-select"
                            value={clinicalData.hypertension}
                            onChange={(e) => handleChange('hypertension', e.target.value)}
                        >
                            <option value="0">No</option>
                            <option value="1">Yes</option>
                        </select>
                    </div>

                    {/* Heart Disease */}
                    <div className="form-group">
                        <label className="form-label">Heart Disease</label>
                        <select
                            className="form-select"
                            value={clinicalData.heart_disease}
                            onChange={(e) => handleChange('heart_disease', e.target.value)}
                        >
                            <option value="0">No</option>
                            <option value="1">Yes</option>
                        </select>
                    </div>

                    {/* Glucose */}
                    <div className="form-group">
                        <label className="form-label">Avg Glucose (mg/dL)</label>
                        <input
                            className="form-input"
                            type="number"
                            min="0" max="500"
                            step="0.01"
                            placeholder="e.g. 106.14"
                            value={clinicalData.avg_glucose_level}
                            onChange={(e) => handleChange('avg_glucose_level', e.target.value)}
                        />
                    </div>

                    {/* BMI */}
                    <div className="form-group">
                        <label className="form-label">BMI</label>
                        <input
                            className="form-input"
                            type="number"
                            min="10" max="80"
                            step="0.1"
                            placeholder="e.g. 28.5"
                            value={clinicalData.bmi}
                            onChange={(e) => handleChange('bmi', e.target.value)}
                        />
                    </div>

                    {/* Smoking */}
                    <div className="form-group">
                        <label className="form-label">Smoking Status</label>
                        <select
                            className="form-select"
                            value={clinicalData.smoking_status}
                            onChange={(e) => handleChange('smoking_status', e.target.value)}
                        >
                            <option value="never_smoked">Never Smoked</option>
                            <option value="formerly_smoked">Formerly Smoked</option>
                            <option value="smokes">Currently Smokes</option>
                            <option value="unknown">Unknown</option>
                        </select>
                    </div>

                    {/* Cholesterol */}
                    <div className="form-group">
                        <label className="form-label">Cholesterol</label>
                        <select
                            className="form-select"
                            value={clinicalData.cholesterol}
                            onChange={(e) => handleChange('cholesterol', e.target.value)}
                        >
                            <option value="normal">Normal</option>
                            <option value="high">High</option>
                            <option value="low">Low</option>
                        </select>
                    </div>
                </div>
            )}
        </div>
    );
}

export { INITIAL as INITIAL_CLINICAL };
