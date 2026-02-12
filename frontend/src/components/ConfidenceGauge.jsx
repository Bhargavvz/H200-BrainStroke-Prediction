import React, { useMemo } from 'react';

export default function ConfidenceGauge({ confidence }) {
    const pct = Math.round(confidence * 100);
    const radius = 62;
    const circumference = 2 * Math.PI * radius;
    const offset = circumference - (confidence * circumference);

    const color = useMemo(() => {
        if (pct >= 80) return '#22c55e';
        if (pct >= 50) return '#f59e0b';
        return '#ef4444';
    }, [pct]);

    return (
        <div className="gauge-container">
            <svg className="gauge-svg" viewBox="0 0 160 160">
                <circle className="gauge-bg" cx="80" cy="80" r={radius} />
                <circle
                    className="gauge-fill"
                    cx="80"
                    cy="80"
                    r={radius}
                    stroke={color}
                    strokeDasharray={circumference}
                    strokeDashoffset={offset}
                />
                <text className="gauge-text" x="80" y="74">
                    {pct}%
                </text>
                <text className="gauge-label" x="80" y="98">
                    Confidence
                </text>
            </svg>
        </div>
    );
}
