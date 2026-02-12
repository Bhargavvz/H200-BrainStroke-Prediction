import React, { useRef, useState, useCallback } from 'react';

export default function MRIUpload({ file, onFileChange }) {
    const inputRef = useRef(null);
    const [dragOver, setDragOver] = useState(false);
    const [preview, setPreview] = useState(null);

    const handleFile = useCallback((f) => {
        if (!f) return;
        if (!['image/jpeg', 'image/png', 'image/jpg'].includes(f.type)) {
            alert('Please upload a JPEG or PNG image.');
            return;
        }
        onFileChange(f);
        const reader = new FileReader();
        reader.onload = (e) => setPreview(e.target.result);
        reader.readAsDataURL(f);
    }, [onFileChange]);

    const handleDrop = useCallback((e) => {
        e.preventDefault();
        setDragOver(false);
        const f = e.dataTransfer?.files?.[0];
        if (f) handleFile(f);
    }, [handleFile]);

    const handleRemove = (e) => {
        e.stopPropagation();
        onFileChange(null);
        setPreview(null);
        if (inputRef.current) inputRef.current.value = '';
    };

    return (
        <div className="glass-card">
            <div className="card-title">🧠 MRI Brain Scan</div>
            <div className="card-description">
                Upload an axial brain MRI or CT image (JPEG / PNG)
            </div>

            <div
                className={`upload-zone${dragOver ? ' drag-over' : ''}`}
                onClick={() => inputRef.current?.click()}
                onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
                onDragLeave={() => setDragOver(false)}
                onDrop={handleDrop}
            >
                <input
                    ref={inputRef}
                    type="file"
                    accept="image/jpeg,image/png"
                    style={{ display: 'none' }}
                    onChange={(e) => handleFile(e.target.files?.[0])}
                />

                {preview ? (
                    <div className="upload-preview">
                        <img src={preview} alt="MRI Preview" />
                        <button className="upload-preview__remove" onClick={handleRemove} title="Remove">
                            ✕
                        </button>
                    </div>
                ) : (
                    <>
                        <span className="upload-zone__icon">🖼️</span>
                        <div className="upload-zone__text">
                            Drag & drop your MRI image here
                        </div>
                        <div className="upload-zone__hint">
                            or click to browse · JPEG, PNG up to 50 MB
                        </div>
                    </>
                )}
            </div>
        </div>
    );
}
