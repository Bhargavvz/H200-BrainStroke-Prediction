import axios from 'axios';

const API_BASE = 'http://localhost:8000';

export async function predictStroke(mriFile, clinicalData) {
    const formData = new FormData();
    formData.append('mri_image', mriFile);
    formData.append('clinical_data', JSON.stringify(clinicalData));

    const response = await axios.post(`${API_BASE}/predict`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
        timeout: 60000,
    });

    return response.data;
}

export async function getHealth() {
    const response = await axios.get(`${API_BASE}/health`);
    return response.data;
}

export async function getModelInfo() {
    const response = await axios.get(`${API_BASE}/model-info`);
    return response.data;
}
