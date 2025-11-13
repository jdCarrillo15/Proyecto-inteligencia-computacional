import axios from 'axios';
import { API_URL } from '../data/config';

// Enviar predicción
export const predictDisease = async (file) => {
  const formData = new FormData();
  formData.append('file', file);

  const response = await axios.post(`${API_URL}/predict`, formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  });

  return response.data;
};
