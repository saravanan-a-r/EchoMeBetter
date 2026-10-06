import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import '../styles/pages.css';
import { PopupApp } from './PopupApp';

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <PopupApp />
  </StrictMode>,
);
