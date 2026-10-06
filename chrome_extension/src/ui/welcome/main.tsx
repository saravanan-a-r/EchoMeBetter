import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import { installForeground } from '../../foreground/bootstrap';
import { installShortcuts } from '../../foreground/shortcutsBootstrap';
import '../styles/pages.css';
import { WelcomeApp } from './WelcomeApp';

// The practice box on this page uses the same in-page experience as any
// website, shortcuts included; extension pages cannot be injected into, so
// it is loaded here.
installForeground();
installShortcuts();

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <WelcomeApp />
  </StrictMode>,
);
