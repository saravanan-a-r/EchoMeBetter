// Entry point injected into a page frame on demand (see background/rewriteLauncher.ts).
import { installForeground } from './bootstrap';

installForeground();
