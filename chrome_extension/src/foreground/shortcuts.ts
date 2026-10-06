// Content script registered on every site once the user allows shortcuts there (see background/shortcutAccess.ts).
import { installShortcuts } from './shortcutsBootstrap';

installShortcuts();
