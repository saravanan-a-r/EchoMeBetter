// Content script registered on every site once the user allows EchoMeBetter there, for the
// shortcuts and the press and hold on selected text (see background/shortcutAccess.ts).
import { installShortcuts } from './shortcutsBootstrap';

installShortcuts();
