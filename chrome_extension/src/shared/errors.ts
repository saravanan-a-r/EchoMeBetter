/**
 * Every failure a user can run into, as a closed set of codes.
 *
 * Errors cross three process boundaries (worker → offscreen document →
 * service worker → page), so they travel as plain `{ code, details }`
 * payloads and are turned back into copy only at the edge that shows them.
 */
export type ErrorCode =
  | 'NO_SELECTION'
  | 'UNSUPPORTED_FIELD'
  | 'PASSWORD_FIELD'
  | 'INPUT_TOO_LONG'
  | 'RESERVED_MARKUP'
  | 'TEXT_CHANGED'
  | 'FRAME_INACCESSIBLE'
  | 'BUSY'
  | 'MODEL_LOAD_FAILED'
  | 'INFERENCE_FAILED'
  | 'EMPTY_RESULT'
  | 'OUTPUT_TOO_LONG'
  | 'CANCELLED'
  | 'DISCONNECTED'
  | 'INTERNAL';

export type ErrorDetails = Readonly<Record<string, string | number>>;

export interface ErrorPayload {
  readonly code: ErrorCode;
  readonly details?: ErrorDetails;
}

export class EchoError extends Error {
  readonly code: ErrorCode;
  readonly details: ErrorDetails;

  constructor(code: ErrorCode, details: ErrorDetails = {}, message?: string) {
    super(message ?? `${code}${Object.keys(details).length ? ` ${JSON.stringify(details)}` : ''}`);
    this.name = 'EchoError';
    this.code = code;
    this.details = details;
  }

  toPayload(): ErrorPayload {
    return { code: this.code, details: this.details };
  }
}

/** Normalise anything thrown into a payload that can cross a message boundary. */
export function toErrorPayload(error: unknown, fallback: ErrorCode = 'INTERNAL'): ErrorPayload {
  if (error instanceof EchoError) return error.toPayload();
  const message = error instanceof Error ? error.message : String(error);
  return { code: fallback, details: { message } };
}

export function isErrorPayload(value: unknown): value is ErrorPayload {
  return typeof value === 'object' && value !== null && typeof (value as ErrorPayload).code === 'string';
}

/** The sentence shown to the user. Kept short: it lives in a small toast. */
export function describeError(error: ErrorPayload): string {
  const details = error.details ?? {};
  switch (error.code) {
    case 'NO_SELECTION':
      return 'Select the text you want to rewrite first.';
    case 'UNSUPPORTED_FIELD':
      return "This kind of field can't be rewritten. Try a regular text box.";
    case 'PASSWORD_FIELD':
      return 'Password fields are never read or rewritten.';
    case 'INPUT_TOO_LONG':
      return `That selection is too long (${details.actual ?? '?'} of ${details.limit ?? '?'} tokens). Try a shorter passage.`;
    case 'RESERVED_MARKUP':
      return 'The selection contains markup the model reserves for itself, so it was left unchanged.';
    case 'TEXT_CHANGED':
      return 'The text changed while it was being rewritten, so it was left as is.';
    case 'FRAME_INACCESSIBLE':
      return "EchoMeBetter can't reach this text box: it's inside a frame from another site.";
    case 'BUSY':
      return 'Still working on your last rewrite. One moment.';
    case 'MODEL_LOAD_FAILED':
      return "The writing model couldn't be loaded. Open the EchoMeBetter popup to retry.";
    case 'INFERENCE_FAILED':
      return 'Something went wrong while rewriting. Please try again.';
    case 'EMPTY_RESULT':
      return "The model didn't produce a rewrite for that text.";
    case 'OUTPUT_TOO_LONG':
      return "The rewrite ran past the model's length limit, so it was discarded. Try a shorter passage.";
    case 'CANCELLED':
      return 'Rewrite cancelled.';
    case 'DISCONNECTED':
      return 'EchoMeBetter was updated or restarted. Please try again.';
    case 'INTERNAL':
      return 'Something unexpected went wrong. Please try again.';
  }
}
