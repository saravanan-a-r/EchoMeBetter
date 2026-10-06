/**
 * The model's card, the same on the popup and the welcome page: a download
 * in progress (or one that stopped) first, then the model's status once it
 * is installed, and the first-run setup before that.
 */
import type { StyleId } from '../../shared/styles';
import type { EngineStatus } from '../../shared/status';
import type { CatalogState, ModelAvailability } from '../hooks/useModel';
import { ModelDownloadCard } from './ModelDownloadCard';
import { ModelSetupCard } from './ModelSetupCard';
import { StatusCard } from './StatusCard';

export interface ModelCardProps {
  readonly status: EngineStatus;
  readonly model: ModelAvailability;
  readonly catalog: CatalogState;
  readonly onLoadModel: () => void;
  readonly onDownload: (adapters: StyleId[]) => void;
  readonly onCancelDownload: () => void;
  readonly onRetryCatalog: () => void;
  /** Shown on the setup card, e.g. right after the model was removed. */
  readonly notice?: string;
}

export function ModelCard({ status, model, catalog, onLoadModel, onDownload, onCancelDownload, onRetryCatalog, notice }: ModelCardProps) {
  if (!model.known) return null;
  if (model.download.state !== 'idle') {
    return <ModelDownloadCard download={model.download} onRetry={(target) => onDownload([...target.adapters])} onCancel={onCancelDownload} />;
  }
  if (model.installed) return <StatusCard status={status} onLoad={onLoadModel} />;
  return <ModelSetupCard catalog={catalog} onDownload={onDownload} onRetryCatalog={onRetryCatalog} notice={notice} />;
}
