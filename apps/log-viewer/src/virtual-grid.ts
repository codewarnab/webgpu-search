import type { LogViewerSearchResult, StructuredLogRecord } from './types';

export interface VirtualGridItem {
  record: StructuredLogRecord;
  score?: number;
  matchedField?: string;
  highlightedText?: Record<string, string>;
}

export interface VirtualGridOptions {
  container: HTMLElement;
  rowHeight?: number;
  overscan?: number;
  onRowClick?: (item: VirtualGridItem) => void;
}

export class VirtualGrid {
  private container: HTMLElement;
  private rowHeight: number;
  private overscan: number;
  private items: VirtualGridItem[] = [];
  private onRowClick?: (item: VirtualGridItem) => void;

  private scrollWrapper: HTMLDivElement;
  private topSpacer: HTMLDivElement;
  private bottomSpacer: HTMLDivElement;
  private rowsContainer: HTMLDivElement;

  private isTicking: boolean = false;

  constructor(options: VirtualGridOptions) {
    this.container = options.container;
    this.rowHeight = options.rowHeight ?? 36;
    this.overscan = options.overscan ?? 6;
    this.onRowClick = options.onRowClick;

    this.container.innerHTML = '';
    this.container.style.overflowY = 'auto';
    this.container.style.position = 'relative';

    this.scrollWrapper = document.createElement('div');
    this.scrollWrapper.className = 'virtual-scroll-wrapper';
    this.scrollWrapper.style.position = 'relative';
    this.scrollWrapper.style.width = '100%';

    this.topSpacer = document.createElement('div');
    this.bottomSpacer = document.createElement('div');
    this.rowsContainer = document.createElement('div');
    this.rowsContainer.className = 'virtual-rows-container';

    this.scrollWrapper.appendChild(this.topSpacer);
    this.scrollWrapper.appendChild(this.rowsContainer);
    this.scrollWrapper.appendChild(this.bottomSpacer);
    this.container.appendChild(this.scrollWrapper);

    this.container.addEventListener('scroll', () => {
      if (!this.isTicking) {
        requestAnimationFrame(() => {
          this.render();
          this.isTicking = false;
        });
        this.isTicking = true;
      }
    });
  }

  setItems(items: VirtualGridItem[]): void {
    this.items = items;
    this.container.scrollTop = 0;
    this.render();
  }

  setSearchResults(results: LogViewerSearchResult[]): void {
    const items: VirtualGridItem[] = results.map((r) => ({
      record: r.doc,
      score: r.score,
      matchedField: r.matchedField,
      highlightedText: r.highlightedText
    }));
    this.setItems(items);
  }

  render(): void {
    const totalCount = this.items.length;
    const containerHeight = this.container.clientHeight || 500;
    const scrollTop = this.container.scrollTop;

    if (totalCount === 0) {
      this.topSpacer.style.height = '0px';
      this.bottomSpacer.style.height = '0px';
      this.rowsContainer.innerHTML = '<div class="grid-empty-state">No matching log entries found.</div>';
      return;
    }

    const startIndex = Math.max(0, Math.floor(scrollTop / this.rowHeight) - this.overscan);
    const endIndex = Math.min(totalCount, Math.ceil((scrollTop + containerHeight) / this.rowHeight) + this.overscan);

    const topHeight = startIndex * this.rowHeight;
    const bottomHeight = Math.max(0, (totalCount - endIndex) * this.rowHeight);

    this.topSpacer.style.height = `${topHeight}px`;
    this.bottomSpacer.style.height = `${bottomHeight}px`;

    const visibleItems = this.items.slice(startIndex, endIndex);

    let html = '';
    for (let i = 0; i < visibleItems.length; i++) {
      const item = visibleItems[i]!;
      const r = item.record;
      const rowIdx = startIndex + i;

      const levelClass = `level-${r.level.toLowerCase()}`;
      const msg = item.highlightedText?.message || r.message;
      const svc = item.highlightedText?.service || r.service;
      const trace = item.highlightedText?.traceId || r.traceId;

      const scoreCol = item.score !== undefined
        ? `<span class="score-pill">${item.score}</span>`
        : `<span class="latency-val">${r.latencyMs}ms</span>`;

      html += `
        <div class="log-grid-row" data-index="${rowIdx}" style="height: ${this.rowHeight}px;">
          <div class="grid-cell col-id">${r.id}</div>
          <div class="grid-cell col-time">${r.timestamp.slice(11, 23)}</div>
          <div class="grid-cell col-level"><span class="level-badge ${levelClass}">${r.level}</span></div>
          <div class="grid-cell col-service">${svc}</div>
          <div class="grid-cell col-message" title="${r.message}">${msg}</div>
          <div class="grid-cell col-trace">${trace}</div>
          <div class="grid-cell col-score">${scoreCol}</div>
        </div>
      `;
    }

    this.rowsContainer.innerHTML = html;

    if (this.onRowClick) {
      const rowElements = this.rowsContainer.querySelectorAll('.log-grid-row');
      rowElements.forEach((el) => {
        el.addEventListener('click', () => {
          const idx = parseInt(el.getAttribute('data-index') || '-1', 10);
          if (idx >= 0 && this.items[idx]) {
            this.onRowClick!(this.items[idx]!);
          }
        });
      });
    }
  }

  getItemCount(): number {
    return this.items.length;
  }
}
