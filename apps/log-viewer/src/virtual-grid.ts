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

function escapeHtml(str: string): string {
  return str.replace(/[&<>"']/g, (c) => ({
    '&': '&amp;',
    '<': '&lt;',
    '>': '&gt;',
    '"': '&quot;',
    "'": '&#39;'
  }[c] || c));
}

/**
 * Defense-in-depth sanitizer for library-provided `highlightedText`.
 * The library escapes with `escapeHtml:true`, but the grid inserts via
 * `innerHTML` — re-escape everything except `<mark>`/`</mark>` so a future
 * library regression or a caller omitting the flag cannot become stored XSS.
 */
function sanitizeHighlighted(html: string): string {
  const escaped = escapeHtml(html);
  return escaped
    .replace(/&lt;mark&gt;/g, '<mark>')
    .replace(/&lt;\/mark&gt;/g, '</mark>');
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
  private resizeObserver: ResizeObserver | null = null;

  private isTicking: boolean = false;

  constructor(options: VirtualGridOptions) {
    this.container = options.container;
    this.rowHeight = options.rowHeight ?? 36;
    this.overscan = options.overscan ?? 10;
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

    // Single delegated click listener for all grid rows
    this.rowsContainer.addEventListener('click', (e) => {
      if (!this.onRowClick) return;
      const rowEl = (e.target as HTMLElement).closest('.log-grid-row');
      if (!rowEl) return;
      const idx = parseInt(rowEl.getAttribute('data-index') || '-1', 10);
      if (idx >= 0 && this.items[idx]) {
        this.onRowClick(this.items[idx]!);
      }
    });

    // Observe container resizing to prevent blank viewport zones
    if (typeof ResizeObserver !== 'undefined') {
      this.resizeObserver = new ResizeObserver(() => {
        if (!this.isTicking) {
          requestAnimationFrame(() => {
            this.render();
            this.isTicking = false;
          });
          this.isTicking = true;
        }
      });
      this.resizeObserver.observe(this.container);
    }
  }

  setItems(items: VirtualGridItem[], options?: { resetScroll?: boolean }): void {
    this.items = items;
    if (options?.resetScroll ?? true) {
      this.container.scrollTop = 0;
    }
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

      const levelClass = `level-${escapeHtml(r.level.toLowerCase())}`;
      const msg = item.highlightedText?.message ? sanitizeHighlighted(item.highlightedText.message) : escapeHtml(r.message);
      const svc = item.highlightedText?.service ? sanitizeHighlighted(item.highlightedText.service) : escapeHtml(r.service);
      const trace = item.highlightedText?.traceId ? sanitizeHighlighted(item.highlightedText.traceId) : escapeHtml(r.traceId);

      const scoreCol = item.score !== undefined
        ? `<span class="score-pill">${item.score}</span>`
        : `<span class="latency-val">${r.latencyMs}ms</span>`;

      html += `
        <div class="log-grid-row" data-index="${rowIdx}" style="height: ${this.rowHeight}px;">
          <div class="grid-cell col-id">${escapeHtml(r.id)}</div>
          <div class="grid-cell col-time">${escapeHtml(r.timestamp.slice(11, 23))}</div>
          <div class="grid-cell col-level"><span class="level-badge ${levelClass}">${escapeHtml(r.level)}</span></div>
          <div class="grid-cell col-service">${svc}</div>
          <div class="grid-cell col-message" title="${escapeHtml(r.message)}">${msg}</div>
          <div class="grid-cell col-trace">${trace}</div>
          <div class="grid-cell col-score">${scoreCol}</div>
        </div>
      `;
    }

    this.rowsContainer.innerHTML = html;
  }

  getItemCount(): number {
    return this.items.length;
  }

  destroy(): void {
    if (this.resizeObserver) {
      this.resizeObserver.disconnect();
      this.resizeObserver = null;
    }
  }
}
