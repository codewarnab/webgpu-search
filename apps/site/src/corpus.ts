import './fonts.css';
import './style.css';
import { normalizeText, scoreExactMatches, type SearchResultItem } from 'webgpu-search';
import { generateCorpus } from './corpus-gen';

const PAGE_SIZE = 100;
const MAX_MATCHES_SHOWN = 1000;

const meta = document.querySelector<HTMLElement>('#c-meta');
const matchesList = document.querySelector<HTMLOListElement>('#c-matches');
const matchesPager = document.querySelector<HTMLElement>('#c-matches-pager');
const corpusList = document.querySelector<HTMLOListElement>('#c-corpus');
const corpusPager = document.querySelector<HTMLElement>('#c-corpus-pager');

function escapeHtml(value: string): string {
  return value.replace(/[&<>"']/g, char => (
    { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[char] ?? char
  ));
}

function highlight(text: string, query: string): string {
  const safe = escapeHtml(text);
  const needle = query.trim().toLowerCase();
  if (!needle) return safe;
  const at = text.toLowerCase().indexOf(needle);
  if (at < 0) return safe;
  return `${escapeHtml(text.slice(0, at))}<mark>${escapeHtml(text.slice(at, at + needle.length))}</mark>${escapeHtml(text.slice(at + needle.length))}`;
}

function attachPager(
  pager: HTMLElement | null,
  list: HTMLOListElement | null,
  rows: string[],
  renderRow: (row: string, absoluteIndex: number) => string
): void {
  if (!pager || !list) return;
  const label = pager.querySelector('span');
  const prev = pager.querySelector<HTMLButtonElement>('[data-page="prev"]');
  const next = pager.querySelector<HTMLButtonElement>('[data-page="next"]');
  let page = 0;
  const pages = Math.max(1, Math.ceil(rows.length / PAGE_SIZE));
  const render = (): void => {
    const start = page * PAGE_SIZE;
    list.innerHTML = rows
      .slice(start, start + PAGE_SIZE)
      .map((row, i) => renderRow(row, start + i))
      .join('');
    if (label) label.textContent = `Page ${page + 1} of ${pages}`;
    if (prev) prev.disabled = page === 0;
    if (next) next.disabled = page >= pages - 1;
  };
  prev?.addEventListener('click', () => { if (page > 0) { page -= 1; render(); } });
  next?.addEventListener('click', () => { if (page < pages - 1) { page += 1; render(); } });
  pager.hidden = false;
  render();
}

function main(): void {
  const params = new URLSearchParams(location.search);
  const query = (params.get('query') ?? params.get('q') ?? '').slice(0, 200);
  const requested = Number(params.get('size'));
  const size = Number.isFinite(requested)
    ? Math.min(2000000, Math.max(1, Math.floor(requested)))
    : 100000;

  if (!query) {
    if (meta) meta.textContent = 'No query given. Run the benchmark on the homepage first, then follow its link.';
    return;
  }

  const strings = generateCorpus(size);
  const tokens = strings.map(value => normalizeText(value, true).tokens);
  const queryTokens = normalizeText(query, true).tokens;
  const response = scoreExactMatches(tokens, queryTokens, 'fuzzy', MAX_MATCHES_SHOWN, strings);
  const shown: SearchResultItem[] = response.results;
  const hiddenCount = Math.max(0, response.totalMatches - shown.length);

  if (meta) meta.textContent =
    `Query “${query}” over ${size.toLocaleString()} generated records · ${response.totalMatches.toLocaleString()} matches.`;

  if (matchesList) {
    if (shown.length === 0) {
      matchesList.innerHTML = '<li>No matches for this query.</li>';
    } else {
      const rows = shown.map(hit =>
        `<li><span class="row-index">#${hit.index}</span> ${highlight(hit.text, query)} <span class="row-score">score ${hit.score}</span></li>`
      );
      if (hiddenCount > 0) rows.push(`<li>… and ${hiddenCount.toLocaleString()} more (showing top ${shown.length.toLocaleString()})</li>`);
      attachPager(matchesPager, matchesList, rows, row => row);
    }
  }

  if (corpusList) {
    attachPager(corpusPager, corpusList, strings,
      (line, index) => `<li><span class="row-index">#${index}</span> ${escapeHtml(line)}</li>`);
  }
}

main();
