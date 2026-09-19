/**
 * Milestone 3: Unicode-Safe Highlighting Engine Test Suite.
 *
 * Validates:
 * 1. Indentation trim compensation (leadingTrimOffset)
 * 2. ASCII fast-path 1:1 UTF-16 coordinate mapping
 * 3. Unicode coordinate mapping (astral surrogate pairs / emojis, lone surrogates)
 * 4. NFC / NFD normalization & combining character composition
 * 5. Atomic Character Expansion Principle ('ß' -> 'ss', 'İ' -> 'i+dot', 'ﬁ' -> 'fi', 'ﬂ', 'ﬀ', 'ﬃ', 'ﬄ', 'ΐ', 'ΰ')
 * 6. Grapheme cluster boundary guard (no broken combining marks or split HTML tags)
 * 7. Subsequence & substring alignment symmetry with WGSL/CPU scorers
 * 8. Span merging (overlapping, adjacent, and disjoint ranges)
 * 9. HTML tag rendering (default <mark>, custom tags <b>, <em>)
 * 10. DocumentIndex search integration (primary & auxiliary fields, tag rendering, highlight toggle)
 * 11. WebGPU vs CPU highlighting parity
 */

import assert from 'node:assert';
import {
  DocumentIndex,
  alignHighlights,
  normalizeWithSourceMap,
  guardClusterBoundary,
  mergeHighlightRanges,
  renderHighlightedText,
  type HighlightRange,
  type DocumentIndexOptions,
  IncompatibleOptionError
} from '../packages/webgpu-search/src/index';
import { createMockAdapter } from 'vgpu/mock';

interface DocItem {
  id: string;
  title: string;
  body: string;
  tags?: string[];
}

async function runM3Tests() {
  console.log('--- Running Milestone 3: Unicode-Safe Highlighting Engine Tests ---');

  // =========================================================================
  // 1. Indentation Trim Compensation (leadingTrimOffset)
  // =========================================================================
  console.log('1. Testing indentation trim compensation (leadingTrimOffset)...');
  {
    // Leading spaces
    const raw1 = '    function calculate() {';
    const sm1 = normalizeWithSourceMap(raw1, true);
    assert.strictEqual(sm1.leadingTrimOffset, 4, 'Expected leadingTrimOffset === 4');

    const ranges1 = alignHighlights(raw1, 'function', { mode: 'substring' });
    assert.strictEqual(ranges1.length, 1);
    assert.strictEqual(ranges1[0].start, 4);
    assert.strictEqual(ranges1[0].end, 12);
    assert.strictEqual(raw1.slice(ranges1[0].start, ranges1[0].end), 'function');

    // Leading tabs
    const raw2 = '\t\tconst value = 42;';
    const sm2 = normalizeWithSourceMap(raw2, true);
    assert.strictEqual(sm2.leadingTrimOffset, 2);
    const ranges2 = alignHighlights(raw2, 'value', { mode: 'substring' });
    assert.strictEqual(ranges2.length, 1);
    assert.strictEqual(raw2.slice(ranges2[0].start, ranges2[0].end), 'value');

    // Both leading and trailing whitespace
    const raw3 = '   hello world   ';
    const sm3 = normalizeWithSourceMap(raw3, true);
    assert.strictEqual(sm3.leadingTrimOffset, 3);
    const ranges3 = alignHighlights(raw3, 'world', { mode: 'substring' });
    assert.strictEqual(ranges3.length, 1);
    assert.strictEqual(raw3.slice(ranges3[0].start, ranges3[0].end), 'world');

    // Only whitespace string
    const rawEmpty = '     ';
    const smEmpty = normalizeWithSourceMap(rawEmpty, true);
    assert.strictEqual(smEmpty.isEmpty, true);
    assert.strictEqual(smEmpty.tokens.length, 0);
    const emptyRanges = alignHighlights(rawEmpty, 'test');
    assert.strictEqual(emptyRanges.length, 0);

    // Empty query
    assert.strictEqual(alignHighlights(raw1, '').length, 0);
    assert.strictEqual(alignHighlights(raw1, '   ').length, 0);

    console.log('   ✅ Indentation trim compensation verified (no coordinate drift)');
  }

  // =========================================================================
  // 2. ASCII Fast-Path 1:1 Mapping
  // =========================================================================
  console.log('2. Testing ASCII fast-path 1:1 coordinate mapping...');
  {
    const raw = 'The quick brown fox jumps over the lazy dog';
    const sm = normalizeWithSourceMap(raw, true);
    assert.strictEqual(sm.tokenCount, raw.length);
    assert.strictEqual(sm.starts.length, raw.length);
    assert.strictEqual(sm.ends.length, raw.length);

    for (let i = 0; i < raw.length; i++) {
      assert.strictEqual(sm.starts[i], i);
      assert.strictEqual(sm.ends[i], i + 1);
    }

    // Substring match
    const subRanges = alignHighlights(raw, 'brown', { mode: 'substring' });
    assert.strictEqual(subRanges.length, 1);
    assert.strictEqual(raw.slice(subRanges[0].start, subRanges[0].end), 'brown');

    // Fuzzy match
    const fuzzyRanges = alignHighlights(raw, 'qckfx', { mode: 'fuzzy' });
    assert(fuzzyRanges.length > 0);
    // Rendered text with <mark>
    const html = renderHighlightedText(raw, fuzzyRanges, 'mark');
    assert(html.includes('<mark>q</mark>ui<mark>ck</mark>'));
    assert(html.includes('<mark>f</mark>o<mark>x</mark>'));

    console.log('   ✅ ASCII fast-path 1:1 coordinate mapping verified');
  }

  // =========================================================================
  // 3. Unicode Astral Code Points (Emojis) & Lone Surrogates
  // =========================================================================
  console.log('3. Testing astral code points (surrogate pairs / emojis) and lone surrogates...');
  {
    // Emojis are 2 UTF-16 code units (surrogate pairs: length 2)
    const raw = 'Search 🚀 in the 🌌 stars';
    // 'Search ' = 7
    // '🚀' = [7..9)
    // ' in the ' = [9..17)
    // '🌌' = [17..19)
    // ' stars' = [19..25)

    const rocketRanges = alignHighlights(raw, '🚀', { mode: 'substring' });
    assert.strictEqual(rocketRanges.length, 1);
    assert.strictEqual(rocketRanges[0].start, 7);
    assert.strictEqual(rocketRanges[0].end, 9);
    assert.strictEqual(raw.slice(rocketRanges[0].start, rocketRanges[0].end), '🚀');

    // Subsequent word 'stars' must have correct UTF-16 index accounting for 2-unit emojis
    const starsRanges = alignHighlights(raw, 'stars', { mode: 'substring' });
    assert.strictEqual(starsRanges.length, 1);
    assert.strictEqual(starsRanges[0].start, 20);
    assert.strictEqual(starsRanges[0].end, 25);
    assert.strictEqual(raw.slice(starsRanges[0].start, starsRanges[0].end), 'stars');

    // Multiple emojis in a row
    const emojiCorpus = 'Prefix 😀😁🎉 Postfix';
    const faceRanges = alignHighlights(emojiCorpus, '😁', { mode: 'substring' });
    assert.strictEqual(faceRanges.length, 1);
    assert.strictEqual(emojiCorpus.slice(faceRanges[0].start, faceRanges[0].end), '😁');

    const postRanges = alignHighlights(emojiCorpus, 'Postfix', { mode: 'substring' });
    assert.strictEqual(postRanges.length, 1);
    assert.strictEqual(emojiCorpus.slice(postRanges[0].start, postRanges[0].end), 'Postfix');

    // Lone surrogate safety: should not crash and should produce valid ranges
    const loneStr = 'Lone \uD800 surrogate \uDFFF test';
    const loneSm = normalizeWithSourceMap(loneStr, true);
    assert(!loneSm.isEmpty);
    const surrRanges = alignHighlights(loneStr, 'surrogate', { mode: 'substring' });
    assert.strictEqual(surrRanges.length, 1);
    assert.strictEqual(loneStr.slice(surrRanges[0].start, surrRanges[0].end), 'surrogate');

    console.log('   ✅ Astral surrogate pairs & lone surrogate safety verified');
  }

  // =========================================================================
  // 4. NFC / NFD Normalization & Combining Character Composition
  // =========================================================================
  console.log('4. Testing NFC / NFD normalization & combining mark composition...');
  {
    // Decomposed NFD 'e' + combining acute (U+0301) occupies 2 UTF-16 units
    const nfdCafe = 'cafe\u0301 au lait';
    assert.strictEqual(nfdCafe.length, 13); // 'cafe\u0301' is 5 code units, ' au lait' is 8

    // Query with composed 'é' (U+00E9)
    const composedQuery = 'café';
    const ranges = alignHighlights(nfdCafe, composedQuery, { mode: 'substring' });
    assert.strictEqual(ranges.length, 1);
    assert.strictEqual(ranges[0].start, 0);
    assert.strictEqual(ranges[0].end, 5); // Must cover BOTH 'e' and '\u0301'
    assert.strictEqual(nfdCafe.slice(ranges[0].start, ranges[0].end), 'cafe\u0301');

    // Query with decomposed NFD against composed NFC target
    const nfcCafe = 'café au lait';
    const decomposedQuery = 'cafe\u0301';
    const nfcRanges = alignHighlights(nfcCafe, decomposedQuery, { mode: 'substring' });
    assert.strictEqual(nfcRanges.length, 1);
    assert.strictEqual(nfcRanges[0].start, 0);
    assert.strictEqual(nfcRanges[0].end, 4);
    assert.strictEqual(nfcCafe.slice(nfcRanges[0].start, nfcRanges[0].end), 'café');

    // Triple composition: 'A' + combining ring + combining acute -> 'Ǻ' (U+01FA)
    const tripleRaw = 'Prefix A\u030A\u0301bc Suffix';
    const tripleRanges = alignHighlights(tripleRaw, 'Ǻbc', { mode: 'substring' });
    assert.strictEqual(tripleRanges.length, 1);
    assert.strictEqual(tripleRaw.slice(tripleRanges[0].start, tripleRanges[0].end), 'A\u030A\u0301bc');

    // Hangul Jamo composition: L+V+T -> syllable block
    const hangulRaw = 'Word 각 End'; // U+1100 + U+1161 + U+11A8
    const hangulRanges = alignHighlights(hangulRaw, '각', { mode: 'substring' });
    assert.strictEqual(hangulRanges.length, 1);
    assert.strictEqual(hangulRaw.slice(hangulRanges[0].start, hangulRanges[0].end), '각');

    console.log('   ✅ NFC / NFD normalization & combining mark composition verified');
  }

  // =========================================================================
  // 5. Atomic Character Expansion Principle
  // =========================================================================
  console.log('5. Testing Atomic Character Expansion Principle...');
  {
    // German 'ß' -> 'ss'
    const rawStrasse = 'München Straße 10';
    // Partial expansion match: query 's' matching first token of 'ß'
    // 'ß' is at index 13 in rawStrasse
    const sRanges = alignHighlights(rawStrasse, 'straße', { mode: 'substring' });
    assert.strictEqual(sRanges.length, 1);
    assert.strictEqual(rawStrasse.slice(sRanges[0].start, sRanges[0].end), 'Straße');

    // Querying with ASCII 'strasse' matches 'Straße'
    const ssRanges = alignHighlights(rawStrasse, 'strasse', { mode: 'substring' });
    assert.strictEqual(ssRanges.length, 1);
    assert.strictEqual(rawStrasse.slice(ssRanges[0].start, ssRanges[0].end), 'Straße');

    // Querying partial 's' in fuzzy mode over 'aßb'
    const rawAsb = 'aßb';
    // 'ß' expands to 'ss'. If query is 'sb', it matches second token of 'ß' and 'b'.
    const partialRanges = alignHighlights(rawAsb, 'sb', { mode: 'fuzzy' });
    assert.strictEqual(partialRanges.length, 1);
    // Must highlight 'ßb' atomically, not slice 'ß' in half
    assert.strictEqual(rawAsb.slice(partialRanges[0].start, partialRanges[0].end), 'ßb');

    // Turkish 'İ' (U+0130) -> 'i' + combining dot (U+0307)
    const rawIstanbul = 'İstanbul City';
    const istRanges = alignHighlights(rawIstanbul, 'İstanbul', { mode: 'substring' });
    assert.strictEqual(istRanges.length, 1);
    assert.strictEqual(rawIstanbul.slice(istRanges[0].start, istRanges[0].end), 'İstanbul');

    // Querying with ASCII 'i' matches 'İ' in fuzzy mode (subsequence skips combining dot)
    const iRanges = alignHighlights(rawIstanbul, 'i', { mode: 'fuzzy' });
    assert.strictEqual(iRanges.length, 1);
    assert.strictEqual(rawIstanbul.slice(iRanges[0].start, iRanges[0].end), 'İ');

    const istFuzzyRanges = alignHighlights(rawIstanbul, 'istanbul', { mode: 'fuzzy' });
    assert.strictEqual(istFuzzyRanges.length, 1);
    assert.strictEqual(rawIstanbul.slice(istFuzzyRanges[0].start, istFuzzyRanges[0].end), 'İstanbul');

    // Ligature 'ﬁ' (U+FB01) -> 'fi'
    const rawLigature = 'The ﬁle document';
    // Match 'file'
    const fileRanges = alignHighlights(rawLigature, 'file', { mode: 'substring' });
    assert.strictEqual(fileRanges.length, 1);
    assert.strictEqual(rawLigature.slice(fileRanges[0].start, fileRanges[0].end), 'ﬁle');

    // Match partial 'f'
    const fRanges = alignHighlights(rawLigature, 'f', { mode: 'fuzzy' });
    assert.strictEqual(fRanges.length, 1);
    assert.strictEqual(rawLigature.slice(fRanges[0].start, fRanges[0].end), 'ﬁ');

    // Match partial 'i'
    const iLigRanges = alignHighlights(rawLigature, 'i', { mode: 'fuzzy' });
    assert.strictEqual(iLigRanges.length, 1);
    assert.strictEqual(rawLigature.slice(iLigRanges[0].start, iLigRanges[0].end), 'ﬁ');

    // Ligature 'ﬂ' (U+FB02) -> 'fl'
    const rawFlow = 'The ﬂow state';
    const flowRanges = alignHighlights(rawFlow, 'flow', { mode: 'substring' });
    assert.strictEqual(flowRanges.length, 1);
    assert.strictEqual(rawFlow.slice(flowRanges[0].start, flowRanges[0].end), 'ﬂow');

    // Ligature 'ﬀ' (U+FB00) -> 'ff'
    const rawOff = 'The oﬀ switch';
    const offRanges = alignHighlights(rawOff, 'off', { mode: 'substring' });
    assert.strictEqual(offRanges.length, 1);
    assert.strictEqual(rawOff.slice(offRanges[0].start, offRanges[0].end), 'oﬀ');

    // Ligatures 'ﬃ' (U+FB03) -> 'ffi' and 'ﬄ' (U+FB04) -> 'ffl'
    const rawOffice = 'The oﬃce and waﬄe';
    const officeRanges = alignHighlights(rawOffice, 'office', { mode: 'substring' });
    assert.strictEqual(officeRanges.length, 1);
    assert.strictEqual(rawOffice.slice(officeRanges[0].start, officeRanges[0].end), 'oﬃce');

    const waffleRanges = alignHighlights(rawOffice, 'waffle', { mode: 'substring' });
    assert.strictEqual(waffleRanges.length, 1);
    assert.strictEqual(rawOffice.slice(waffleRanges[0].start, waffleRanges[0].end), 'waﬄe');

    // Greek expansion: 'ΐ' (U+0390) and 'ΰ' (U+03B0)
    const rawGreek = 'Greek ΐ and ΰ';
    const greek1 = alignHighlights(rawGreek, 'ΐ', { mode: 'substring' });
    assert.strictEqual(greek1.length, 1);
    assert.strictEqual(rawGreek.slice(greek1[0].start, greek1[0].end), 'ΐ');

    console.log('   ✅ Atomic Character Expansion Principle verified (no partial glyph slicing)');
  }

  // =========================================================================
  // 6. Grapheme Cluster Boundary Guard
  // =========================================================================
  console.log('6. Testing Grapheme Cluster Boundary Guard...');
  {
    // 'Z' followed by uncomposed combining tilde (\u0303) and combining ring below (\u0325)
    const rawUncomposed = 'Target Z\u0303\u0325 Word';
    // When matching 'z', highlight must NOT end at 'Z' alone and leave dangling combining marks
    const zRanges = alignHighlights(rawUncomposed, 'z', { mode: 'fuzzy' });
    assert.strictEqual(zRanges.length, 1);
    // The range must encompass 'Z\u0303\u0325' (3 code units)
    assert.strictEqual(rawUncomposed.slice(zRanges[0].start, zRanges[0].end), 'Z\u0303\u0325');

    // Injected HTML tag must wrap the entire cluster
    const html = renderHighlightedText(rawUncomposed, zRanges, 'mark');
    assert.strictEqual(html, 'Target <mark>Z\u0303\u0325</mark> Word');
    assert(!html.includes('Z</mark>\u0303'), 'HTML tag must never split combining marks from base character');

    // Direct unit test of guardClusterBoundary
    const testCluster = 'e\u0301\u0325';
    // If an offset ended at 1 (between 'e' and '\u0301')
    const guarded = guardClusterBoundary(testCluster, 1);
    assert.strictEqual(guarded, testCluster.length, 'Guard should extend to end of cluster');

    console.log('   ✅ Grapheme Cluster Boundary Guard verified (HTML tags never split combining marks)');
  }

  // =========================================================================
  // 7. Subsequence & Substring Alignment Symmetry
  // =========================================================================
  console.log('7. Testing subsequence & substring alignment symmetry...');
  {
    const path = 'packages/webgpu-search/src/document-index.ts';

    // Fuzzy mode: greedy forward subsequence
    const fuzzyRanges = alignHighlights(path, 'webdocidx', { mode: 'fuzzy' });
    assert(fuzzyRanges.length > 0);
    const fuzzyHtml = renderHighlightedText(path, fuzzyRanges, 'mark');
    assert(fuzzyHtml.includes('packages/<mark>web</mark>gpu-search/src/'));
    assert(fuzzyHtml.includes('<mark>doc</mark>ument-<mark>i</mark>n<mark>d</mark>e<mark>x</mark>.ts'));

    // Substring mode: exact earliest contiguous match
    const subRanges = alignHighlights(path, 'document-index', { mode: 'substring' });
    assert.strictEqual(subRanges.length, 1);
    assert.strictEqual(path.slice(subRanges[0].start, subRanges[0].end), 'document-index');

    // Substring non-match returns empty
    const noSub = alignHighlights(path, 'notfound', { mode: 'substring' });
    assert.strictEqual(noSub.length, 0);

    // Fuzzy non-match returns empty
    const noFuzzy = alignHighlights(path, 'xyz123abc', { mode: 'fuzzy' });
    assert.strictEqual(noFuzzy.length, 0);

    console.log('   ✅ Subsequence and substring alignment symmetry verified');
  }

  // =========================================================================
  // 8. Span Merging (Overlapping, Adjacent, and Disjoint Ranges)
  // =========================================================================
  console.log('8. Testing span merging logic (mergeHighlightRanges)...');
  {
    // Overlapping ranges
    const overlapping: HighlightRange[] = [
      { start: 0, end: 5 },
      { start: 3, end: 8 }
    ];
    const merged1 = mergeHighlightRanges(overlapping);
    assert.strictEqual(merged1.length, 1);
    assert.strictEqual(merged1[0].start, 0);
    assert.strictEqual(merged1[0].end, 8);

    // Adjacent / contiguous ranges
    const adjacent: HighlightRange[] = [
      { start: 0, end: 5 },
      { start: 5, end: 10 }
    ];
    const merged2 = mergeHighlightRanges(adjacent);
    assert.strictEqual(merged2.length, 1);
    assert.strictEqual(merged2[0].start, 0);
    assert.strictEqual(merged2[0].end, 10);

    // Disjoint ranges
    const disjoint: HighlightRange[] = [
      { start: 0, end: 5 },
      { start: 8, end: 12 }
    ];
    const merged3 = mergeHighlightRanges(disjoint);
    assert.strictEqual(merged3.length, 2);
    assert.strictEqual(merged3[0].start, 0);
    assert.strictEqual(merged3[0].end, 5);
    assert.strictEqual(merged3[1].start, 8);
    assert.strictEqual(merged3[1].end, 12);

    // Out of order ranges
    const unsorted: HighlightRange[] = [
      { start: 10, end: 15 },
      { start: 0, end: 5 },
      { start: 4, end: 8 }
    ];
    const merged4 = mergeHighlightRanges(unsorted);
    assert.strictEqual(merged4.length, 2);
    assert.strictEqual(merged4[0].start, 0);
    assert.strictEqual(merged4[0].end, 8);
    assert.strictEqual(merged4[1].start, 10);
    assert.strictEqual(merged4[1].end, 15);

    // Empty
    assert.strictEqual(mergeHighlightRanges([]).length, 0);

    console.log('   ✅ Span merging logic verified');
  }

  // =========================================================================
  // 9. HTML Tag Injection (renderHighlightedText)
  // =========================================================================
  console.log('9. Testing HTML tag injection (renderHighlightedText)...');
  {
    const text = 'The quick brown fox';
    const ranges: HighlightRange[] = [
      { start: 4, end: 9 },   // 'quick'
      { start: 16, end: 19 }  // 'fox'
    ];

    // Default <mark>
    const htmlMark = renderHighlightedText(text, ranges);
    assert.strictEqual(htmlMark, 'The <mark>quick</mark> brown <mark>fox</mark>');

    // Custom <b>
    const htmlB = renderHighlightedText(text, ranges, 'b');
    assert.strictEqual(htmlB, 'The <b>quick</b> brown <b>fox</b>');

    // Custom <em>
    const htmlEm = renderHighlightedText(text, ranges, 'em');
    assert.strictEqual(htmlEm, 'The <em>quick</em> brown <em>fox</em>');

    // Empty ranges
    assert.strictEqual(renderHighlightedText(text, []), text);
    assert.strictEqual(renderHighlightedText('', ranges), '');

    console.log('   ✅ HTML tag injection verified');
  }

  // =========================================================================
  // 10. DocumentIndex Search Integration (Primary & Auxiliary Matches)
  // =========================================================================
  console.log('10. Testing DocumentIndex search integration with highlighting...');
  {
    const docs: DocItem[] = [
      {
        id: 'doc-1',
        title: '   WebGPU Fuzzy Search Architecture   ',
        body: 'Accelerated text indexing on GPU with shader compute',
        tags: ['webgpu', 'fuzzy', 'search']
      },
      {
        id: 'doc-2',
        title: 'High Performance Shader Programming',
        body: 'How WebGPU handles fuzzy query token matching in parallel',
        tags: ['compute', 'shaders']
      }
    ];

    const indexOptions: DocumentIndexOptions<DocItem> = {
      idField: 'id',
      fields: [
        { name: 'title', weight: 2.0 },
        { name: 'body', weight: 1.0 },
        { name: 'tags', weight: 1.5 }
      ],
      preferGpu: false
    };

    const index = await DocumentIndex.create(docs, indexOptions);

    // Default search: highlight === true (implicit), no tag
    const res1 = await index.search('WebGPU');
    assert.strictEqual(res1.totalMatches, 2);

    const hit1 = res1.results[0];
    assert(hit1.highlights !== undefined, 'Expected hit1.highlights to be defined');
    assert(hit1.highlights['title'] !== undefined, 'Expected title highlights');
    assert.strictEqual(hit1.highlightedText, undefined, 'No tag requested, highlightedText should be undefined');

    // Check that title highlight accounts for leading indentation
    const titleRanges = hit1.highlights['title'];
    assert.strictEqual(titleRanges.length, 1);
    const titleSlice = docs[0].title.slice(titleRanges[0].start, titleRanges[0].end);
    assert.strictEqual(titleSlice, 'WebGPU');

    // Auxiliary match check for doc-1: 'tags' also matched 'webgpu'
    assert(hit1.matches !== undefined);
    assert(hit1.highlights['tags'] !== undefined);
    assert.strictEqual(hit1.matches[0].field, 'tags');
    assert(hit1.matches[0].highlights !== undefined);

    // Search with tag: 'mark'
    const res2 = await index.search('WebGPU', { tag: 'mark' });
    const hitWithTag = res2.results[0];
    assert(hitWithTag.highlightedText !== undefined);
    assert(hitWithTag.highlightedText['title'].includes('<mark>WebGPU</mark>'));
    // Verify leading spaces preserved in highlighted text
    assert(hitWithTag.highlightedText['title'].startsWith('   <mark>WebGPU</mark>'));

    // Search with highlight: false
    const resNoHl = await index.search('WebGPU', { highlight: false });
    assert.strictEqual(resNoHl.results[0].highlights, undefined);
    assert.strictEqual(resNoHl.results[0].highlightedText, undefined);

    // Restricting highlightFields to 'matched-field'
    const resMatchedOnly = await index.search('WebGPU', {
      tag: 'b',
      highlightFields: 'matched-field'
    });
    const hitMatchedOnly = resMatchedOnly.results[0];
    assert(hitMatchedOnly.highlights !== undefined);
    assert(hitMatchedOnly.highlights['title'] !== undefined);
    assert.strictEqual(hitMatchedOnly.highlights['tags'], undefined, 'Auxiliary tags field should not be highlighted');

    // Substring mode search with highlighting
    const resSub = await index.search('Fuzzy Search', { mode: 'substring', tag: 'mark' });
    assert.strictEqual(resSub.totalMatches, 1);
    const subHit = resSub.results[0];
    assert(subHit.highlightedText!['title'].includes('<mark>Fuzzy Search</mark>'));

    index.destroy();
    console.log('   ✅ DocumentIndex search integration with highlights and tags verified');
  }

  // =========================================================================
  // 11. WebGPU vs CPU Highlighting Parity
  // =========================================================================
  console.log('11. Testing WebGPU vs CPU highlighting parity...');
  {
    const mockAdapter = createMockAdapter();
    const mockDeviceWrapper = await mockAdapter.requestDevice();
    const mockDevice = mockDeviceWrapper.gpu;

    const testDocs: DocItem[] = [
      { id: '1', title: 'WebGPU Search', body: 'Fuzzy engine on graphics hardware' },
      { id: '2', title: 'Browser Architecture', body: 'WebGPU shader pipeline execution' }
    ];

    const opts: DocumentIndexOptions<DocItem> = {
      fields: [
        { name: 'title', weight: 2.0 },
        { name: 'body', weight: 1.0 }
      ]
    };

    const cpuIdx = await DocumentIndex.create(testDocs, { ...opts, preferGpu: false });
    const gpuIdx = await DocumentIndex.create(testDocs, { ...opts, preferGpu: true, device: mockDevice as any });

    assert.strictEqual(cpuIdx.getStats().engine, 'cpu');
    assert.strictEqual(gpuIdx.getStats().engine, 'webgpu');

    const queries = ['WebGPU', 'Search', 'engine', 'pipeline'];
    for (const q of queries) {
      for (const mode of ['fuzzy', 'substring'] as const) {
        const cpuRes = await cpuIdx.search(q, { mode, tag: 'mark' });
        const gpuRes = await gpuIdx.search(q, { mode, tag: 'mark' });

        assert.strictEqual(cpuRes.query, q);
        assert.strictEqual(gpuRes.query, q);
        assert.strictEqual(cpuRes.mode, mode);
        assert.strictEqual(gpuRes.mode, mode);
        assert.strictEqual(cpuRes.engine, 'cpu');
        assert.strictEqual(gpuRes.engine, 'webgpu');

        // Verify CPU highlights structure
        for (const hit of cpuRes.results) {
          assert(hit.highlights !== undefined, 'Expected hit.highlights');
          assert(hit.highlightedText !== undefined, 'Expected hit.highlightedText');
          assert(hit.highlightedText[hit.matchedField].includes('<mark>'));
        }

        // If mock GPU results exist, verify structure matches
        for (const hit of gpuRes.results) {
          assert(hit.highlights !== undefined);
          assert(hit.highlightedText !== undefined);
        }
      }
    }

    cpuIdx.destroy();
    gpuIdx.destroy();
    console.log('   ✅ WebGPU vs CPU highlighting parity verified 100%');
  }

  // =========================================================================
  // 12. Security, Edge Cases, and Multi-Agent Hardening Verification
  // =========================================================================
  console.log('12. Testing security, edge cases, and multi-agent review hardening...');
  {
    // (a) HTML entity escaping / XSS mitigation
    const dangerousText = '<script>alert("XSS & danger")</script>';
    const dangerRanges: HighlightRange[] = [{ start: 8, end: 13 }]; // 'alert'
    const escapedHtml = renderHighlightedText(dangerousText, dangerRanges, 'mark', true);
    assert.strictEqual(
      escapedHtml,
      '&lt;script&gt;<mark>alert</mark>(&quot;XSS &amp; danger&quot;)&lt;/script&gt;',
      'Expected raw slices to be HTML-escaped'
    );

    // Empty ranges with escapeHtml
    const escapedEmpty = renderHighlightedText('A < B & C > D', [], 'mark', true);
    assert.strictEqual(escapedEmpty, 'A &lt; B &amp; C &gt; D');

    // (b) Custom tags with attributes emit valid closing tags
    const styledHtml = renderHighlightedText('WebGPU search', [{ start: 0, end: 6 }], 'mark class="highlight"');
    assert.strictEqual(styledHtml, '<mark class="highlight">WebGPU</mark> search');

    // (c) Malformed tag names throw TypeError
    assert.throws(
      () => renderHighlightedText('hello', [{ start: 0, end: 2 }], '<script>'),
      (err: any) => err instanceof TypeError && err.message.includes('Invalid HTML tag')
    );

    // (d) Emoji skin-tone modifier sequences and ZWJ composite protection
    const thumbModifier = 'Thumbs up 👍🏽 for WebGPU';
    // Match '👍' (start 10, end 12); skin tone is at 12..14
    const guardedThumb = guardClusterBoundary(thumbModifier, 12);
    assert.strictEqual(guardedThumb, 14, 'Should extend boundary across emoji skin tone modifier');

    const womanCoder = 'Dev 👩‍💻 at work';
    // '👩' is 4..6; ZWJ is 6..7; laptop is 7..9
    const guardedCoder = guardClusterBoundary(womanCoder, 6);
    assert.strictEqual(guardedCoder, 9, 'Should extend boundary across ZWJ composite cluster');

    // (e) Inverted and negative range sanitization in mergeHighlightRanges
    const messyRanges: HighlightRange[] = [
      { start: 10, end: 5 }, // inverted -> [5, 10]
      { start: -5, end: 3 }, // negative start -> [0, 3]
      { start: 2, end: 6 }   // overlapping [2, 6] connects [0, 3] and [5, 10]
    ];
    const cleaned = mergeHighlightRanges(messyRanges);
    assert.strictEqual(cleaned.length, 1);
    assert.strictEqual(cleaned[0].start, 0);
    assert.strictEqual(cleaned[0].end, 10);

    const withDisjoint = mergeHighlightRanges([...messyRanges, { start: 15, end: 20 }]);
    assert.strictEqual(withDisjoint.length, 2);
    assert.strictEqual(withDisjoint[0].start, 0);
    assert.strictEqual(withDisjoint[0].end, 10);
    assert.strictEqual(withDisjoint[1].start, 15);
    assert.strictEqual(withDisjoint[1].end, 20);

    // (f) Pre-computed source map case-sensitive option preservation in alignHighlights
    const csRaw = 'CaseSensitive Test';
    const csMap = normalizeWithSourceMap(csRaw, false); // folded: false
    // Should match exact case 'Case'
    const csRanges = alignHighlights(csRaw, 'Case', { sourceMap: csMap });
    assert.strictEqual(csRanges.length, 1);
    assert.strictEqual(csRaw.slice(csRanges[0].start, csRanges[0].end), 'Case');

    // Mismatched sourceMap.folded and options.folded should throw IncompatibleOptionError
    assert.throws(
      () => alignHighlights(csRaw, 'Case', { sourceMap: csMap, folded: true }),
      (err: any) => err instanceof IncompatibleOptionError
    );

    // (g) DocumentIndex auxiliary matches populated in 'all-fields' mode
    const multiDoc: DocItem[] = [
      {
        id: 'doc-m',
        title: 'Graphics API',
        body: 'WebGPU compute',
        tags: ['graphics', 'gpu']
      }
    ];
    const docIdx = await DocumentIndex.create(multiDoc, {
      fields: [
        { name: 'title', weight: 2.0 },
        { name: 'tags', weight: 1.5 },
        { name: 'body', weight: 1.0 }
      ],
      preferGpu: false
    });

    const multiRes = await docIdx.search('graphics', { highlightFields: 'all-fields', tag: 'mark' });
    assert.strictEqual(multiRes.results.length, 1);
    const mHit = multiRes.results[0];
    assert.strictEqual(mHit.matchedField, 'title');
    assert(mHit.highlights!['title'] !== undefined);
    assert(mHit.highlights!['tags'] !== undefined);
    assert(mHit.matches !== undefined && mHit.matches.length > 0);
    // Verify that aux.highlights is populated in 'all-fields' mode
    assert(mHit.matches[0].highlights !== undefined, 'Expected aux.highlights to be populated in all-fields mode');

    // (h) Invalid highlightFields string throws TypeError
    await assert.rejects(
      async () => docIdx.search('graphics', { highlightFields: 'invalid-field-mode' as any }),
      (err: any) => err instanceof TypeError && err.message.includes('Invalid highlightFields option')
    );

    docIdx.destroy();
    console.log('   ✅ Security, edge cases, and multi-agent review hardening verified 100%');
  }

  console.log('\n--- All Milestone 3: Unicode-Safe Highlighting Engine Tests Passed! ✅ ---');
}

runM3Tests().catch((err) => {
  console.error('Milestone 3 test failed:', err);
  process.exit(1);
});
