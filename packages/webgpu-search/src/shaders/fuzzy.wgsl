struct QueryUniforms { total_rows: u32, query_len: u32, max_candidates: u32, flagsAndProfile: u32, _pad: vec4<u32>, };
struct Match { index: u32, score: i32, };
struct OutBuf { count: atomic<u32>, _pad0: u32, results: array<Match>, };
@group(0) @binding(0) var<uniform> uni: QueryUniforms;
@group(0) @binding(1) var<storage, read> off: array<u32>;
@group(0) @binding(2) var<storage, read> rec: array<u32>;
@group(0) @binding(3) var<storage, read> query: array<u32>;
@group(0) @binding(4) var<storage, read_write> out: OutBuf;
fn is_wb(prev_char: u32) -> bool {
  return prev_char == 47u || prev_char == 95u || prev_char == 45u ||
     prev_char == 46u || prev_char == 32u || prev_char == 58u || prev_char == 92u;
}
@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let rid = uni._pad.x + gid.x;
  if (rid >= uni.total_rows) {
    return;
  }
  let ql = uni.query_len;
  if (ql == 0u) {
    return;
  }
  let t0 = off[rid];
  let t1 = off[rid + 1u];
  let str_len = t1 - t0;
  if (str_len < ql) {
    return;
  }
  var q = 0u;
  var score = 100i;
  var run = 0i;
  var first = -1i;
  var last = 0i;
  for (var i = 0u; i < str_len; i++) {
    let sc = rec[t0 + i];
    let qc = query[q];
    if (sc == qc) {
      if (first < 0i) {
        first = i32(i);
        if (i == 0u) {
          score += 40;
        }
      }
      last = i32(i);
      if (i > 0u) {
        let prev = rec[t0 + i - 1u];
        if (is_wb(prev)) {
          score += 30;
        }
      }
      score += 15 + (run * 10);
      run += 1;
      q += 1u;
      if (q == ql) {
        break;
      }
    } else {
      run = 0;
    }
  }
  if (q == ql) {
    let span = last - first + 1i;
    score -= (span - i32(ql)) * 2;
    score -= i32(str_len) - i32(ql);
    let oi = atomicAdd(&out.count, 1u);
    if (oi < uni.max_candidates) {
      out.results[oi] = Match(rid, score);
    }
  }
}
