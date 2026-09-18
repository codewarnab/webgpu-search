struct QueryUniforms { total_rows: u32, query_len: u32, max_candidates: u32, flagsAndProfile: u32, _pad: vec4<u32>, };
struct Match { index: u32, score: i32, };
struct OutBuf { count: atomic<u32>, _pad0: u32, results: array<Match>, };
@group(0) @binding(0) var<uniform> uni: QueryUniforms;
@group(0) @binding(1) var<storage, read> off: array<u32>;
@group(0) @binding(2) var<storage, read> rec: array<u32>;
@group(0) @binding(3) var<storage, read> query: array<u32>;
@group(0) @binding(4) var<storage, read_write> out: OutBuf;
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
  let maxstart = str_len - ql;
  var ok = false;
  var pos = 0u;
  for (var start = 0u; start <= maxstart; start++) {
    var hit = true;
    for (var j = 0u; j < ql; j++) {
      let sc = rec[t0 + start + j];
      let qc = query[j];
      if (sc != qc) {
        hit = false;
        break;
      }
    }
    if (hit) {
      ok = true;
      pos = start;
      break;
    }
  }
  if (ok) {
    let oi = atomicAdd(&out.count, 1u);
    if (oi < uni.max_candidates) {
      let score = 1000i - i32(pos * 10u) - i32(str_len - ql);
      out.results[oi] = Match(rid, score);
    }
  }
}
