struct QueryUniforms {
    total_rows: u32,
    query_len: u32,
    max_candidates: u32,
    case_sensitive: u32,
    query_chars: array<vec4<u32>, 16>, // 64 characters (16 vec4s)
};

struct MatchResult {
    index: u32,
    score: i32,
};

struct OutputBuffer {
    count: atomic<u32>,
    _pad0: u32,
    results: array<MatchResult>,
};

@group(0) @binding(0) var<uniform> uniforms: QueryUniforms;
@group(0) @binding(1) var<storage, read> records: array<u32>;
@group(0) @binding(2) var<storage, read_write> output: OutputBuffer;

fn to_lower(c: u32) -> u32 {
    if (c >= 65u && c <= 90u) {
        return c + 32u;
    }
    return c;
}

fn get_char(row_id: u32, pos: u32) -> u32 {
    let word_idx = 1u + (pos >> 2u);
    let shift = (pos & 3u) * 8u;
    return (records[row_id * 16u + word_idx] >> shift) & 0xFFu;
}

fn get_query_char(pos: u32) -> u32 {
    let vec_idx = pos >> 2u;
    let comp_idx = pos & 3u;
    let v = uniforms.query_chars[vec_idx];
    if (comp_idx == 0u) { return v.x; }
    if (comp_idx == 1u) { return v.y; }
    if (comp_idx == 2u) { return v.z; }
    return v.w;
}

fn is_word_boundary(prev_char: u32) -> bool {
    return prev_char == 47u || prev_char == 95u || prev_char == 45u ||
           prev_char == 46u || prev_char == 32u || prev_char == 58u || prev_char == 92u;
}

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row_id = global_id.x;
    if (row_id >= uniforms.total_rows) {
        return;
    }

    let query_len = uniforms.query_len;
    if (query_len == 0u) {
        return;
    }

    let str_len = records[row_id * 16u];
    if (str_len < query_len) {
        return;
    }

    let is_case_sens = uniforms.case_sensitive == 1u;
    var q_idx = 0u;
    var score = 100i;
    var consecutive = 0i;
    var first_match_pos = -1i;
    var last_match_pos = 0i;

    for (var i = 0u; i < str_len; i++) {
        var sc = get_char(row_id, i);
        var qc = get_query_char(q_idx);
        
        if (!is_case_sens) {
            sc = to_lower(sc);
            qc = to_lower(qc);
        }

        if (sc == qc) {
            if (first_match_pos < 0i) {
                first_match_pos = i32(i);
                if (i == 0u) {
                    score += 40;
                }
            }
            last_match_pos = i32(i);

            if (i > 0u) {
                let prev = get_char(row_id, i - 1u);
                if (is_word_boundary(prev)) {
                    score += 30;
                }
            }

            score += 15 + (consecutive * 10);
            consecutive += 1;

            q_idx += 1u;
            if (q_idx == query_len) {
                break;
            }
        } else {
            consecutive = 0;
        }
    }

    if (q_idx == query_len) {
        let span = last_match_pos - first_match_pos + 1i;
        score -= (span - i32(query_len)) * 2;
        score -= i32(str_len - query_len);

        let out_idx = atomicAdd(&output.count, 1u);
        if (out_idx < uniforms.max_candidates) {
            output.results[out_idx] = MatchResult(row_id, score);
        }
    }
}
