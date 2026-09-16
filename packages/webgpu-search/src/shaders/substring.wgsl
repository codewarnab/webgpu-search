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
@group(0) @binding(1) var<storage, read> offsets: array<u32>;
@group(0) @binding(2) var<storage, read> records: array<u32>;
@group(0) @binding(3) var<storage, read_write> output: OutputBuffer;

fn to_lower(c: u32) -> u32 {
    if (c >= 65u && c <= 90u) {
        return c + 32u;
    }
    return c;
}

fn get_char(byte_idx: u32) -> u32 {
    let word_idx = byte_idx >> 2u;
    let shift = (byte_idx & 3u) * 8u;
    return (records[word_idx] >> shift) & 0xFFu;
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

    let start_byte = offsets[row_id];
    let end_byte = offsets[row_id + 1u];
    let str_len = end_byte - start_byte;
    if (str_len < query_len) {
        return;
    }

    let max_start = str_len - query_len;
    let is_case_sens = uniforms.case_sensitive == 1u;
    var matched = false;
    var match_start = 0u;

    for (var start = 0u; start <= max_start; start++) {
        var sub_match = true;
        for (var j = 0u; j < query_len; j++) {
            var sc = get_char(start_byte + start + j);
            var qc = get_query_char(j);
            if (!is_case_sens) {
                sc = to_lower(sc);
                qc = to_lower(qc);
            }
            if (sc != qc) {
                sub_match = false;
                break;
            }
        }
        if (sub_match) {
            matched = true;
            match_start = start;
            break;
        }
    }

    if (matched) {
        let out_idx = atomicAdd(&output.count, 1u);
        if (out_idx < uniforms.max_candidates) {
            let score = 1000i - i32(match_start * 10u) - i32(str_len - query_len);
            output.results[out_idx] = MatchResult(row_id, score);
        }
    }
}
